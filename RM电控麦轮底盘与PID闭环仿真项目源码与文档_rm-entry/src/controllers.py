"""
===============================================================================
RoboMaster 电控入门仿真项目 - 串级底盘控制器 (速度内环 + 航向外环)
===============================================================================
本模块对应实车上的什么：
- 实车硬件/软件对应：STM32 底盘核心闭环控制任务 (chassis_control.c)
- 经典串级控制链路 (Cascaded Control Architecture):
  1. 航向外环 (Heading Outer Loop, P 控制器):
     - 输入：上层期望航向角 target_yaw 与陀螺仪测量航向 yaw_feedback。
     - 输出：底盘期望自转角速度 omega_target (rad/s)。
  2. 逆运动学解算 (IK):
     - 将平移速度期望 (vx, vy) 与自转期望 omega_target 分解为 4 个电机的目标转速 [w_fl, w_fr, w_rl, w_rr]。
  3. 轮速内环 (Wheel Speed Inner Loop, 4x PI 控制器):
     - 输入：单轮目标转速与电机编码器反馈转速之差。
     - 输出：给 4 个电机的控制电压/电流指令，驱动电机精准跟随目标转速。
- 调参经验：
  - 先调内环速度 PI，确保单轮响应快、无超调、无静差；
  - 再调外环航向 P，确保底盘能快速纠偏且不发生蛇形震荡。
===============================================================================
"""

import numpy as np
from typing import Tuple, List, Optional
from .kinematics import MecanumKinematics
from .pid import PIDController


def angle_difference(target_rad: float, current_rad: float) -> float:
    """计算两个角度之间的最短角位移差值 [-pi, pi]"""
    diff = (target_rad - current_rad + np.pi) % (2.0 * np.pi) - np.pi
    return float(diff)


class ChassisCascadeController:
    """
    四轮麦轮底盘串级控制器 (外环航向 P + 内环四轮速度 PI)
    """

    def __init__(
        self,
        kinematics: Optional[MecanumKinematics] = None,
        speed_kp: float = 1.6,            # 速度内环 Kp
        speed_ki: float = 12.0,           # 速度内环 Ki (积分消除负载稳态误差)
        speed_kd: float = 0.0,            # 速度内环 Kd (通常速度环不需要 D，避免高频噪声)
        heading_kp: float = 4.0,          # 航向外环 Kp (将航向误差映射为角速度指令)
        max_wheel_accel: float = 80.0,    # 电机最大输出加速度/限幅
        dt: float = 0.005                 # 控制周期 5ms (200Hz)
    ):
        self.dt = float(dt)
        self.kinematics = kinematics if kinematics is not None else MecanumKinematics()
        self.heading_kp = float(heading_kp)

        # 4 个独立轮速 PI 控制器
        self.wheel_pids = [
            PIDController(
                kp=speed_kp,
                ki=speed_ki,
                kd=speed_kd,
                max_output=self.kinematics.max_wheel_speed * 1.5,
                max_integral=self.kinematics.max_wheel_speed * 0.8,
                dt=dt,
                anti_windup=True
            )
            for _ in range(4)
        ]

    def reset(self) -> None:
        """重置所有 PID 控制器内部状态"""
        for pid in self.wheel_pids:
            pid.reset()

    def set_speed_gains(self, kp: float, ki: float, kd: float = 0.0) -> None:
        """动态修改速度内环 PID 增益"""
        for pid in self.wheel_pids:
            pid.kp = float(kp)
            pid.ki = float(ki)
            pid.kd = float(kd)

    def set_heading_gain(self, kp: float) -> None:
        """动态修改航向外环 P 增益"""
        self.heading_kp = float(kp)

    def compute_control(
        self,
        target_vx_body: float,
        target_vy_body: float,
        target_yaw: float,
        current_yaw: float,
        measured_wheel_speeds: np.ndarray,
        closed_loop: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算底盘控制输出
        
        参数:
            target_vx_body: 机体 X 轴期望线速度 (m/s)
            target_vy_body: 机体 Y 轴期望线速度 (m/s)
            target_yaw: 期望航向角 (rad)
            current_yaw: 当前航向测量角 (rad，来自陀螺仪)
            measured_wheel_speeds: 4 轮编码器反馈实际转速 [w_fl, w_fr, w_rl, w_rr] (rad/s)
            closed_loop: True 为闭环 PID 控制，False 为开环直接前馈 (用于对比展示)
            
        返回:
            Tuple[np.ndarray, np.ndarray, float]:
              1. motor_commands: 4 个电机的控制指令 [u_0, u_1, u_2, u_3]
              2. target_wheel_speeds: 4 个电机的目标转速 [w_target_0, ...]
              3. target_omega: 外环计算出的期望角速度 omega_cmd
        """
        # 1. 航向外环 P 控制：计算目标自转角速度
        yaw_err = angle_difference(target_yaw, current_yaw)
        target_omega = self.heading_kp * yaw_err
        # 对自转角速度做安全限幅
        target_omega = float(np.clip(target_omega, -6.0, 6.0))

        # 2. 逆运动学解算：分解为四轮目标线速度/角速度
        target_wheel_speeds = self.kinematics.inverse_kinematics(
            target_vx_body,
            target_vy_body,
            target_omega,
            scale_if_saturated=True
        )

        # 3. 速度内环控制
        if not closed_loop:
            # 开环模式：直接将逆运动学计算的期望转速送给电机，不经过 PID 纠偏
            return target_wheel_speeds.copy(), target_wheel_speeds, target_omega

        # 闭环模式：4 轮独立 PI 闭环纠偏
        motor_commands = np.zeros(4, dtype=np.float64)
        for i in range(4):
            motor_commands[i] = self.wheel_pids[i].update(
                target=target_wheel_speeds[i],
                feedback=measured_wheel_speeds[i]
            )

        return motor_commands, target_wheel_speeds, target_omega
