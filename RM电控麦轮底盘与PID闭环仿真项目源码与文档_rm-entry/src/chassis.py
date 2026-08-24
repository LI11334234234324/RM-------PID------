"""
===============================================================================
RoboMaster 电控入门仿真项目 - 麦轮底盘整机物理动力学与传感器仿真模块
===============================================================================
本模块对应实车上的什么：
- 实车硬件/软件对应：底盘机械实体 + 里程计估算任务 (chassis_odometry.c) + 陀螺仪 IMU (BMI088/MPU6050)
- 仿真功能定位：
  1. 动力学推进：将 4 个电机的物理转速通过麦轮正运动学方程解算为底盘实时质心线速度与自转角速度。
  2. 位姿积分：在世界坐标系下积分更新底盘真实坐标 (x, y) 与航向角 yaw (theta)。
  3. 真实环境扰动注入：
     - 轮子不对称摩擦/打滑 (Asymmetric Friction / Roller Slip)
     - 电池/云台重心偏心力矩 (Mass Eccentricity Yaw Disturbance)
     - 陀螺仪零偏漂移与测量噪声 (Gyro Bias & Noise)
===============================================================================
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .kinematics import MecanumKinematics
from .motor import DCMotorModel


class MecanumChassisSim:
    """
    四轮麦克纳姆轮底盘整机物理仿真环境
    """

    def __init__(
        self,
        kinematics: Optional[MecanumKinematics] = None,
        motor_tau: float = 0.050,         # 电机时间常数 50ms
        motor_gain: float = 1.0,          # 电机静态增益
        motor_max_speed: float = 35.0,    # 电机最大角速度
        dt: float = 0.005,                # 仿真步长 5ms (200Hz)
        enable_disturbances: bool = False # 是否开启不对称摩擦与偏心扰动
    ):
        self.dt = float(dt)
        self.kinematics = kinematics if kinematics is not None else MecanumKinematics()
        self.enable_disturbances = bool(enable_disturbances)

        # 4 个物理电机实例: 0:FL, 1:FR, 2:RL, 3:RR
        self.motors = [
            DCMotorModel(tau=motor_tau, gain_k=motor_gain, max_speed=motor_max_speed, dt=dt)
            for _ in range(4)
        ]

        # 真实世界物理状态 (Ground Truth Pose & Velocity in World Frame)
        self.x = 0.0           # 世界坐标 X (米)
        self.y = 0.0           # 世界坐标 Y (米)
        self.yaw = 0.0         # 世界航向角 (弧度, [-pi, pi])
        self.vx_body = 0.0     # 机体当前线速度 vx (m/s)
        self.vy_body = 0.0     # 机体当前线速度 vy (m/s)
        self.omega_body = 0.0  # 机体当前自转角速度 (rad/s)

        # 扰动参数配置 (开环时足以导致明显跑偏，闭环时展现强抗扰能力)
        self.wheel_friction_multipliers = np.array([1.0, 1.0, 1.0, 1.0])
        self.mass_offset_yaw_torque = 0.0
        self.gyro_drift_rate = 0.005      # 陀螺仪零偏漂移 (rad/s)
        self.simulated_gyro_yaw = 0.0     # 模拟陀螺仪积分角度

        if self.enable_disturbances:
            self.set_default_disturbances()

    def set_default_disturbances(self) -> None:
        """设置典型 RoboMaster 实车工况扰动 (如左侧磨损 15% + 电池偏心力矩)"""
        self.wheel_friction_multipliers = np.array([0.85, 1.00, 0.88, 1.00])
        self.mass_offset_yaw_torque = 0.12 # 模拟自转扰动力矩
        self.enable_disturbances = True

    def disable_disturbances(self) -> None:
        """关闭外界扰动"""
        self.wheel_friction_multipliers = np.array([1.0, 1.0, 1.0, 1.0])
        self.mass_offset_yaw_torque = 0.0
        self.enable_disturbances = False

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """重置底盘位姿与电机内部状态"""
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.simulated_gyro_yaw = float(yaw)
        self.vx_body = 0.0
        self.vy_body = 0.0
        self.omega_body = 0.0
        for m in self.motors:
            m.reset()

    def step(self, motor_controls: np.ndarray) -> Dict[str, Any]:
        """
        底盘单步物理仿真推进
        
        参数:
            motor_controls: 4 个电机的控制输入指令 [u_fl, u_fr, u_rl, u_rr]
            
        返回:
            Dict: 包含底盘最新真实状态与传感器观测值
        """
        controls = np.asarray(motor_controls, dtype=np.float64)

        # 1. 四轮独立电机推进 (含不对称摩擦扰动)
        actual_wheel_speeds = np.zeros(4, dtype=np.float64)
        measured_encoder_speeds = np.zeros(4, dtype=np.float64)

        for i, motor in enumerate(self.motors):
            # 引入各轮摩擦不对称导致的实际等效控制输入差异
            effective_u = controls[i] * self.wheel_friction_multipliers[i]
            measured_speed = motor.step(effective_u)
            actual_wheel_speeds[i] = motor.actual_speed
            measured_encoder_speeds[i] = measured_speed

        # 2. 通过正运动学解算底盘机体线速度与角速度
        vx, vy, omega = self.kinematics.forward_kinematics(actual_wheel_speeds)
        
        # 叠加质心偏心扰动力矩引起的角速度偏移
        if self.enable_disturbances:
            omega += self.mass_offset_yaw_torque

        self.vx_body = vx
        self.vy_body = vy
        self.omega_body = omega

        # 3. 将机体速度转换到世界坐标系并积分位姿
        vx_world, vy_world = self.kinematics.body_to_world_velocity(vx, vy, self.yaw)
        self.x += vx_world * self.dt
        self.y += vy_world * self.dt
        self.yaw += self.omega_body * self.dt
        
        # 航向角正规化到 [-pi, pi]
        self.yaw = (self.yaw + np.pi) % (2.0 * np.pi) - np.pi

        # 4. 模拟陀螺仪 IMU 数据采集 (含微弱零偏漂移与噪声)
        measured_gyro_rate = self.omega_body + (self.gyro_drift_rate if self.enable_disturbances else 0.0)
        self.simulated_gyro_yaw += measured_gyro_rate * self.dt
        self.simulated_gyro_yaw = (self.simulated_gyro_yaw + np.pi) % (2.0 * np.pi) - np.pi

        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "vx_body": self.vx_body,
            "vy_body": self.vy_body,
            "omega_body": self.omega_body,
            "actual_wheel_speeds": actual_wheel_speeds,
            "encoder_speeds": measured_encoder_speeds,
            "gyro_yaw": self.simulated_gyro_yaw,
            "gyro_rate": measured_gyro_rate
        }
