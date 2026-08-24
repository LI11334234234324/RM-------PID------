"""
===============================================================================
RoboMaster 电控入门仿真项目 - 麦克纳姆轮底盘运动学解算模块
===============================================================================
本模块对应实车上的什么：
- 实车硬件/软件对应：STM32 底盘控制任务 (如 chassis_task.c 或 chassis_behaviour.c)
- 实车数据流：
  1. 逆运动学 (IK): 接收来自遥控器/上层决策的底盘整体期望速度 (vx, vy, omega)，
     解算出 4 个大疆 M3508 电机的目标转速 (rad/s 或 rpm)，随后交由 CAN 总线发送给电调。
  2. 正运动学 (FK): 接收 4 个电机编码器反馈回来的实际转速，解算出底盘当前在机体坐标系下的
     实际运动速度 (vx, vy, omega)，供底盘里程计 (Odometry) 和云台跟随环做状态估计。
===============================================================================
"""

import numpy as np
from typing import Tuple, Union, List


class MecanumKinematics:
    """
    四轮标准 O 型麦克纳姆轮底盘运动学解算器
    
    底盘坐标系定义 (右手系，标准机器人学与 RoboMaster 官方规范):
      - X 轴 (前方): 正向朝前 (Forward)
      - Y 轴 (左方): 正向朝左 (Left)
      - Z 轴 (垂直向上): 绕 Z 轴逆时针旋转 (CCW, 俯视) 为正 (omega > 0)
      
    四轮编号与物理排布 (O 型麦轮，辊子轴线斜向 45 度):
      - 轮 0 (FL, Front-Left,  左前轮): 坐标 (+L_x, +L_y), 辊子倾角 +45°
      - 轮 1 (FR, Front-Right, 右前轮): 坐标 (+L_x, -L_y), 辊子倾角 -45°
      - 轮 2 (RL, Rear-Left,   左后轮): 坐标 (-L_x, +L_y), 辊子倾角 -45°
      - 轮 3 (RR, Rear-Right,  右后轮): 坐标 (-L_x, -L_y), 辊子倾角 +45°
    """

    def __init__(
        self,
        wheel_base: float = 0.40,      # 前后轴距 2 * L_x (米)，默认 400 mm (典型步兵底盘)
        track_width: float = 0.40,     # 左右轮距 2 * L_y (米)，默认 400 mm
        wheel_radius: float = 0.076,   # 麦克纳姆轮半径 R (米)，标准 152 mm 麦轮半径 76 mm
        max_wheel_speed: float = 30.0  # 单轮最大角速度 (rad/s)，对应 M3508 额定减速比后输出
    ):
        """
        初始化麦轮底盘几何参数与动力学约束
        """
        self.wheel_base = float(wheel_base)
        self.track_width = float(track_width)
        self.wheel_radius = float(wheel_radius)
        self.max_wheel_speed = float(max_wheel_speed)

        # 几何半距离: L_x 为中心到前后轴距离，L_y 为中心到左右轮中心距离
        self.lx = self.wheel_base / 2.0
        self.ly = self.track_width / 2.0
        self.l_sum = self.lx + self.ly

        # 逆运动学变换矩阵 H_ik (4x3): [w_fl, w_fr, w_rl, w_rr]^T = (1/R) * H_ik * [vx, vy, omega]^T
        # 线速度形式:
        # v_fl = vx - vy - (lx + ly) * omega
        # v_fr = vx + vy + (lx + ly) * omega
        # v_rl = vx + vy - (lx + ly) * omega
        # v_rr = vx - vy + (lx + ly) * omega
        self.H_ik = np.array([
            [1.0, -1.0, -self.l_sum],
            [1.0,  1.0,  self.l_sum],
            [1.0,  1.0, -self.l_sum],
            [1.0, -1.0,  self.l_sum]
        ], dtype=np.float64)

        # 正运动学变换矩阵 H_fk (3x4)，由 H_ik 的伪逆矩阵 (H_ik^+ = (H^T H)^-1 H^T) 推导得出
        # [vx, vy, omega]^T = (R / 4) * [
        #   [ 1,  1,  1,  1],
        #   [-1,  1,  1, -1],
        #   [-1/L, 1/L, -1/L, 1/L]
        # ] * [w_fl, w_fr, w_rl, w_rr]^T
        self.H_fk = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [-1.0 / self.l_sum, 1.0 / self.l_sum, -1.0 / self.l_sum, 1.0 / self.l_sum]
        ], dtype=np.float64) * 0.25

    def inverse_kinematics(
        self,
        vx: float,
        vy: float,
        omega: float,
        scale_if_saturated: bool = True
    ) -> np.ndarray:
        """
        逆运动学解算 (IK): 机体速度 (vx, vy, omega) -> 四轮电机目标角速度 (rad/s)
        
        参数:
            vx: 机体 X 轴线速度期望 (m/s，前向为正)
            vy: 机体 Y 轴线速度期望 (m/s，左向为正)
            omega: 机体 Z 轴角速度期望 (rad/s，逆时针为正)
            scale_if_saturated: 若某轮转速超过上限，是否进行等比例衰减以保方向
            
        返回:
            np.ndarray: 四轮目标角速度 [w_fl, w_fr, w_rl, w_rr] (rad/s)
        """
        chassis_vel = np.array([vx, vy, omega], dtype=np.float64)
        linear_wheel_vels = self.H_ik @ chassis_vel  # 轮缘线速度 (m/s)
        wheel_angular_vels = linear_wheel_vels / self.wheel_radius  # 角速度 (rad/s)

        if scale_if_saturated:
            wheel_angular_vels = self.scale_wheel_speeds(wheel_angular_vels)

        return wheel_angular_vels

    def forward_kinematics(
        self,
        wheel_angular_vels: Union[np.ndarray, List[float]]
    ) -> Tuple[float, float, float]:
        """
        正运动学解算 (FK): 四轮电机实际角速度 (rad/s) -> 机体运动速度 (vx, vy, omega)
        
        参数:
            wheel_angular_vels: 四轮电机实际角速度 [w_fl, w_fr, w_rl, w_rr] (rad/s)
            
        返回:
            Tuple[float, float, float]: 机体速度 (vx (m/s), vy (m/s), omega (rad/s))
        """
        w = np.asarray(wheel_angular_vels, dtype=np.float64)
        linear_wheel_vels = w * self.wheel_radius
        chassis_vel = self.H_fk @ linear_wheel_vels
        return float(chassis_vel[0]), float(chassis_vel[1]), float(chassis_vel[2])

    def scale_wheel_speeds(
        self,
        wheel_angular_vels: np.ndarray,
        max_speed: float = None
    ) -> np.ndarray:
        """
        轮速等比例衰减 (Desaturation / Scaling):
        当由于 (vx, vy, omega) 叠加导致某一轮或多轮超出物理上限时，
        按最大超限倍数对全部 4 个车轮进行等比压缩，从而保证运动方向（航向与平移合成角）不畸变。
        """
        limit = self.max_wheel_speed if max_speed is None else float(max_speed)
        max_requested = np.max(np.abs(wheel_angular_vels))
        if max_requested > limit and max_requested > 1e-6:
            scale_factor = limit / max_requested
            return wheel_angular_vels * scale_factor
        return wheel_angular_vels.copy()

    @staticmethod
    def world_to_body_velocity(
        vx_world: float,
        vy_world: float,
        yaw: float
    ) -> Tuple[float, float]:
        """
        世界坐标系速度 (导航速度) 投影至机体坐标系 (控制速度)
        R_z(yaw)^T * [vx_w, vy_w]^T
        """
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        vx_body =  vx_world * cos_y + vy_world * sin_y
        vy_body = -vx_world * sin_y + vy_world * cos_y
        return float(vx_body), float(vy_body)

    @staticmethod
    def body_to_world_velocity(
        vx_body: float,
        vy_body: float,
        yaw: float
    ) -> Tuple[float, float]:
        """
        机体坐标系速度投影至世界坐标系
        R_z(yaw) * [vx_b, vy_b]^T
        """
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        vx_world = vx_body * cos_y - vy_body * sin_y
        vy_world = vx_body * sin_y + vy_body * cos_y
        return float(vx_world), float(vy_world)
