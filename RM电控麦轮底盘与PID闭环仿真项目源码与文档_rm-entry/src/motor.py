"""
===============================================================================
RoboMaster 电控入门仿真项目 - 一阶无刷直流电机模型与编码器模块
===============================================================================
本模块对应实车上的什么：
- 实车硬件/软件对应：大疆 RoboMaster M3508 / M2006 无刷直流减速电机 + C620 电调 + 14位磁编码器
- 实车数据流与控制链路：
  1. 输入控制量：STM32 通过 CAN 总线向 C620 电调发送的电流控制指令 (-16384 ~ +16384，对应电机输出力矩)，
     在基础运动学与速度环仿真中可等效为一阶电压/速度控制输入 u。
  2. 物理动态：电机转子惯量与定子电感呈现经典的一阶惯性环节响应 (典型时间常数 tau 约 30~70ms)。
  3. 反馈传感器：电调每隔 1ms 通过 CAN 回传反馈报文 (ID 0x201~0x204)，包含机械角度、转速 (RPM)、实际转矩电流。
===============================================================================
"""

import numpy as np
from typing import Optional


class DCMotorModel:
    """
    一阶直流无刷电机物理仿真模型 (First-Order DC Motor Model)
    
    传递函数:
        G(s) = Omega(s) / U(s) = K / (tau * s + 1)
        
    微分方程:
        tau * d(omega)/dt + omega = K * u(t) - load_loss
        
    离散化更新 (固定步长 dt = 5ms / 200Hz):
        alpha = exp(-dt / tau)
        omega[k+1] = alpha * omega[k] + (1 - alpha) * (K * u[k] - load_loss)
    """

    def __init__(
        self,
        tau: float = 0.050,               # 电机机电时间常数 tau (秒)，默认 50ms (贴近 M3508 负载响应)
        gain_k: float = 1.0,              # 静态增益 K (稳态输出与输入的比例系数)
        max_speed: float = 35.0,          # 最大物理角速度 (rad/s)，对应减速后约 330 RPM
        deadband: float = 0.0,            # 静摩擦死区阈值 (u < deadband 时电机不动作，默认 0 关闭)
        friction_factor: float = 0.0,     # 额外动摩擦阻力系数 (用于模拟车轮老化或负载不均扰动)
        enable_encoder_noise: bool = False,# 是否启用编码器离散量化与噪声
        encoder_cpr: int = 8192,          # 编码器单圈线数/分辨率 (M3508 内置 8192 线磁编码器)
        dt: float = 0.005                 # 仿真计算步长 dt (秒)，默认 5ms (200Hz)
    ):
        self.tau = float(tau)
        self.gain_k = float(gain_k)
        self.max_speed = float(max_speed)
        self.deadband = float(deadband)
        self.friction_factor = float(friction_factor)
        self.enable_encoder_noise = bool(enable_encoder_noise)
        self.encoder_cpr = int(encoder_cpr)
        self.dt = float(dt)

        # 内部状态量
        self.actual_speed = 0.0      # 连续物理角速度 (rad/s)
        self.actual_position = 0.0   # 连续累计机械位置 (rad)
        self.last_control = 0.0      # 上一步控制量

    def reset(self, initial_speed: float = 0.0, initial_position: float = 0.0) -> None:
        """重置电机内部状态"""
        self.actual_speed = float(initial_speed)
        self.actual_position = float(initial_position)
        self.last_control = 0.0

    def step(self, control_input: float, external_load_torque: float = 0.0) -> float:
        """
        单步仿真推进
        
        参数:
            control_input: 控制器输出指令 u (rad/s 或等效控制量)
            external_load_torque: 外界负载扰动力矩 (等效减速阻力)
            
        返回:
            float: 编码器测得的转速反馈值 (rad/s)
        """
        u = float(control_input)
        self.last_control = u

        # 1. 考虑死区特性 (Static Friction / Deadband)
        if abs(u) < self.deadband:
            effective_u = 0.0
        else:
            effective_u = u - np.sign(u) * self.deadband

        # 2. 考虑额外摩擦与外力阻力 (Friction & External Load)
        load_speed_loss = (self.friction_factor * self.actual_speed) + external_load_torque

        # 3. 严格精确一阶离散更新 (Exact Zero-Order Hold Discretization)
        # alpha = exp(-dt / tau)
        # omega_{k+1} = omega_k * alpha + (K * u - load_loss) * (1 - alpha)
        alpha = np.exp(-self.dt / self.tau)
        target_steady_speed = self.gain_k * effective_u - load_speed_loss
        self.actual_speed = self.actual_speed * alpha + target_steady_speed * (1.0 - alpha)

        # 4. 电机物理转速饱和钳位 (Physical Saturation)
        self.actual_speed = float(np.clip(self.actual_speed, -self.max_speed, self.max_speed))

        # 5. 机械角度积分
        self.actual_position += self.actual_speed * self.dt

        # 6. 编码器反馈采样 (含可选量化与测量噪声)
        measured_speed = self._sample_encoder_speed()
        return measured_speed

    def _sample_encoder_speed(self) -> float:
        """模拟 C620 磁编码器测速过程与量化效应"""
        if not self.enable_encoder_noise:
            return self.actual_speed

        # 编码器测速原理: M 法测速 (在采样周期 dt 内统计脉冲增量 delta_counts)
        # 单圈脉冲数 CPR = 8192
        exact_counts_delta = (self.actual_speed * self.dt) * (self.encoder_cpr / (2.0 * np.pi))
        quantized_counts = np.round(exact_counts_delta)
        quantized_speed = (quantized_counts / self.encoder_cpr) * (2.0 * np.pi) / self.dt

        # 叠加微弱白噪声 (高斯噪声)
        noise = np.random.normal(0.0, 0.02)
        return float(quantized_speed + noise)

    def get_state(self) -> dict:
        """获取电机当前物理状态"""
        return {
            "actual_speed": self.actual_speed,
            "actual_position": self.actual_position,
            "last_control": self.last_control
        }
