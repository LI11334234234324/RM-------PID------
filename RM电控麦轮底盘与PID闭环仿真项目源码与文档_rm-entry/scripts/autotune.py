"""
===============================================================================
RoboMaster 电控入门仿真项目 - Ziegler-Nichols (Z-N) 自动整定与参数计算工具
===============================================================================
本脚本功能：
1. 基于一阶惯性加纯滞后模型 (FOPDT: First Order Plus Dead Time):
   G(s) = K * exp(-L*s) / (tau * s + 1)
2. 利用 Ziegler-Nichols 阶跃响应法 (反应曲线法) 与临界比例法计算 P / PI / PID 初始参数。
3. 针对 RoboMaster 真实无刷电机 (如 M3508) 的工程经验，输出实车调参修正建议。
===============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.motor import DCMotorModel
from src.pid import PIDController


def estimate_fopdt_parameters(tau: float = 0.050, gain_k: float = 1.0, dead_time_L: float = 0.010):
    """
    计算并输出基于 Ziegler-Nichols 阶跃响应法 (Reaction Curve Method) 的初始 PID 参数
    
    Z-N 阶跃经验公式:
    ---------------------------------------------------------------
    控制器类型     Kp                Ti (积分时间)     Td (微分时间)
    ---------------------------------------------------------------
    P              tau / (K * L)     -                -
    PI             0.9 * tau / (K*L) 3.33 * L         -
    PID            1.2 * tau / (K*L) 2.0 * L          0.5 * L
    ---------------------------------------------------------------
    离散化换算 (采样周期 dt):
    Ki = Kp * dt / Ti
    Kd = Kp * Td / dt
    """
    R_slope = (gain_k / tau)  # 最大变化率斜率 R = K / tau
    
    # 经典 Z-N 连续域参数
    # P 控制
    kp_p = tau / (gain_k * dead_time_L)
    
    # PI 控制
    kp_pi = 0.9 * tau / (gain_k * dead_time_L)
    ti_pi = 3.33 * dead_time_L
    
    # PID 控制
    kp_pid = 1.2 * tau / (gain_k * dead_time_L)
    ti_pid = 2.0 * dead_time_L
    td_pid = 0.5 * dead_time_L

    dt = 0.005 # 200 Hz
    ki_pi = kp_pi * (dt / ti_pi)
    
    ki_pid = kp_pid * (dt / ti_pid)
    kd_pid = kp_pid * (td_pid / dt)

    print("=" * 70)
    print("      RoboMaster 电机 PID Ziegler-Nichols 自动整定分析报告")
    print("=" * 70)
    print(f"【被控对象参数】一阶时间常数 tau = {tau*1000:.1f} ms | 静态增益 K = {gain_k:.2f} | 等效纯滞后 L = {dead_time_L*1000:.1f} ms")
    print(f"【控制采样周期】dt = {dt*1000:.1f} ms (200 Hz)")
    print("-" * 70)
    print(f"1. 纯 P 控制器建议初值:")
    print(f"   Kp = {kp_p:.3f} | Ki = 0.000 | Kd = 0.000")
    print(f"   特点：响应快，但在外载或摩擦下存在不可消除的静差 (Steady-state error)。")
    print("-" * 70)
    print(f"2. PI 控制器建议初值 (RM 速度环强烈推荐):")
    print(f"   Kp = {kp_pi:.3f} | Ki_continuous = {kp_pi/ti_pi:.2f} (离散步进增益 Ki_step = {ki_pi:.4f})")
    print(f"   实车调参微调建议：建议 Kp 取 Z-N 理论值的 50%~70% (约 2.0~3.0)，Ki 取 10~25，防止机械共振。")
    print("-" * 70)
    print(f"3. PID 控制器建议初值:")
    print(f"   Kp = {kp_pid:.3f} | Ki_step = {ki_pid:.4f} | Kd_step = {kd_pid:.4f}")
    print(f"   注意：速度环一般不引入 Kd 或仅引入极微小 Kd (如 0.01~0.05) 并开启微分低通滤波，")
    print(f"   否则磁编码器量化噪声会被微分项剧烈放大，引起电机啸叫和发热！")
    print("=" * 70)

    return {
        "P": {"kp": kp_p, "ki": 0.0, "kd": 0.0},
        "PI": {"kp": kp_pi, "ki": kp_pi / ti_pi, "kd": 0.0},
        "PID": {"kp": kp_pid, "ki": kp_pid / ti_pid, "kd": kp_pid * td_pid}
    }


if __name__ == "__main__":
    estimate_fopdt_parameters()
