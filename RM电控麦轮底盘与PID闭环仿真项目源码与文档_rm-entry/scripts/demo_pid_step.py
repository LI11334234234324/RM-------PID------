"""
===============================================================================
RoboMaster 电控入门仿真项目 - 单电机 PID 阶跃响应对比演示 (demo_pid_step.py)
===============================================================================
演示目标：
1. 直观对比同一一阶电机模型在 5 种控制模式下的阶跃响应曲线：
   - 开环控制 (Open-Loop Step)
   - 纯 P 控制 (Proportional Only) -> 观察稳态误差 (Steady-State Error)
   - PI 控制 (无抗饱和, No Anti-Windup) -> 观察大误差/限幅下的积分饱和超调 (Windup Overshoot)
   - PI 控制 (含抗饱和, With Anti-Windup) -> 消除超调与静差，达到平稳无冲
   - PID 控制 (含微小微分 D 抑制) -> 进一步加速收敛
2. 自动生成高质量 PNG 分析图与 GIF 演示动图，保存至 output/ 目录。
===============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.motor import DCMotorModel
from src.pid import PIDController


def run_step_simulation():
    dt = 0.005  # 5ms (200Hz)
    t_total = 1.2 # 1.2s
    time_steps = int(t_total / dt)
    time_array = np.linspace(0, t_total, time_steps)

    target_speed = 20.0  # 目标转速 20.0 rad/s (阶跃信号)
    friction_factor = 0.15 # 引入 15% 动摩擦阻力，模拟实车轴承与地面负荷
    motor_tau = 0.050    # 50ms 时间常数
    max_output = 30.0    # 电机最大控制限幅

    # 5 组实验控制器配置 (图例采用清晰英文字符以保证跨平台字体兼容性)
    configs = [
        {
            "name": "Open-Loop",
            "legend": "Open-Loop (No Feedback)",
            "color": "#7f7f7f",
            "style": "--",
            "type": "open_loop",
            "controller": None
        },
        {
            "name": "Pure P",
            "legend": "Pure P (Kp=1.8, Steady-State Error)",
            "color": "#d62728",
            "style": "-.",
            "type": "closed_loop",
            "controller": PIDController(kp=1.8, ki=0.0, kd=0.0, max_output=max_output, dt=dt)
        },
        {
            "name": "PI without AW",
            "legend": "PI (No Anti-Windup, Large Overshoot)",
            "color": "#ff7f0e",
            "style": ":",
            "type": "closed_loop",
            "controller": PIDController(kp=2.2, ki=25.0, kd=0.0, max_output=max_output, max_integral=100.0, dt=dt, anti_windup=False)
        },
        {
            "name": "PI with AW",
            "legend": "PI (With Anti-Windup, Recommended)",
            "color": "#2ca02c",
            "style": "-",
            "type": "closed_loop",
            "controller": PIDController(kp=2.2, ki=25.0, kd=0.0, max_output=max_output, max_integral=20.0, dt=dt, anti_windup=True)
        },
        {
            "name": "Full PID",
            "legend": "PID (Kp=2.5, Ki=25, Kd=0.02)",
            "color": "#1f77b4",
            "style": "-",
            "type": "closed_loop",
            "controller": PIDController(kp=2.5, ki=25.0, kd=0.02, max_output=max_output, max_integral=20.0, dt=dt, anti_windup=True, derivative_filter_alpha=0.3)
        }
    ]

    results = {}

    for cfg in configs:
        motor = DCMotorModel(tau=motor_tau, gain_k=1.0, friction_factor=friction_factor, max_speed=35.0, dt=dt)
        speeds = []
        controls = []
        integrals = []

        ctrl = cfg["controller"]
        if ctrl is not None:
            ctrl.reset()

        for step_i in range(time_steps):
            current_speed = motor.actual_speed

            if cfg["type"] == "open_loop":
                u = target_speed  # 开环直接给目标值
                i_val = 0.0
            else:
                u = ctrl.update(target=target_speed, feedback=current_speed)
                i_val = ctrl.i_out

            motor.step(u)
            speeds.append(motor.actual_speed)
            controls.append(u)
            integrals.append(i_val)

        results[cfg["name"]] = {
            "speed": np.array(speeds),
            "control": np.array(controls),
            "integral": np.array(integrals),
            "color": cfg["color"],
            "style": cfg["style"],
            "legend": cfg["legend"]
        }

    return time_array, target_speed, results, configs


def save_static_plot(time_array, target_speed, results, configs, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)

    # 1. 速度响应对比
    ax1.axhline(target_speed, color='black', linestyle='--', linewidth=1.5, label='Target Setpoint (20 rad/s)')
    for cfg in configs:
        name = cfg["name"]
        res = results[name]
        ax1.plot(time_array, res["speed"], label=res["legend"], color=res["color"], linestyle=res["style"], linewidth=2.0)

    ax1.set_ylabel("Motor Speed (rad/s)", fontsize=11, fontweight='bold')
    ax1.set_title("RoboMaster DC Motor Speed Step Response Comparison (dt = 5ms)", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # 2. 控制器输出与饱和分析
    for cfg in configs:
        name = cfg["name"]
        res = results[name]
        ax2.plot(time_array, res["control"], label=res["legend"], color=res["color"], linestyle=res["style"], linewidth=1.8)

    ax2.axhline(30.0, color='red', linestyle=':', linewidth=1.2, alpha=0.7, label='Max Voltage/Control Limit (+30)')
    ax2.set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Control Input u", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] 阶跃响应对比静态图已保存至: {output_path}")


def save_animated_gif(time_array, target_speed, results, configs, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), dpi=120, sharex=True)

    ax1.set_xlim(0, time_array[-1])
    ax1.set_ylim(-2, 32)
    ax1.set_ylabel("Motor Speed (rad/s)", fontsize=10, fontweight='bold')
    ax1.set_title("RoboMaster Motor Step Response Dynamic Animation", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axhline(target_speed, color='black', linestyle='--', linewidth=1.5, label='Target (20 rad/s)')

    ax2.set_xlim(0, time_array[-1])
    ax2.set_ylim(-5, 40)
    ax2.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Control Input u", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    lines_speed = {}
    lines_ctrl = {}
    for cfg in configs:
        name = cfg["name"]
        res = results[name]
        line1, = ax1.plot([], [], label=res["legend"], color=res["color"], linestyle=res["style"], linewidth=2.0)
        line2, = ax2.plot([], [], label=res["legend"], color=res["color"], linestyle=res["style"], linewidth=1.8)
        lines_speed[name] = line1
        lines_ctrl[name] = line2

    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)

    frame_indices = list(range(0, len(time_array), 4))
    if frame_indices[-1] != len(time_array) - 1:
        frame_indices.append(len(time_array) - 1)

    def init():
        for line in lines_speed.values():
            line.set_data([], [])
        for line in lines_ctrl.values():
            line.set_data([], [])
        return list(lines_speed.values()) + list(lines_ctrl.values())

    def update(frame_idx):
        t_sub = time_array[:frame_idx+1]
        for cfg in configs:
            name = cfg["name"]
            res = results[name]
            lines_speed[name].set_data(t_sub, res["speed"][:frame_idx+1])
            lines_ctrl[name].set_data(t_sub, res["control"][:frame_idx+1])
        return list(lines_speed.values()) + list(lines_ctrl.values())

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, blit=True, interval=25)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='pillow', fps=25)
    plt.close()
    print(f"[OK] 阶跃响应动态 GIF 已保存至: {output_path}")


if __name__ == "__main__":
    t_arr, tgt, res, cfgs = run_step_simulation()
    png_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'pid_step_response.png'))
    gif_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'pid_step_response.gif'))
    save_static_plot(t_arr, tgt, res, cfgs, png_path)
    save_animated_gif(t_arr, tgt, res, cfgs, gif_path)
