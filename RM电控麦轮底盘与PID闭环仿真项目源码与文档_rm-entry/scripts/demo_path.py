"""
===============================================================================
RoboMaster 电控入门仿真项目 - 底盘轨迹跟踪与开环/闭环抗扰对比 (demo_path.py)
===============================================================================
演示目标：
1. 轨迹跟踪算法验证：
   - 正方形轨迹 (Square Path, 2m x 2m 闭环折线巡航)
   - 8 字形轨迹 (Figure-8 / Lemniscate 连续曲率平滑机动)
2. 开环 vs 闭环抗扰核心对比：
   - 在底盘注入真实环境扰动 (左侧车轮摩擦衰减 15% + 电池/云台偏心自转力矩)
   - 开环模式 (Open-Loop): 轮速无闭环纠偏，底盘迅速跑偏失控，轨迹严重畸变。
   - 串级闭环模式 (Closed-Loop, 航向 P + 轮速 PI): 实时抵抗轮子打滑与偏心，轨迹高精度贴合参考路径。
3. 自动生成 output/ 目录下的对比分析图表与演示 GIF 动图。
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
import matplotlib.patches as patches

from src.kinematics import MecanumKinematics
from src.chassis import MecanumChassisSim
from src.controllers import ChassisCascadeController, angle_difference


# =============================================================================
# 1. 轨迹生成器 (Trajectory Generators)
# =============================================================================

def generate_square_trajectory(dt=0.005, side_length=2.0, speed=0.8):
    """生成 2m x 2m 正方形轨迹点与切线速度期望"""
    time_per_side = side_length / speed
    t_total = 4.0 * time_per_side
    steps = int(t_total / dt)
    
    t_arr = np.linspace(0, t_total, steps)
    ref_x = np.zeros(steps)
    ref_y = np.zeros(steps)
    ref_vx = np.zeros(steps)
    ref_vy = np.zeros(steps)
    ref_yaw = np.zeros(steps)
    
    for i, t in enumerate(t_arr):
        side = int(t / time_per_side) % 4
        sub_t = t - (int(t / time_per_side) * time_per_side)
        
        if side == 0:  # (0,0) -> (side_length, 0)
            ref_x[i] = sub_t * speed
            ref_y[i] = 0.0
            ref_vx[i] = speed
            ref_vy[i] = 0.0
            ref_yaw[i] = 0.0
        elif side == 1: # (side_length, 0) -> (side_length, side_length)
            ref_x[i] = side_length
            ref_y[i] = sub_t * speed
            ref_vx[i] = 0.0
            ref_vy[i] = speed
            ref_yaw[i] = np.pi / 2.0
        elif side == 2: # (side_length, side_length) -> (0, side_length)
            ref_x[i] = side_length - sub_t * speed
            ref_y[i] = side_length
            ref_vx[i] = -speed
            ref_vy[i] = 0.0
            ref_yaw[i] = np.pi
        else: # (0, side_length) -> (0, 0)
            ref_x[i] = 0.0
            ref_y[i] = side_length - sub_t * speed
            ref_vx[i] = 0.0
            ref_vy[i] = -speed
            ref_yaw[i] = -np.pi / 2.0
            
    return dt, steps, t_arr, ref_x, ref_y, ref_vx, ref_vy, ref_yaw


def generate_figure8_trajectory(dt=0.005, a=1.5, b=1.0, t_total=8.0):
    """生成 8 字形 (伯努利双纽线 / Lissajous) 平滑轨迹"""
    steps = int(t_total / dt)
    t_arr = np.linspace(0, t_total, steps)
    w0 = 2.0 * np.pi / t_total
    
    # x(t) = a * sin(w0 * t)
    # y(t) = b * sin(2 * w0 * t)
    ref_x = a * np.sin(w0 * t_arr)
    ref_y = b * np.sin(2.0 * w0 * t_arr)
    
    ref_vx = a * w0 * np.cos(w0 * t_arr)
    ref_vy = 2.0 * b * w0 * np.cos(2.0 * w0 * t_arr)
    
    ref_yaw = np.arctan2(ref_vy, ref_vx)
    return dt, steps, t_arr, ref_x, ref_y, ref_vx, ref_vy, ref_yaw


# =============================================================================
# 2. 轨迹跟踪仿真循环 (Simulation Runner)
# =============================================================================

def run_tracking_simulation(gen_func, closed_loop=True, enable_disturbances=True):
    dt, steps, t_arr, ref_x, ref_y, ref_vx, ref_vy, ref_yaw = gen_func()
    
    kin = MecanumKinematics(wheel_base=0.40, track_width=0.40, wheel_radius=0.076, max_wheel_speed=30.0)
    chassis = MecanumChassisSim(kinematics=kin, dt=dt, enable_disturbances=enable_disturbances)
    controller = ChassisCascadeController(kinematics=kin, speed_kp=2.2, speed_ki=20.0, heading_kp=4.5, dt=dt)
    
    chassis.reset(x=ref_x[0], y=ref_y[0], yaw=ref_yaw[0])
    
    hist_x, hist_y, hist_yaw = [], [], []
    hist_err_pos, hist_err_yaw = [], []
    
    # 位置前馈 + 比例位置校正 (P-Position Tracking)
    pos_kp = 2.5
    
    for i in range(steps):
        # 1. 计算世界系下的跟踪偏差
        ex_w = ref_x[i] - chassis.x
        ey_w = ref_y[i] - chassis.y
        eyaw = angle_difference(ref_yaw[i], chassis.yaw)
        
        # 2. 生成世界坐标系目标速度 (前馈速度 + 闭环位置误差反馈)
        if closed_loop:
            cmd_vx_w = ref_vx[i] + pos_kp * ex_w
            cmd_vy_w = ref_vy[i] + pos_kp * ey_w
            cmd_yaw = ref_yaw[i]
        else:
            cmd_vx_w = ref_vx[i]
            cmd_vy_w = ref_vy[i]
            cmd_yaw = ref_yaw[i]
            
        # 3. 将世界系速度投影到底盘机体坐标系
        cmd_vx_b, cmd_vy_b = kin.world_to_body_velocity(cmd_vx_w, cmd_vy_w, chassis.yaw)
        
        # 4. 控制器计算
        measured_speeds = np.array([m.actual_speed for m in chassis.motors])
        motor_cmds, _, _ = controller.compute_control(
            target_vx_body=cmd_vx_b,
            target_vy_body=cmd_vy_b,
            target_yaw=cmd_yaw,
            current_yaw=chassis.simulated_gyro_yaw,
            measured_wheel_speeds=measured_speeds,
            closed_loop=closed_loop
        )
        
        # 5. 仿真推进一步
        state = chassis.step(motor_cmds)
        
        hist_x.append(state["x"])
        hist_y.append(state["y"])
        hist_yaw.append(state["yaw"])
        
        pos_err = np.sqrt((state["x"] - ref_x[i])**2 + (state["y"] - ref_y[i])**2)
        hist_err_pos.append(pos_err)
        hist_err_yaw.append(abs(angle_difference(ref_yaw[i], state["yaw"])))
        
    return {
        "dt": dt,
        "steps": steps,
        "time": t_arr,
        "ref_x": ref_x,
        "ref_y": ref_y,
        "ref_yaw": ref_yaw,
        "x": np.array(hist_x),
        "y": np.array(hist_y),
        "yaw": np.array(hist_yaw),
        "err_pos": np.array(hist_err_pos),
        "err_yaw": np.array(hist_err_yaw),
        "kin": kin
    }


# =============================================================================
# 3. 渲染静态综合分析对比图 (Static Comparison Plot)
# =============================================================================

def save_static_comparison(res_sq_cl, res_sq_ol, res_f8_cl, res_f8_ol, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    
    # 图 1: 正方形 2D 轨迹对比
    ax1 = axes[0, 0]
    ax1.plot(res_sq_cl["ref_x"], res_sq_cl["ref_y"], 'k--', linewidth=2.0, label='Reference Path (2m x 2m)')
    ax1.plot(res_sq_ol["x"], res_sq_ol["y"], color='#d62728', linestyle='-.', linewidth=1.8, label='Open-Loop (Severely Drifts)')
    ax1.plot(res_sq_cl["x"], res_sq_cl["y"], color='#2ca02c', linestyle='-', linewidth=2.2, label='Closed-Loop (Cascaded PID)')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title("Square Path: Open-Loop vs Closed-Loop under Disturbance", fontsize=11, fontweight='bold')
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
    
    # 图 2: 正方形位置误差演变
    ax2 = axes[0, 1]
    ax2.plot(res_sq_ol["time"], res_sq_ol["err_pos"], color='#d62728', linestyle='-.', label='Open-Loop Tracking Error (m)')
    ax2.plot(res_sq_cl["time"], res_sq_cl["err_pos"], color='#2ca02c', linestyle='-', linewidth=2.0, label='Closed-Loop Tracking Error (m)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_title("Square Path: Position Tracking Error (m)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (m)")
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # 图 3: 8 字形 2D 轨迹对比
    ax3 = axes[1, 0]
    ax3.plot(res_f8_cl["ref_x"], res_f8_cl["ref_y"], 'k--', linewidth=2.0, label='Reference Figure-8')
    ax3.plot(res_f8_ol["x"], res_f8_ol["y"], color='#d62728', linestyle='-.', linewidth=1.8, label='Open-Loop (Distorted & Diverging)')
    ax3.plot(res_f8_cl["x"], res_f8_cl["y"], color='#1f77b4', linestyle='-', linewidth=2.2, label='Closed-Loop (Cascaded PID)')
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.set_title("Figure-8 Path: Open-Loop vs Closed-Loop under Disturbance", fontsize=11, fontweight='bold')
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    
    # 图 4: 8 字形航向角跟踪对比
    ax4 = axes[1, 1]
    ax4.plot(res_f8_cl["time"], np.degrees(res_f8_cl["ref_yaw"]), 'k--', linewidth=1.5, label='Reference Yaw (deg)')
    ax4.plot(res_f8_ol["time"], np.degrees(res_f8_ol["yaw"]), color='#d62728', linestyle='-.', label='Open-Loop Yaw (deg)')
    ax4.plot(res_f8_cl["time"], np.degrees(res_f8_cl["yaw"]), color='#1f77b4', linestyle='-', linewidth=2.0, label='Closed-Loop Yaw (deg)')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.set_title("Figure-8 Path: Heading / Yaw Tracking (deg)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Heading Yaw (deg)")
    ax4.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] 轨迹跟踪开闭环对比分析图已保存至: {output_path}")


# =============================================================================
# 4. 渲染轨迹跟踪动图 (Animated GIF Renderers)
# =============================================================================

def render_path_gif(res_cl, res_ol, title_str, output_gif_path, x_lim=(-1.0, 3.0), y_lim=(-1.0, 3.0)):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(title_str, fontsize=12, fontweight='bold')
    ax.set_xlabel("World X (m)", fontsize=10)
    ax.set_ylabel("World Y (m)", fontsize=10)
    
    # 绘制参考轨迹线
    ax.plot(res_cl["ref_x"], res_cl["ref_y"], 'k--', linewidth=1.8, label='Reference Target Path')
    
    # 开环与闭环轨迹线
    line_ol, = ax.plot([], [], color='#d62728', linestyle='-.', linewidth=1.8, label='Open-Loop (Disturbed)')
    line_cl, = ax.plot([], [], color='#2ca02c', linestyle='-', linewidth=2.2, label='Closed-Loop PID (Robust)')
    
    # 当前位置点
    dot_ol, = ax.plot([], [], 'o', color='#d62728', markersize=7)
    dot_cl, = ax.plot([], [], 'o', color='#2ca02c', markersize=7)
    
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # 抽帧 (每 6 步 1 帧)
    steps = res_cl["steps"]
    dt = res_cl["dt"]
    frame_indices = list(range(0, steps, 6))
    if frame_indices[-1] != steps - 1:
        frame_indices.append(steps - 1)
        
    def init():
        line_ol.set_data([], [])
        line_cl.set_data([], [])
        dot_ol.set_data([], [])
        dot_cl.set_data([], [])
        return [line_ol, line_cl, dot_ol, dot_cl]
        
    def update(frame_idx):
        line_ol.set_data(res_ol["x"][:frame_idx+1], res_ol["y"][:frame_idx+1])
        line_cl.set_data(res_cl["x"][:frame_idx+1], res_cl["y"][:frame_idx+1])
        dot_ol.set_data([res_ol["x"][frame_idx]], [res_ol["y"][frame_idx]])
        dot_cl.set_data([res_cl["x"][frame_idx]], [res_cl["y"][frame_idx]])
        return [line_ol, line_cl, dot_ol, dot_cl]
        
    ani = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, interval=25)
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    ani.save(output_gif_path, writer='pillow', fps=25)
    plt.close()
    print(f"[OK] 轨迹跟踪动画已保存至: {output_gif_path}")


if __name__ == "__main__":
    print("[1/4] 正在运行正方形轨迹开环与闭环仿真...")
    res_sq_cl = run_tracking_simulation(generate_square_trajectory, closed_loop=True, enable_disturbances=True)
    res_sq_ol = run_tracking_simulation(generate_square_trajectory, closed_loop=False, enable_disturbances=True)
    
    print("[2/4] 正在运行 8 字形轨迹开环与闭环仿真...")
    res_f8_cl = run_tracking_simulation(generate_figure8_trajectory, closed_loop=True, enable_disturbances=True)
    res_f8_ol = run_tracking_simulation(generate_figure8_trajectory, closed_loop=False, enable_disturbances=True)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    
    print("[3/4] 正在导出静态综合对比分析图...")
    save_static_comparison(
        res_sq_cl, res_sq_ol, res_f8_cl, res_f8_ol,
        os.path.join(out_dir, "path_tracking_comparison.png")
    )
    
    print("[4/4] 正在渲染正方形与 8 字形轨迹跟踪 GIF 动图...")
    render_path_gif(
        res_sq_cl, res_sq_ol,
        "RoboMaster Square Path Tracking (Disturbance Injected)",
        os.path.join(out_dir, "path_tracking_square.gif"),
        x_lim=(-0.8, 2.8), y_lim=(-0.8, 2.8)
    )
    render_path_gif(
        res_f8_cl, res_f8_ol,
        "RoboMaster Figure-8 Path Tracking (Disturbance Injected)",
        os.path.join(out_dir, "path_tracking_figure8.gif"),
        x_lim=(-2.2, 2.2), y_lim=(-1.6, 1.6)
    )
    print("全部轨迹跟踪仿真与可视化生成完毕！")
