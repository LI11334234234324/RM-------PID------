"""
===============================================================================
RoboMaster 电控入门仿真项目 - 麦轮底盘手动操控与动态交互演示 (demo_drive.py)
===============================================================================
演示目标：
1. 视觉化呈现麦克纳姆轮底盘在 2D 平面上的全向平移与旋转特性。
2. 呈现 4 个麦轮的几何布局 (O型排布与 45° 辊子朝向)、四轮转速条形仪表盘与实时速度合成矢量。
3. 演示底盘典型动作序列：
   - 前进 (Forward, W)
   - 左横移 (Strafe Left, A)
   - 45° 右前斜移 (Diagonal Move, W+D)
   - 原地逆时针自转 (Spin CCW, Q)
   - 边平移边自转小陀螺 (Spin & Strafe, S+E)
4. 导出高质量动画 demo_drive.gif 与分析截图 demo_drive.png。
===============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from src.kinematics import MecanumKinematics
from src.chassis import MecanumChassisSim
from src.controllers import ChassisCascadeController


def generate_drive_sequence():
    """生成一段展现麦轮全向机动性能的动作指令序列 (总计 5.0 秒)"""
    dt = 0.005 # 5ms
    t_total = 5.0
    steps = int(t_total / dt)
    
    # 动作时间段划分:
    # 0.0 ~ 1.0s: 纯前进 vx = 1.0 m/s
    # 1.0 ~ 2.0s: 纯左横移 vy = 1.0 m/s
    # 2.0 ~ 3.0s: 45° 右前斜移 vx = 0.8, vy = -0.8 m/s
    # 3.0 ~ 4.0s: 原地自转 (小陀螺) omega = 2.5 rad/s
    # 4.0 ~ 5.0s: 边后退边自转 (复合运动) vx = -0.8, omega = -2.0 rad/s
    commands = []
    labels = []
    
    for i in range(steps):
        t = i * dt
        if t < 1.0:
            cmd = (1.0, 0.0, 0.0)
            lbl = "Action: Forward (W) [vx=1.0 m/s]"
        elif t < 2.0:
            cmd = (0.0, 1.0, 0.0)
            lbl = "Action: Strafe Left (A) [vy=1.0 m/s]"
        elif t < 3.0:
            cmd = (0.8, -0.8, 0.0)
            lbl = "Action: 45° Diagonal Right (W+D) [vx=0.8, vy=-0.8]"
        elif t < 4.0:
            cmd = (0.0, 0.0, 2.5)
            lbl = "Action: Spin CCW (Q) [omega=2.5 rad/s]"
        else:
            cmd = (-0.8, 0.0, -2.0)
            lbl = "Action: Backward + Spin (S+E) [vx=-0.8, omega=-2.0]"
        commands.append(cmd)
        labels.append(lbl)
        
    return dt, steps, commands, labels


def run_drive_simulation():
    dt, steps, commands, labels = generate_drive_sequence()
    
    kin = MecanumKinematics(wheel_base=0.40, track_width=0.40, wheel_radius=0.076, max_wheel_speed=30.0)
    chassis = MecanumChassisSim(kinematics=kin, dt=dt, enable_disturbances=False)
    controller = ChassisCascadeController(kinematics=kin, speed_kp=2.2, speed_ki=20.0, heading_kp=4.0, dt=dt)
    
    history = {
        "x": [], "y": [], "yaw": [],
        "vx_body": [], "vy_body": [], "omega_body": [],
        "target_vx": [], "target_vy": [], "target_omega": [],
        "wheel_speeds": [], "target_wheel_speeds": [],
        "labels": labels
    }
    
    target_yaw = 0.0
    for i in range(steps):
        target_vx, target_vy, target_omega = commands[i]
        target_yaw += target_omega * dt
        
        # 闭环解算与电机控制
        motor_cmds, target_ws, _ = controller.compute_control(
            target_vx_body=target_vx,
            target_vy_body=target_vy,
            target_yaw=target_yaw,
            current_yaw=chassis.simulated_gyro_yaw,
            measured_wheel_speeds=np.array([m.actual_speed for m in chassis.motors]),
            closed_loop=True
        )
        
        state = chassis.step(motor_cmds)
        
        history["x"].append(state["x"])
        history["y"].append(state["y"])
        history["yaw"].append(state["yaw"])
        history["vx_body"].append(state["vx_body"])
        history["vy_body"].append(state["vy_body"])
        history["omega_body"].append(state["omega_body"])
        history["target_vx"].append(target_vx)
        history["target_vy"].append(target_vy)
        history["target_omega"].append(target_omega)
        history["wheel_speeds"].append(state["actual_wheel_speeds"])
        history["target_wheel_speeds"].append(target_ws)
        
    return dt, steps, history, kin


def render_drive_animation(dt, steps, history, kin, output_gif, output_png):
    fig = plt.figure(figsize=(12, 6.5), dpi=120)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0])
    
    ax_map = fig.add_subplot(gs[:, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_telemetry = fig.add_subplot(gs[1, 1])
    
    # 1. 2D 俯视主地图
    ax_map.set_xlim(-1.5, 2.5)
    ax_map.set_ylim(-1.8, 2.2)
    ax_map.set_aspect('equal')
    ax_map.grid(True, linestyle=':', alpha=0.6)
    ax_map.set_title("RoboMaster Mecanum Chassis 2D View & Trail", fontsize=12, fontweight='bold')
    ax_map.set_xlabel("World X (m)", fontsize=10)
    ax_map.set_ylabel("World Y (m)", fontsize=10)
    
    trail_line, = ax_map.plot([], [], 'b--', alpha=0.5, linewidth=1.5, label='Trajectory Trail')
    robot_center_dot, = ax_map.plot([], [], 'ro', markersize=6)
    
    # 2. 轮速柱状图
    wheel_names = ['FL (Wheel 0)', 'FR (Wheel 1)', 'RL (Wheel 2)', 'RR (Wheel 3)']
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax_bars.bar(wheel_names, [0, 0, 0, 0], color=bar_colors, alpha=0.85, edgecolor='black')
    ax_bars.set_ylim(-35, 35)
    ax_bars.axhline(0, color='black', linewidth=0.8)
    ax_bars.set_ylabel("Angular Velocity (rad/s)", fontsize=9, fontweight='bold')
    ax_bars.set_title("Live 4-Wheel Speed Telemetry", fontsize=11, fontweight='bold')
    ax_bars.grid(True, linestyle=':', alpha=0.5, axis='y')
    
    # 3. 遥测文字面板
    ax_telemetry.axis('off')
    telemetry_text = ax_telemetry.text(
        0.05, 0.90, "", transform=ax_telemetry.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', edgecolor='#cccccc')
    )
    
    # 车体绘制图元暂存
    robot_patches = []
    
    def draw_robot_chassis(x, y, yaw, vx_b, vy_b):
        # 清除上一帧的车身多边形与车轮
        for p in robot_patches:
            p.remove()
        robot_patches.clear()
        
        # 尺寸
        wb = kin.wheel_base # 0.4m
        tw = kin.track_width # 0.4m
        rw = 0.08  # 轮子长
        rh = 0.04  # 轮子宽
        
        # 车体底盘主矩形
        R_mat = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        
        # 车身角点
        half_l = wb / 2.0 + 0.02
        half_w = tw / 2.0 + 0.02
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])
        rot_corners = (R_mat @ corners.T).T + np.array([x, y])
        body_poly = patches.Polygon(rot_corners, closed=True, facecolor='#4a90e2', edgecolor='#1c4587', alpha=0.6, linewidth=2.0)
        ax_map.add_patch(body_poly)
        robot_patches.append(body_poly)
        
        # 4 个轮子位置
        wheel_offsets = [
            ( kin.lx,  kin.ly, +45), # FL (+45°)
            ( kin.lx, -kin.ly, -45), # FR (-45°)
            (-kin.lx,  kin.ly, -45), # RL (-45°)
            (-kin.lx, -kin.ly, +45)  # RR (+45°)
        ]
        
        for w_x, w_y, roller_angle in wheel_offsets:
            w_center = R_mat @ np.array([w_x, w_y]) + np.array([x, y])
            w_corners = np.array([
                [-rw/2, -rh/2],
                [ rw/2, -rh/2],
                [ rw/2,  rh/2],
                [-rw/2,  rh/2]
            ])
            rot_w_corners = (R_mat @ w_corners.T).T + w_center
            wheel_poly = patches.Polygon(rot_w_corners, closed=True, facecolor='#222222', edgecolor='#111111', alpha=0.9)
            ax_map.add_patch(wheel_poly)
            robot_patches.append(wheel_poly)
            
            # 绘制 45° 辊子中心线
            r_rad = np.radians(roller_angle)
            roller_dir = np.array([np.cos(r_rad), np.sin(r_rad)]) * (rw * 0.4)
            r_pt1 = R_mat @ (-roller_dir) + w_center
            r_pt2 = R_mat @ (roller_dir) + w_center
            line, = ax_map.plot([r_pt1[0], r_pt2[0]], [r_pt1[1], r_pt2[1]], color='#ffdd57', linewidth=2.0)
            robot_patches.append(line)
            
        # 航向箭头 (前向 Forward)
        heading_dir = R_mat @ np.array([0.30, 0.0])
        arrow = ax_map.annotate(
            '', xy=(x + heading_dir[0], y + heading_dir[1]), xytext=(x, y),
            arrowprops=dict(facecolor='#d9534f', edgecolor='#d9534f', width=2.5, headwidth=7)
        )
        robot_patches.append(arrow)
        
        # 速度矢量箭头 (机体速度转换到世界系)
        v_world = R_mat @ np.array([vx_b, vy_b]) * 0.4
        if np.linalg.norm(v_world) > 0.05:
            vel_arrow = ax_map.annotate(
                '', xy=(x + v_world[0], y + v_world[1]), xytext=(x, y),
                arrowprops=dict(facecolor='#5cb85c', edgecolor='#5cb85c', width=2.0, headwidth=6)
            )
            robot_patches.append(vel_arrow)

    # 抽帧保存
    frame_indices = list(range(0, steps, 4))
    if frame_indices[-1] != steps - 1:
        frame_indices.append(steps - 1)

    def init():
        trail_line.set_data([], [])
        robot_center_dot.set_data([], [])
        for bar in bars:
            bar.set_height(0)
        telemetry_text.set_text("")
        return [trail_line, robot_center_dot, telemetry_text]

    def update(frame_idx):
        cx = history["x"][frame_idx]
        cy = history["y"][frame_idx]
        cyaw = history["yaw"][frame_idx]
        vx_b = history["vx_body"][frame_idx]
        vy_b = history["vy_body"][frame_idx]
        w_b = history["omega_body"][frame_idx]
        ws = history["wheel_speeds"][frame_idx]
        lbl = history["labels"][frame_idx]
        
        # 更新轨迹
        trail_line.set_data(history["x"][:frame_idx+1], history["y"][:frame_idx+1])
        robot_center_dot.set_data([cx], [cy])
        
        # 重绘车体
        draw_robot_chassis(cx, cy, cyaw, vx_b, vy_b)
        
        # 更新轮速柱状图
        for bar, h in zip(bars, ws):
            bar.set_height(h)
            
        # 更新遥测面板
        info = (
            f"=== RoboMaster Telemetry ===\n"
            f"Time: {frame_idx * dt:.2f} s / {steps * dt:.1f} s\n"
            f"{lbl}\n\n"
            f"Pose (World):   X = {cx:+5.2f} m, Y = {cy:+5.2f} m\n"
            f"Yaw Angle:      Theta = {np.degrees(cyaw):+6.1f} deg ({cyaw:+4.2f} rad)\n"
            f"Body Velocity:  vx = {vx_b:+4.2f} m/s, vy = {vy_b:+4.2f} m/s\n"
            f"Spin Rate:      omega = {w_b:+4.2f} rad/s\n\n"
            f"Wheel Speeds (rad/s):\n"
            f"  FL: {ws[0]:+5.1f}  |  FR: {ws[1]:+5.1f}\n"
            f"  RL: {ws[2]:+5.1f}  |  RR: {ws[3]:+5.1f}"
        )
        telemetry_text.set_text(info)
        
        return [trail_line, robot_center_dot, telemetry_text]

    # 保存静态关键帧分析图 (取动作变换丰富的中间时刻 2.8s)
    mid_idx = int(2.8 / dt)
    update(mid_idx)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=140, bbox_inches='tight')
    print(f"[OK] 手动操作遥测分析静态图已保存至: {output_png}")

    # 保存动画
    ani = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, interval=25)
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)
    ani.save(output_gif, writer='pillow', fps=25)
    plt.close()
    print(f"[OK] 手动操控动态 GIF 已保存至: {output_gif}")


if __name__ == "__main__":
    dt, steps, hist, kin = run_drive_simulation()
    png_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'demo_drive.png'))
    gif_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'demo_drive.gif'))
    render_drive_animation(dt, steps, hist, kin, gif_path, png_path)
