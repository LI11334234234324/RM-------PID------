"""
===============================================================================
RoboMaster 电控入门仿真项目 - 经典闭环 PID 控制器模块
===============================================================================
本模块对应实车上的什么：
- 实车硬件/软件对应：STM32 底层控制算法库 (例如 pid.c / pid.h)
- 运行机制：通常在 1kHz 或 200Hz 定时器中断 (TIM ISR) 中被周期性调用
- 经典控制架构：
  1. 速度环 (PI 控制器，内环)：输入为目标轮速与编码器反馈转速之差，输出为给电调的电流指令。
  2. 角度环 (P / PD 控制器，外环)：输入为目标底盘航向角与陀螺仪/云台角度之差，输出为目标旋转角速度 omega。
- 工程亮点：
  - 包含完备的抗积分饱和 (Anti-Windup) 机制：积分限幅 + 条件反算钳位。
  - 支持微分低通滤波 (Derivative Filter)，抑制测量噪声导致的微分抖动。
===============================================================================
"""

from typing import Dict, Any


class PIDController:
    """
    通用工业级离散 PID 控制器 (支持抗积分饱和与微分滤波)
    
    连续域算法:
        u(t) = Kp * e(t) + Ki * integral(e(t) dt) + Kd * de(t)/dt
        
    离散域算法 (采样周期 dt):
        e[k] = target - feedback
        P_out = Kp * e[k]
        I_out = I_out + Ki * e[k] * dt (配合 Anti-Windup 反算与钳位)
        D_out = Kd * (e[k] - e[k-1]) / dt (配合一阶低通滤波)
        u_raw = P_out + I_out + D_out
        u_out = clamp(u_raw, -max_output, max_output)
    """

    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        max_output: float = 100.0,       # 控制器最大输出限幅
        max_integral: float = 50.0,      # 积分项最大幅值限幅
        dt: float = 0.005,               # 采样控制周期 dt (秒，默认 5ms/200Hz)
        anti_windup: bool = True,        # 是否启用抗积分饱和 (Anti-Windup)
        derivative_filter_alpha: float = 0.0 # 微分项一阶低通滤波系数 (0.0 为不过滤，0.8 为强滤波)
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_output = float(max_output)
        self.max_integral = float(max_integral)
        self.dt = float(dt)
        self.anti_windup = bool(anti_windup)
        self.d_filter_alpha = float(derivative_filter_alpha)

        # 内部状态变量
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_d_term = 0.0
        self.is_first_run = True

        # 诊断与监控状态
        self.last_target = 0.0
        self.last_feedback = 0.0
        self.last_error = 0.0
        self.p_out = 0.0
        self.i_out = 0.0
        self.d_out = 0.0
        self.output = 0.0
        self.is_saturated = False

    def reset(self) -> None:
        """重置 PID 控制器所有历史累计量"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_d_term = 0.0
        self.is_first_run = True
        self.p_out = 0.0
        self.i_out = 0.0
        self.d_out = 0.0
        self.output = 0.0
        self.is_saturated = False

    def update(self, target: float, feedback: float) -> float:
        """
        单步执行 PID 计算
        
        参数:
            target: 设定值 (Setpoint / Ref)
            feedback: 实际反馈测量值 (Measurement / Fdb)
            
        返回:
            float: 最终限幅后的控制器输出 (Control Output)
        """
        error = float(target - feedback)
        self.last_target = float(target)
        self.last_feedback = float(feedback)
        self.last_error = error

        # 1. 比例项 (Proportional)
        self.p_out = self.kp * error

        # 2. 微分项计算 (Derivative with Filter)
        if self.is_first_run:
            raw_d = 0.0
            self.is_first_run = False
        else:
            raw_d = self.kd * (error - self.prev_error) / self.dt

        # 微分低通滤波: D = alpha * D_prev + (1 - alpha) * D_raw
        if self.d_filter_alpha > 0.0:
            self.d_out = self.d_filter_alpha * self.prev_d_term + (1.0 - self.d_filter_alpha) * raw_d
        else:
            self.d_out = raw_d
        self.prev_d_term = self.d_out
        self.prev_error = error

        # 3. 积分项累加与抗积分饱和 (Anti-Windup via Clamping & Back-Calculation)
        # 先预计算临时输出 (不含新积分步)
        temp_out = self.p_out + (self.integral * self.ki) + self.d_out

        # 判断是否发生饱和以及饱和方向与误差方向是否同向
        # 如果已经正向饱和且 error > 0 (试图继续推大输出)，或者负向饱和且 error < 0，则冻结积分累加
        can_integrate = True
        if self.anti_windup:
            if temp_out >= self.max_output and error > 0:
                can_integrate = False
            elif temp_out <= -self.max_output and error < 0:
                can_integrate = False

        if can_integrate and self.ki != 0.0:
            self.integral += error * self.dt
            # 积分值内部独立钳位
            self.integral = max(min(self.integral, self.max_integral / self.ki), -self.max_integral / self.ki)

        self.i_out = self.ki * self.integral

        # 4. 汇总总输出并进行硬限幅 (Output Clamping)
        raw_output = self.p_out + self.i_out + self.d_out
        if raw_output >= self.max_output:
            self.output = self.max_output
            self.is_saturated = True
        elif raw_output <= -self.max_output:
            self.output = -self.max_output
            self.is_saturated = True
        else:
            self.output = raw_output
            self.is_saturated = False

        return self.output

    def get_debug_info(self) -> Dict[str, Any]:
        """获取当前控制周期的详细内部计算状态 (供绘图与调参分析)"""
        return {
            "target": self.last_target,
            "feedback": self.last_feedback,
            "error": self.last_error,
            "p_out": self.p_out,
            "i_out": self.i_out,
            "d_out": self.d_out,
            "integral": self.integral,
            "output": self.output,
            "is_saturated": self.is_saturated
        }
