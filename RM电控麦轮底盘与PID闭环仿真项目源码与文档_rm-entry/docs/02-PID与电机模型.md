# PID 闭环控制算法与电机物理建模

> **本模块对应实车上的什么**：
> 在 RoboMaster 实车中，本模块对应 STM32 中的 **核心算法库 (`pid.c` / `pid.h`)** 与 **电机硬件抽象层 (`bsp_can.c` / `m3508.c`)**。
> - **定时器中断 (TIM ISR)**：通常配置 1 kHz 或 200 Hz 的硬件定时器中断，以固定的周期 $dt$ 执行 PID 计算。
> - **输入/反馈**：输入为期望轮速，反馈为 C620 电调每毫秒通过 CAN 报文回传的转子转速 (RPM)。
> - **输出执行**：PID 输出限幅后打包为 CAN 报文（目标电流值 -16384 ~ +16384）发送至电调驱动无刷电机线圈。

---

## 1. 直流电机一阶物理模型与离散化

### 1.1 物理建模与微分方程
在经典机电系统中，直流无刷减速电机（如大疆 M3508P19）的电气时间常数远小于机械时间常数。在速度控制的频段内，电机及其负载惯量可高度近似为一个**一阶惯性环节**：

$$G(s) = \frac{\Omega(s)}{U(s)} = \frac{K}{\tau s + 1}$$

其中：
- $\tau$：**机电时间常数**（Time Constant，M3508 典型负载下约为 $30 \sim 70\text{ ms}$，本项目默认取 $50\text{ ms}$）。物理意义为电机受到阶跃指令后，速度上升到稳态终值 $63.2\%$ 所需的时间。
- $K$：**静态放大增益**（DC Gain，归一化电压/转速指令比例）。

在连续时域中，其对应的微分方程为：
$$\tau \frac{d\omega(t)}{dt} + \omega(t) = K \cdot u(t) - T_{\text{load}}$$

### 1.2 精确零阶保持离散化 (Zero-Order Hold Discretization)
为了在单片机/仿真程序中以步长 $dt = 5\text{ ms}$ 准确更新状态，求解该非齐次一阶线性微分方程：

$$\omega[k+1] = \omega[k] \cdot e^{-dt / \tau} + (K \cdot u[k] - T_{\text{load}}) \cdot (1 - e^{-dt / \tau})$$

定义衰减系数 $\alpha = e^{-dt / \tau}$，则：
$$\omega[k+1] = \alpha \cdot \omega[k] + (1 - \alpha) \cdot u_{\text{target}}$$

**优势**：相比简单的显式欧拉积分（Euler Method），精确指数离散化在任何步长下均**绝对稳定且无截断失真**。

---

## 2. PID 各环节的作用与物理本质

PID 控制器的连续形式为：
$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

在控制周期 $dt$ 下的离散化实现：

$$\begin{aligned}
e[k] &= \text{target}[k] - \text{feedback}[k] \\
P_{\text{out}} &= K_p \cdot e[k] \\
I_{\text{out}}[k] &= I_{\text{out}}[k-1] + K_i \cdot e[k] \cdot dt \\
D_{\text{out}} &= K_d \cdot \frac{e[k] - e[k-1]}{dt} \\
u_{\text{raw}} &= P_{\text{out}} + I_{\text{out}} + D_{\text{out}}
\end{aligned}$$

### 2.1 各控制项的物理直觉

| 控制项 | 物理意义与比喻 | 核心优点 | 潜在风险与缺点 |
| :--- | :--- | :--- | :--- |
| **比例项 P (Proportional)** | **"立足当下"**：根据当前的误差大小施加成比例的恢复力（类似弹簧力 $F=-kx$）。 | 减小误差，加快系统响应速度。 | 过大导致高频震荡与超调；**纯 P 控制在有阻力时无法消除稳态误差**。 |
| **积分项 I (Integral)** | **"铭记过去"**：累计历史微小误差，随着时间推移产生强大的持续纠偏力。 | **彻底消除静差（稳态误差为 0）**，抵抗恒定外载摩擦。 | 引入相位滞后，易引发超调；在执行机构饱和时易发生**积分饱和 (Windup)**。 |
| **微分项 D (Derivative)** | **"预判未来"**：根据误差变化趋势提供阻尼制动力（类似阻尼器 $F=-c\dot{x}$）。 | 抑制超调，提供超前相位，增强系统阻尼。 | **对高频噪声极度敏感**。在编码器测速存在量化噪声时易引起剧烈抖动与电机啸叫。 |

---

## 3. 稳态误差与积分项消除静差的数学证明

### 3.1 纯 P 控制为什么必定存在静差？
考虑一阶受控对象带有负载阻尼力矩 $B \cdot \omega_{ss}$：
- 稳态时导数项为 0：$\omega_{ss} = K \cdot u_{ss} - B \cdot \omega_{ss} \implies \omega_{ss} = \frac{K u_{ss}}{1 + B}$
- 纯 P 控制律：$u_{ss} = K_p (r - \omega_{ss})$
- 代入求解稳态转速：
  $$\omega_{ss} = \frac{K K_p}{1 + B + K K_p} \cdot r$$
- 稳态误差：
  $$e_{ss} = r - \omega_{ss} = \frac{1 + B}{1 + B + K K_p} \cdot r \neq 0$$

**结论**：只要系统存在阻尼/外阻力且 $K_p$ 有限，纯 P 控制下的稳态误差 $e_{ss}$ 严格大于 0。增大 $K_p$ 只能减小静差，但会诱发剧烈振荡。

### 3.2 积分项消除静差的证明
引入积分项后，闭环系统稳态要求 $\dot{I}_{\text{out}} = 0$：
$$\frac{d I_{\text{out}}}{dt} = K_i \cdot e(t) = 0 \implies e_{ss} \equiv 0$$
只要闭环系统稳定收敛，积分项必定持续调整直至误差**精确归零**。

---

## 4. 积分饱和 (Integral Windup) 及其防护机制

### 4.1 积分饱和的危害
当目标转速大幅阶跃或电机负载过大导致控制器输出达到最大物理限幅 $u_{\text{max}}$ 时：
1. 电机已全力输出，无法进一步加速，误差 $e(t)$ 持续存在；
2. 若积分项继续无限制累加，积分值会膨胀到一个极其巨大的数值（Windup）；
3. 当电机转速终于接近目标值时，由于庞大的积分量无法瞬间消退，控制器将持续输出最大反向/正向指令，导致系统产生**极其严重的超调（Overshoot）与长时间恢复振荡**。

```
普通 PI (无抗饱和):
速度 ────────────────────────────/~~~~~~\──────── (产生 30%~50% 巨大超调)
                                /        \
目标 ──────────────────────────/──────────\──────

抗饱和 PI (Anti-Windup):
速度 ─────────────────────────/─────────────── (快速无超调平滑收敛)
目标 ────────────────────────/────────────────
```

### 4.2 本项目实现的 Anti-Windup 动态钳位与反算机制
本项目在 `src/pid.py` 中实现了工业标准的**条件反算积分抗饱和 (Conditional Integration / Clamping)**：

```python
# 1. 预计算当前输出趋势
temp_out = p_out + (integral * ki) + d_out

# 2. 判断是否饱和：若已饱和且误差方向正在加剧饱和，则立即冻结积分累加
can_integrate = True
if temp_out >= max_output and error > 0:
    can_integrate = False
elif temp_out <= -max_output and error < 0:
    can_integrate = False

# 3. 仅在有效区间内积分，并施加独立幅值钳位
if can_integrate and ki != 0.0:
    integral += error * dt
    integral = np.clip(integral, -max_integral / ki, max_integral / ki)
```

---

## 5. Ziegler-Nichols (Z-N) 整定法与实车调参经验

### 5.1 Z-N 阶跃反应曲线法经验公式
对于一阶加滞后模型 $G(s) = \frac{K e^{-Ls}}{\tau s + 1}$：

| 模式 | 比例增益 $K_p$ | 积分时间 $T_i$ | 微分时间 $T_d$ |
| :--- | :--- | :--- | :--- |
| **P** | $\frac{\tau}{K \cdot L}$ | $\infty$ | $0$ |
| **PI** | $0.9 \frac{\tau}{K \cdot L}$ | $3.33 L$ | $0$ |
| **PID** | $1.2 \frac{\tau}{K \cdot L}$ | $2.0 L$ | $0.5 L$ |

### 5.2 RM 实车速度环调参关键避坑
1. **速度环优先采用 PI，严禁滥用 D 项**：
   - 磁编码器测速采用 $M$ 法离散差分，本身存在量化阶梯噪声；
   - 对含有噪声的速度信号求导（$D$ 项）会导致高频噪声被放大百倍以上，引起电机严重高频啸叫、电调发热甚至母线过流保护；
   - 若必须加 $D$，必须配合一阶低通滤波器（Derivative Low-pass Filter）。
2. **Kp 建议取 Z-N 理论值的 $50\% \sim 70\%$**：
   - Z-N 经验公式是按 $25\%$ 衰减率设计的，偏向激进；实车为了保护机械齿轮箱寿命并避免底盘共振，通常取更平稳的保守增益。
