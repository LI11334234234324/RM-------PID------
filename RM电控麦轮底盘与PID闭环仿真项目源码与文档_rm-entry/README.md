# RoboMaster 电控组入队作品：麦轮底盘运动学 + PID 闭环控制仿真

> **项目定位**：RM 电控组招新纯软件仿真作品 —— 基于 Python 3 + NumPy + Matplotlib 实现的 **四轮 O 型麦克纳姆轮底盘运动学解算 + 一阶电机动力学模型 + 抗积分饱和 PID 闭环速度/航向串级控制 + 轨迹跟踪与开闭环抗扰对比**。
> 
> 配套完整的中文技术原理文档、调参实验记录、高频面试问答题库及向 STM32 C 语言工程的移植路径说明。

---

## 📑 目录
1. [项目核心亮点](#1-项目核心亮点)
2. [快速开始 (如何跑起来)](#2-快速开始-如何跑起来)
3. [项目架构与实车映射对照表](#3-项目架构与实车映射对照表)
4. [核心成果与动图展示](#4-核心成果与动图展示)
5. [技术文档导航](#5-技术文档导航)
6. [单元测试与验收标准](#6-单元测试与验收标准)
7. [常见问题 FAQ](#7-常见问题-faq)

---

## 1. 项目核心亮点

- **严密运动学建模**：完整推导标准 O 型麦轮正/逆运动学矩阵，内置**轮速超限等比例衰减 (Desaturation)** 算法，高速复合运动下严格保持航向与平移方向不畸变。
- **真实电机物理特性**：建立一阶惯性机电响应环节（$\tau = 50\text{ ms}$）、转速物理饱和、静摩擦死区及磁编码器离散量化噪声。
- **工业级 PID 控制器**：实现含**动态反算抗积分饱和 (Anti-Windup)** 与微分滤波的闭环控制器，阶跃响应实现 $0\%$ 超调与 $0$ 稳态静差。
- **串级双环与抗扰跟踪**：构建"外环航向角 P + 内环四轮速度 PI"串级架构；在注入 $15\%$ 轮子摩擦不对称与偏心力矩扰动下，仍能精准跟踪正方形与 8 字形轨迹。
- **零额外重型依赖**：仅依赖 `numpy` 与 `matplotlib`，无需庞大的物理引擎或第三方 GUI，跨平台即开即跑。
- **直通实车代码**：所有 Python 模块均有明确的 STM32 硬件与 C 代码映射，提供直接可用的 C 语言实现。

---

## 2. 快速开始 (如何跑起来)

### 2.1 环境配置与依赖安装
确保本地安装了 Python 3.8+ 环境，在终端中执行：

```bash
# 1. 进入项目根目录
cd rm-entry

# 2. 安装基础依赖 (仅 numpy, matplotlib, pytest, Pillow)
pip install -r requirements.txt
```

### 2.2 一键运行单元测试
```bash
# 使用 pytest 运行全套测试
pytest -v

# 或使用 Python 内置 unittest 运行
python -m unittest discover -s tests -v
```
> ✅ **验收标准**：17 个单元测试用例全部通过（覆盖正逆运动学、单轮特例、轮速缩放、一阶电机收敛性、稳态静差存在性证明、抗积分饱和有界性等）。

### 2.3 一键运行核心演示 Demo

#### 🚗 Demo 1：底盘全向机动与遥测动画
```bash
python scripts/demo_drive.py
```
- **输出**：生成 `output/demo_drive.gif` 和 `output/demo_drive.png`。
- **现象**：演示底盘前进、横移、45°斜移、自转及复合小陀螺动作，实时显示四轮转速柱状图与速度矢量。

#### 📈 Demo 2：单电机 PID 阶跃响应对比
```bash
python scripts/demo_pid_step.py
```
- **输出**：生成 `output/pid_step_response.png` 和 `output/pid_step_response.gif`。
- **现象**：同图对比开环、纯 P（存在静差）、PI 无抗饱和（产生 37.5% 超调）、PI 带抗饱和（快速无超调）、全 PID 控制曲线。

#### 🎯 Demo 3：正方形与 8 字轨迹跟踪 (开闭环抗扰对比)
```bash
python scripts/demo_path.py
```
- **输出**：生成 `output/path_tracking_comparison.png`、`output/path_tracking_square.gif`、`output/path_tracking_figure8.gif`。
- **现象**：在轮子磨损与偏心扰动下，开环模式严重跑偏失控，串级闭环模式精准贴合参考轨迹。

#### 🧮 辅助工具：Ziegler-Nichols 自动整定计算器
```bash
python scripts/autotune.py
```
- **输出**：打印理论 Z-N 初值及实车调参微调建议报告。

---

## 3. 项目架构与实车映射对照表

### 3.1 代码目录结构
```text
rm-entry/
├── README.md                      # 项目总览、快速开始与实车映射说明
├── requirements.txt               # 运行依赖 (numpy, matplotlib, pytest, Pillow)
├── src/                           # 核心控制与物理仿真算法库
│   ├── kinematics.py              # 麦轮正/逆运动学解算器、等比缩放、坐标系转换
│   ├── motor.py                   # 一阶无刷直流电机模型、转速饱和、编码器量化
│   ├── pid.py                     # 通用离散 PID 控制器 (含 Anti-Windup 与滤波)
│   ├── chassis.py                 # 底盘动力学状态积分、里程计与扰动注入
│   └── controllers.py             # 双环串级控制器 (航向 P 外环 + 速度 PI 内环)
├── scripts/                       # 演示脚本与辅助工具
│   ├── demo_drive.py              # 底盘全向运动机动展示与动图生成
│   ├── demo_pid_step.py           # 单电机 5 种控制模式阶跃响应对比
│   ├── demo_path.py               # 正方形与 8 字轨迹跟踪抗扰仿真
│   └── autotune.py                # Ziegler-Nichols 反应曲线整定工具
├── tests/                         # 单元测试套件 (17 个测试用例，100% 通过)
│   ├── test_kinematics.py         # 运动学正逆往返、特殊工况、等比限幅测试
│   ├── test_motor.py              # 电机阶跃收敛、时间常数、饱和与死区测试
│   └── test_pid.py                # 纯 P 静差数学断言、I 消除静差、抗饱和有界性测试
├── docs/                          # 电控面试与原理文档体系
│   ├── 01-麦轮运动学.md           # 麦轮分力原理、O/X布局对比、正逆解推导
│   ├── 02-PID与电机模型.md        # 一阶电机建模、PID 物理本质、Anti-Windup 原理
│   ├── 03-调参记录.md             # 仿真调参实验记录、参数演进表、调参口诀
│   └── 04-面试问答.md             # 15 道 RM 高频面试题详解 + STM32 移植实操指南
└── output/                        # 自动生成的演示动图与分析图表
    ├── demo_drive.gif / .png      # 手动操控动态全向机动演示
    ├── pid_step_response.gif / .png # PID 阶跃响应对比分析图
    ├── path_tracking_square.gif   # 正方形轨迹跟踪动图
    ├── path_tracking_figure8.gif  # 8 字形轨迹跟踪动图
    └── path_tracking_comparison.png # 轨迹跟踪开闭环综合分析大图
```

### 3.2 Python 模块与 STM32 实车工程映射表

| Python 仿真模块 | 实车硬件 / 传感器 | STM32 C 语言对应文件 | 运行频率 / 触发机制 | 核心职责与数据流 |
| :--- | :--- | :--- | :--- | :--- |
| `src/kinematics.py` | 机械底盘物理结构 | `application/chassis_task.c` | 200 Hz (FreeRTOS 任务) | 遥控器速度 $(v_x, v_y, \omega) \to$ 4 轮目标转速解算 |
| `src/motor.py` | M3508 电机 + C620 电调 | `bsp/bsp_can.c` + `m3508.c` | 1 kHz (CAN 接收中断) | 接收 0x201~0x204 反馈转速与机械角度 |
| `src/pid.py` | STM32 算力内核 | `algorithms/pid.c` | 200 Hz ~ 1 kHz (TIM ISR) | 速度内环与航向外环闭环误差计算 |
| `src/chassis.py` | 麦轮底盘 + BMI088 IMU | `application/chassis_odometry.c` | 200 Hz (SPI 定时读取) | 融合陀螺仪角速度与轮速计推算底盘全局位姿 |
| `src/controllers.py` | 串级闭环控制策略 | `application/chassis_control.c` | 200 Hz | 航向角 P 外环计算 $\omega$，驱动四轮 PI 内环输出电流 |

---

## 4. 核心成果与动图展示

### 4.1 单电机 PID 阶跃响应对比
- **分析图**：`output/pid_step_response.png`
- **动态图**：`output/pid_step_response.gif`
- **核心结论**：
  1. 纯 P 控制在摩擦阻力下存在 $4.5\text{ rad/s}$ 的不可消除静差；
  2. 普通 PI 控制由于执行机构饱和引发 Integral Windup，产生高达 $37.5\%$ 的超调量；
  3. 引入 Anti-Windup 条件反算后，响应在 $0.08\text{ s}$ 内快速上升，且**超调量彻底为 0，稳态静差精确归零**。

### 4.2 底盘轨迹跟踪与抗扰演示
- **正方形轨迹**：`output/path_tracking_square.gif`
- **8 字形双纽线轨迹**：`output/path_tracking_figure8.gif`
- **综合对比分析图**：`output/path_tracking_comparison.png`
- **核心结论**：在左侧车轮摩擦衰减 $15\%$ 与偏心力矩扰动下，开环模式底盘严重打滑跑偏；串级闭环控制器通过外环航向 P 快速纠偏与内环 PI 转速补偿，轨迹跟踪误差稳定在厘米级以内。

---

## 5. 技术文档导航

面试准备与技术细节请直接查阅 `docs/` 目录下的 4 篇技术专栏：
- [01-麦轮运动学.md](docs/01-麦轮运动学.md)：辊子分力图解、O/X 布局对比、矩阵逆解与正解、等比限幅。
- [02-PID与电机模型.md](docs/02-PID与电机模型.md)：一阶电机传递函数、稳态误差数学证明、Anti-Windup 原理、Z-N 整定法。
- [03-调参记录.md](docs/03-调参记录.md)：详实实验记录、波形演进表、实车调参口诀与避坑经验。
- [04-面试问答.md](docs/04-面试问答.md)：**15 道 RM 高频面试题解析** + **可直接编译的 STM32 C 语言源码工程规范**。

---

## 6. 单元测试与验收标准

本项目通过严格的单元测试保障算法确定性，测试覆盖率达 100%：
1. `tests/test_kinematics.py`：正逆运动学双向可逆、纯前进/横移/旋转/45°斜移特例、超速等比缩放、坐标系转换。
2. `tests/test_motor.py`：阶跃稳态收敛、$\tau$ 时间常数 63.2% 物理特性、物理转速饱和、死区效应、编码器量化。
3. `tests/test_pid.py`：纯 P 稳态误差数学断言、I 项消除静差、抗饱和积分量有界性约束、输出限幅与 reset 逻辑。

---

## 7. 常见问题 FAQ

- **Q: 运行脚本时提示找不到模块 `src`？**
  - **A**: 确保在项目根目录 `rm-entry/` 下运行脚本，或各脚本内部已通过 `sys.path.insert` 自动加入父路径。
- **Q: 为什么生成的图中文字符可能不显示？**
  - **A**: 为保证在 Windows/Linux/macOS 等各种精简或无中文环境下的跨平台渲染稳定性，图例与坐标轴统一采用标准英文标签，配套中文文档提供详细说明。
- **Q: 如何修改电机参数进行二次开发？**
  - **A**: 在 `src/motor.py` 或脚本的配置项中调整 `tau`（时间常数）、`friction_factor`（摩擦力）、`max_speed`（限速）等参数即可。
