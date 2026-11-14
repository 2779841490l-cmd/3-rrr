# PAPER_OUTLINE.md — Early Architecture Brainstorm [ARCHIVED / SUPERSEDED]

> **[!NOTE] ARCHIVED EARLY OUTLINE**: This file records the initial pre-redesign outline and contains superseded hypotheses (200 Hz PWM, velocity-based braking, Micro-Yaw compensation, active short-range ILC).  
> **For the current, verified, submission-ready paper structure, ground-truth numbers, and Overleaf LaTeX engineering package, always refer to [`paper/README.md`](README.md), [`paper/manuscript.md`](manuscript.md), and [`paper/latex/main.tex`](latex/main.tex).**

---

# (Original Draft Below)

> **目标期刊**：*IEEE Transactions on Instrumentation and Measurement (IEEE TIM)* 或 *IEEE/ASME Transactions on Mechatronics (T-MECH)*
> **论文题目（Title）**：
> **Sub-100ms Micrometer-Level Positioning on an Ultra-Low-Cost 3-RRR Parallel Stage via Overclocked PWM, In-Flight Optical Braking, and Physics-Guided Ballistic Learning**
> *(基于高频脉冲超频、空中光流预判制动与物理引导弹道学习的超低成本 3-RRR 并联平台百毫秒微米级定位方法)*

---

## Abstract（摘要核心逻辑四段式）
1. **Background & Challenge**：精密平面定位平台（$<10\ \mu\text{m}$、$<100\text{ ms}$）通常依赖昂贵的音圈/压电执行器与光栅尺。使用百元级航模舵机（MG996R）构建的 3-RRR 并联平台虽可将硬件成本降低 99%（$<\$30$），但受限于严重的齿轮背隙（$\sim 46\ \mu\text{s}$）、非对称死区（$8\sim 13\ \mu\text{s}$）及传统 50Hz PWM 控制带宽瓶颈，常规反馈控制会产生剧烈超调与极限环震荡，调节时间长达 $>1\text{ s}$。
2. **Proposed Method**：本文提出一种融合硬件协议超频、空中光流预判制动（IF-OPB）与物理引导神经弹道迭代学习（PG-NB-ILC）的极速精密控制框架。
3. **Key Innovations**：
   - 在 RISC-V (WS63) 芯片上实现 200Hz 原子级高频 PWM 与板载微秒级双相弹道定时器，使有效控制带宽提升 4 倍；
   - 证明大齿隙系统的“百毫秒单向零换向时间最优引理”，利用 1000Hz 光学鼠标传感器（3200 DPI, $7.94\ \mu\text{m}/\text{count}$）的实时速度流构建空中预判早刹切换曲面；
   - 结合二阶空间逆雅可比先验、冗余自由度微偏摆（Micro-Yaw）及残差弹道 ILC，实现 90 秒全自动标定与跨材质表面自适应。
4. **Experimental Results**：实机消融与基准实验表明，所提方法在 $100\text{ ms}$ 内实现全工作空间定位，平均稳态误差低至 $\le 1.5\text{ counts}$（$<12\ \mu\text{m}$），相比传统多步 PID 控制提速超过 **12 倍**，且彻底消除了超调与稳态抖振。

---

## 正文章节规划（Section-by-Section Blueprint）

### Section I: Introduction
- **1.1 Motivation**：低成本精密制造、微操作与自动化测试对“白菜价、微米级、百毫秒响应”平面定位平台的迫切需求。
- **1.2 Bottlenecks of Hobby-Grade Servos in Micrometer Positioning**：
  - 瓶颈 A：50Hz（20ms）采样延迟导致奈奎斯特带宽上限仅 25Hz；
  - 瓶颈 B：高达 $46\ \mu\text{s}$ 的机械背隙使得任何超调换向都会引入 $>40\text{ ms}$ 的空程死区时间；
  - 瓶颈 C：并联机构空间非线性耦合与静摩擦爬行效应（Stick-Slip）。
- **1.3 Main Contributions**（对应四大创新点）。

### Section II: System Architecture & 90-Second Automated Characterization
- **2.1 3-RRR Planar Parallel Stage & Optical Odometry Setup**：
  - BearPi-Pico H3863 (WS63) + 3× MG996R + 3200 DPI 1000Hz Optical Sensor 硬件拓扑图。
- **2.2 Overclocked 200Hz Atomic PWM & On-Chip Ballistic Timer**：
  - 中断锁（IRQ-Locked）零抖动脉冲排序生成算法与 200Hz 高频刚度增强机理。
- **2.3 Automated 90-Second Probe for Deadband, Backlash, and Spatial Jacobian**：
  - 展示实测的非对称死区 $(D^+, D^-)$、齿隙滞回环（Hysteresis Loop）与 $3\times 3$ 空间二阶逆雅可比张量曲面 $K(\delta u_x, \delta u_y)$。

### Section III: Methodology（由 DeepSeek R1 完成核心公式推导）
- **3.1 Zero-Reversal Time-Optimality Theorem under Severe Backlash**（定理 1：单向零换向约束证明）
- **3.2 In-Flight Optical Predictive Braking (IF-OPB)**（空中光流速度截获与切换超平面 $S(x, \dot{x}) = 0$）
- **3.3 Redundant Micro-Yaw Compensation via Dual-Y Differential Actuation**（冗余微偏摆消除残余微米误差）
- **3.4 Physics-Guided Neural Ballistic Iterative Learning Control (PG-NB-ILC)**（一发入魂弹道映射与收敛性证明）

### Section IV: Experimental Validation & Ablation Studies（由 Gemini 3.8 Flash 自动跑实机数据并画图）
- **Fig. 1**：实验平台实物图与软硬件闭环系统框图。
- **Fig. 2**：90 秒自动探针测得的 X/Y 轴非对称齿隙滞回环（Hysteresis Loops）与空间雅可比行列式热力图。
- **Fig. 3（核心对比图）**：阶跃响应对比曲线（Traditional Multi-Step PID vs. Open-Loop IK vs. Proposed IF-OPB + PG-NB-ILC），直观展示传统方法在 $1400\text{ ms}$ 内来回超调，而本文方法在 **$85\text{ ms}$ 内单向丝滑锁死目标**。
- **Fig. 4（频率消融图）**：50Hz vs 100Hz vs 200Hz 驱动下的等效死区宽度与阶跃上升时间柱状对比图。
- **Fig. 5（弹道学习收敛图）**：PG-NB-ILC 在 80 次连续射击中的 100ms 终端误差（RMSE in $\mu\text{m}$）快速下降曲线。
- **Fig. 6（全屏多尺度打靶图）**：微米级微步（$5\text{ counts} \approx 40\ \mu\text{m}$）、中程（$50\text{ counts}$）与大跨度（$250\text{ counts}$）下的二维落点散点图（Bullseye Target Plot）与时间分布直方图。
- **Table I**：与现有文献及商用精密定位平台的“成本-精度-响应时间（Cost–Accuracy–Settling Time）”综合对比表。

### Section V: Conclusion
- 总结全文贡献与未来向 3D 空间并联机构扩展的展望。

