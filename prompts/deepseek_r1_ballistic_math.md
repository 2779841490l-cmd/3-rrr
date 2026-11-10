# DeepSeek R1 极简纯数学推导请求（限纯公式输出，勿写工程代码）

**物理背景与实测参数**：
受控对象为 3 颗 MG996R 舵机驱动的 3-RRR 平面并联平台，末端承载 3200 DPI（$1\text{ count} = 7.9375\ \mu\text{m}$）、$1000\text{ Hz}$ 光学鼠标传感器。硬性指标为 **在 $T_{\max} = 100\text{ ms}$ 内完成全屏任意目标位移 $\Delta \mathbf{x}^* \in [-500, 500]^2\text{ counts}$ 的无超调精准定位（稳态误差 $\le 1.5\text{ counts}$）**。
实测物理极限参数：
- 死区：$D_x^+ = 13\ \mu\text{s}, D_x^- = 10\ \mu\text{s}, D_y^+ = 8\ \mu\text{s}, D_y^- = 8\ \mu\text{s}$
- 齿隙半宽：$B_x^\pm = 23\ \mu\text{s}$（总空程 $46\ \mu\text{s}$），$B_y^+ = 18\ \mu\text{s}, B_y^- = 19\ \mu\text{s}$（总空程 $37\ \mu\text{s}$）
- 中心逆雅可比：$K_0 = \begin{bmatrix} 1.0341 & -1.0967 \\ -0.1222 & -0.7358 \end{bmatrix}\ (\mu\text{s}/\text{count})$

请针对论文第 III 章（Methodology），给出以下 **4 项严密数学推导与定理证明**（中英双语公式，适合直接放入 IEEE Transactions 论文）：

1. **定理 1（大齿隙执行器百毫秒单向零换向时间最优引理）**：
   证明在有限控制周期 $T_p$ 与齿隙渡越时间 $\Delta t_{\text{gap}} = 2B_i / \dot{u}_{\max}$ 下，任何包含超调换向（$\text{sign}(\dot{x})$ 变号）的控制轨线其总调节时间 $T_s > 100\text{ ms}$，从而确立单向逼近约束（Monotonic Approach Constraint: $\dot{x}(t) \cdot \Delta x^* \ge 0$）。
2. **空中光流预判早刹切换超平面（In-Flight Optical Predictive Braking Surface）**：
   设动平台在库仑+粘滞摩擦下的减速段动力学为 $\ddot{x} = -\frac{1}{\tau_m}\dot{x} - \mu g\,\text{sign}(\dot{x})$。利用 $1000\text{ Hz}$ 光流实时测得的瞬时位置 $x(t)$ 与速度 $v(t) = \dot{x}(t)$，推导从过充加速脉冲 $u_{\text{boost}}$ 切换至锁定脉冲 $u_{\text{lock}}$ 的精确零速落点切换曲面方程 $S(x, v; x^*) = 0$。
3. **冗余自由度微偏摆补偿映射（Redundant Micro-Yaw Lever Mapping）**：
   设双 Y 轴舵机（$Y_1, Y_2$）引入同向差模微动 $\delta u_{\text{yaw}}$（不改变各自运动方向以规避反向齿隙），推导其通过传感器偏心距 $L_e$ 产生的切向微位移补偿量 $\Delta \mathbf{x}_{\text{yaw}}(\delta u_{\text{yaw}})$。
4. **物理引导神经弹道迭代学习控制（PG-NB-ILC）收敛性定理**：
   定义第 $k$ 次射击的弹道参数向量 $\boldsymbol{\theta}_k = [u_{\text{boost}}, t_{\text{brake}}, u_{\text{lock}}]^T$ 与终端落点误差 $\mathbf{e}_k = \mathbf{x}^* - \mathbf{x}_k(T_{\max})$。推导范数最优 ILC 更新律 $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \Gamma \nabla_{\boldsymbol{\theta}}\hat{\mathbf{x}}\cdot \mathbf{e}_k$ 及其单调收敛谱半径条件 $\rho(I - \Gamma J_{\text{bal}}) < 1$。
