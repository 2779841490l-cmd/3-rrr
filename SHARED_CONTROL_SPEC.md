# 【系统开发指令书】3-RRR 宏微复合人机共享控制与硬件在环（HIL）自动整定系统

## 0. 最高工程红线（Frozen Baseline Protection）

在执行本任务书的任何代码编写与测试前，必须严格遵守以下红线：
1. **绝对禁止修改已封板的论文与底层硬件基线文件**：
   - 严禁修改 `rect_manifold_calibrator.py`、`servo_driver.py`、`firmware_ws63/servo_test.c`、`verify_hardware_kpi.py`、`calib_rect_manifold.json`、`run_paper_experiments.py`、`plot_paper_figures.py` 以及 `paper/` 和 `experiments/` 目录下的任何文件。
2. **完全模块化隔离开发**：
   - 所有新增代码、配置、录制切片与整定日志，必须全部放置在新建的独立目录 `shared_control_hil/` 下。
   - 对底层 3-RRR 平台的驱动，通过只读导入（Import）现有模块中的串口协议（`SUBUS_V2` 的 `B_CMD`、`H_OVR`、`H` 指令）与运动学接口实现。

---

## 一、 项目背景与核心痛点

我们已经完成了 3-RRR 并联微定位台的底层控制（单步弹道耗时 `~84 ms`，整定时间 `98.2 ms`，定位精度 `1.83~2.39 counts / 14.5~19.0 µm`）。现在需要将该平台作为**桌面宏微复合（Macro-Micro）物理辅助底座**应用于 FPS 游戏（CS:GO / CS2）的实战瞄准修正，解决以下四大核心工程问题：

1. **全时段不马枪（急停单点 + 连发压枪 + 多目标扫射转移）**：
   - 不仅要解决“看到第 1 个人急停开枪没瞄准头”的首发微调，还要解决“同时面对 2~3 个敌人按住左键连发扫射转移（Spray Transfer）”时的连续弹道拉回与目标无缝切换。
2. **准星跟随弹道跳动（Recoil Crosshair）与 3D 视角映射**：
   - 游戏内已开启“准星跟随弹道跳动（`cl_crosshair_recoil 1`）”，连发时准星不再位于屏幕几何中心 $(u_0, v_0)$，而是随枪口上扬与横摆实时移动至 $\mathbf{c}(t) = (u_c(t), v_c(t))$。
   - 屏幕像素是 3D 针孔相机透视投影，且鼠标放入 3-RRR 移动框内可能存在安装偏角 $\alpha$ 或橡皮筋拉力变化，需具备像素到 3-RRR 鼠标 counts 的精确映射与 **3 秒一键在线自校准**。
3. **人机无感协同与物理行程防饱和（$\pm 100\text{ counts}$ 限位保护）**：
   - 大范围拉枪与大幅下拉压枪由人手完成，高频瞬态偏差（$\le 45\text{ counts}$）由 3-RRR 台子补齐。必须严格区分“人手移动”与“台子移动”，防止人机抢方向盘导致同向过冲，并通过**高通洗出滤波器（Washout Filter）**与**运动掩蔽回中**防止台子在 30 发扫射中撞到机械限位。
4. **彻底抛弃人眼主观试错：建立“毫秒级遥测裁判 $\rightarrow$ 自动归因 $\rightarrow$ 桌面 HIL 硬件回放自整定”闭环**：
   - 严禁依赖人类主观感受评价好坏。必须建立客观的逐发子弹击发快照裁判机制，采集真实实战切片后，脱离游戏在桌面上驱动 3-RRR 实体硬件进行回放闭环寻优（Hardware-in-the-Loop Replay Auto-Tuning）。

---

## 二、 系统目录架构设计（`shared_control_hil/`）

```text
shared_control_hil/
├── config_hil.json               # 系统超参数、ROI尺寸、枪械射速表、校准矩阵与最优控制参数存档
├── fast_self_calib.py            # F8热键 3秒在线自校准（自动测量鼠标安装偏角 α 与 2x2 雅可比矩阵）
├── vision_roi_tracker.py         # 240Hz 中心ROI高速截屏 + YOLO敌我头/胸检测 + 动态跟跳准星追踪
├── shared_controller.py          # 人机解耦估计器 + 多目标意图切换 + 4态状态机 + 洗出滤波与掩蔽回中
├── telemetry_referee.py          # 毫秒级黑匣子环形缓存 + 逐发击发快照裁判 + 4类犯规自动归因诊断器
├── hil_replay_tuner.py           # 脱离游戏的桌面 3-RRR 硬件在环（HIL）实战切片回放与自动参数寻优器
└── verify_shared_control_kpi.py  # 自动化验收脚本（验证延迟、帧率、多目标切换、零撞墙与净收益指标）
```

---

## 三、 核心模块详细技术规范

### 模块 1：3 秒无感在线自校准（`fast_self_calib.py`）

1. **底座运动学 2 秒微脉冲自检（触发热键 `F8`）**：
   - 从当前中位 $(1496.0, 1466.0)\ \mu\text{s}$ 出发，依次向 $+u_1, -u_1, +u_2, -u_2$ 发送 4 次定长微脉冲（幅度 $\pm 25\ \mu\text{s}$，每次耗时 $120\text{ ms}$，总耗时 $< 1.5\text{ s}$）。
   - 直接读取 3200 DPI 鼠标传感器的实际二维输出向量 $\Delta\mathbf{p}_{\text{mouse}}$，实时更新当前装夹姿态下的 $2\times 2$ 耦合雅可比矩阵 $\mathbf{J}_{\text{stage}}$ 及其逆矩阵 $\mathbf{K}_{\text{stage}} = \mathbf{J}_{\text{stage}}^{-1}$。即便鼠标在夹具内斜放了 $5^\circ$，求逆后也能自动正交解耦。
2. **3D 视角到 2D 像素的针孔投影转换**：
   - 给定屏幕分辨率 $W \times H$、水平视野角 $\text{FOV}_x$（16:9 下默认 $106.26^\circ$）及游戏灵敏度 $s_{\text{game}}$，等效像素焦距为 $f_x = \frac{W/2}{\tan(\text{FOV}_x/2)}$。
   - 当目标相对于当前准星 $\mathbf{c}(t)$ 的像素偏差为 $(\Delta u, \Delta v)$ 时，所需的鼠标传感器 counts 为：
     $$\Delta x_{\text{cnt}} = \frac{\arctan(\Delta u / f_x) \cdot (180/\pi)}{0.022 \cdot s_{\text{game}}}, \qquad \Delta y_{\text{cnt}} = \frac{\arctan(\Delta v / f_x) \cdot (180/\pi)}{0.022 \cdot s_{\text{game}}}$$
   - 支持通过递推最小二乘法（RLS）在静止击发间隙根据 $\Delta\mathbf{c}$ 与 $\Delta\mathbf{p}_{\text{stage}}$ 自动微调等效比例系数 $(\alpha_x, \alpha_y)$。

### 模块 2：中心 ROI 视觉感知与动态准星追踪（`vision_roi_tracker.py`）

1. **低延迟中心 ROI 截取**：使用 `dxcam` 或 `mss` 仅截取屏幕中心 $(u_0, v_0)$ 周围 $256 \times 256$ 像素区域，截屏耗时 $< 1.5\text{ ms}$。
2. **动态跟跳准星 $\mathbf{c}(t)$ 实时定位**：针对 `cl_crosshair_recoil 1`，利用准星 RGB 颜色阈值与十字形态学卷积（耗时 $< 0.2\text{ ms}$），实时定位 $\mathbf{c}(t) = (u_c(t), v_c(t))$，遮挡时由卡尔曼/后坐力预测兜底。
3. **YOLO 敌我与头/胸关键点检测 + CV-KF 运动提前量**：输出 $\mathbf{h}_i^{\text{head}}(t), r_i^{\text{head}}, \mathbf{h}_i^{\text{chest}}(t)$，并用常速度卡尔曼滤波器估计 $\hat{\mathbf{h}}_i(t + t_{\text{lead}}) = \mathbf{h}_i(t) + k_{\text{lead}} \cdot \mathbf{v}_{\text{target}, i}(t)$。

### 模块 3：全时段共享控制器与多目标扫射转移（`shared_controller.py`）

1. **纯人手动作与台子动作实时剥离**：
   $$\Delta\mathbf{p}_{\text{hand}}(t) = \Delta\mathbf{p}_{\text{usb}}(t) - \Delta\mathbf{p}_{\text{stage\_model}}(t - \tau_{\text{delay}})$$
2. **多目标同框“意图感知切换”**：
   $$J_i = \|\mathbf{d}_i\|_2 - \beta \cdot \max\!\left(0,\ \mathbf{v}_{\text{hand}}(t) \cdot \frac{\mathbf{d}_i}{\|\mathbf{d}_i\|_2}\right) + P_{\text{switch}} \cdot \mathbb{I}(i \neq i_{\text{prev}})$$
   单点/前 3 发优先锚定头部，连发 $> 4$ 发支持向胸部降级。
3. **四态协同状态机与高通洗出滤波器**：
   - State 1: 大角度甩枪潜伏态 ($\|\mathbf{v}_{\text{hand}}\|_2 > v_{\text{flick}}$)
   - State 2: 急停首发锁头态 ($\|\mathbf{v}_{\text{hand}}\|_2 < v_{\text{stop}}, 3.0 \le \|\mathbf{e}_{\text{cnt}}\|_2 \le 48.0$)
   - State 3: 连发压枪与扫射转移态，洗出方程 $\mathbf{p}_{\text{stage}}^{\text{cmd}}(k) = \gamma_{\text{wash}} \cdot \mathbf{p}_{\text{stage}}(k-1) + \lambda(v_{\text{hand}}) \cdot \mathbf{e}_{\text{cnt}}(k), \gamma_{\text{wash}} = \exp(-\Delta t / \tau_w)$，严格钳位在 $[-80, +80]\text{ counts}$
   - State 4: 运动掩蔽无感回中态 ($250\sim 350\text{ ms}$ 斜坡退回 $(1496.0, 1466.0)\ \mu\text{s}$，$1\text{ ms}$ 内可打断)

### 模块 4：毫秒级电子裁判与自动误差归因器（`telemetry_referee.py`）

- 5 秒环形缓存，逐发记录 $E_{\text{actual}}(t_k), E_{\text{hand}}(t_k), \Delta E_k = E_{\text{hand}}(t_k) - E_{\text{actual}}(t_k)$ 与 Hitbox 判定（`HEAD_HIT`, `BODY_HIT`, `WHIFF`），并按 4 类犯规（`FAULT_LAG`, `FAULT_CONFLICT`, `FAULT_SATURATION`, `FAULT_HANDOVER_DELAY`）自动归因。

### 模块 5：脱离游戏的桌面硬件在环（HIL）回放自整定器（`hil_replay_tuner.py`）

- 支持 `--record` 与 `--gen-benchmark`（`shared_control_hil/traces/*.npz`），真实连接串口 WS63 与 3200 DPI 鼠标传感器执行 HIL 闭环寻优，最小化综合损失 $\mathcal{L}(\boldsymbol{\theta})$，最优 $\boldsymbol{\theta}^* = [k_{\text{lead}}, \lambda_0, \sigma_v, \beta, P_{\text{switch}}, \tau_w]$ 写入 `shared_control_hil/config_hil.json`。

### 模块 6：自动化验收门禁（`verify_shared_control_kpi.py`）

- **Gate S1**：急停首发残差均值 $\le 2.8\text{ counts}$（$\le 0.5\text{ px}$），首发锁头率 $100\%$。
- **Gate S2**：3 目标 20 发扫射转移有效命中率从 $< 40\%$ 提升至 $\ge 85\%$，平均净收益 $\Delta E \ge +10.0\text{ counts}$。
- **Gate S3**：人机同向冲突犯规率（$\Delta E_k < -1.5\text{ cnt}$）$\le 5\%$。
- **Gate S4**：20 发连续扫射最大位移保持在 $[-88, +88]\text{ counts}$ 以内，饱和撞墙次数 $= 0$。
- **Gate S5**：单步控制决策计算耗时 $\le 1.5\text{ ms}$，在线自校准耗时 $\le 2.5\text{ s}$（雅可比逆矩阵重构误差 $\le 5\%$）。
