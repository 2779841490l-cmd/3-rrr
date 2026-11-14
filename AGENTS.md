# AGENTS.md — 3-RRR 100ms-3200DPI 极速定位系统 & 学术论文工程 Agent 行为准则

> **状态**：强制生效（Always-On Workspace Rule）
> **终极目标**：
> 1. **工程指标**：在 **100 ms 内**实现 **3200 DPI（$7.94\ \mu\text{m}/\text{count}$）全屏幕精准定位（误差 $\le 1\sim 2\text{ counts}$）**；
> 2. **学术产出**：产出一篇面向 *IEEE TIM / IEEE T-MECH* 或国内顶刊的高水平学术论文，所有实验数据必须按可复现科研标准自动保存为 CSV 并生成 300 DPI 期刊级图表。

---

## 一、 防丢失与 Git / GitHub 强制同步铁律（Zero-Data-Loss Protocol）

1. **禁止破坏性删除**
   - 严禁在未备份的情况下清空目录或直接覆盖标定文件（`calib_params.json`）及实验数据（`experiments/*.csv`）。
   - 每次更新参数必须自动归档至 `backups/`。

2. **双端代码同仓备份（Host + WS63 Firmware Co-Location）**
   - 凡是修改海思 WS63 固件（`D:\ws63\src\application\samples\custom\servo_test.c`），必须**同步拷贝一份**到本仓库 `firmware_ws63/` 目录下纳入 Git 版本控制。

3. **每步必提交、每阶段必推送（Step-by-Step GitHub Sync）**
   - 每完成一个独立子步骤，Agent **必须立即执行 Git 提交并推送至 GitHub（`git@github.com:2779841490l-cmd/3-rrr.git`）**：
     ```powershell
     powershell -ExecutionPolicy Bypass -File .\scripts\git_sync.ps1 "feat/exp/paper: <简明步骤说明>"
     ```

---

## 二、 三模型算力压榨与极致省 Token 铁律（Compute & Token Economy）

为了彻底保护用户的 **Gemini 4 Argon** 与 **DeepSeek R1** 宝贵配额，全力压榨 **Gemini 3.8 Flash**，严格执行以下分工：

### 1. Gemini 3.8 Flash（全天候主力苦力，承担 92%+ 任务，尽情压榨）
- **职责范围**：
  - 全部 Python 代码编写、WS63 200Hz 超频固件 C 代码修改与编译；
  - 1000Hz 鼠标光流数据采集、`COM4` 串口联调与硬件故障排查；
  - 执行海量实机消融实验（Ablation Study：50Hz vs 200Hz、传统 PID vs 空中光流预判早刹 vs 神经弹道 ILC），并将原始轨迹保存至 `experiments/`；
  - 使用 Matplotlib 自动绘制 IEEE 期刊格式对比图（保存至 `paper/figures/`）；
  - 撰写与扩充 `paper/manuscript.md` 的背景、实验设置、数据分析与代码开源说明。

### 2. DeepSeek R1（纯数学公式推导工具人，全周期仅调用 1~2 次，严禁喂长上下文）
- **省 Token 铁律**：
  - **严禁**让 DeepSeek R1 读项目代码、调串口或写工程脚本；
  - 每次调用前，由 Gemini 3.8 Flash 将数学问题浓缩成 `< 400 字` 的独立纯数学 Prompt（存放于 `prompts/deepseek_r1_ballistic_math.md`），让 R1 一次性输出：
    1. **大齿隙系统 100ms 单向零换向时间最优定理（Zero-Reversal Time-Optimality Theorem）**证明；
    2. **空中光流预判早刹切换超平面（In-Flight Braking Switching Surface）** $S(x, \dot{x}) = 0$ 闭式解；
    3. **物理引导弹道迭代学习（PG-NB-ILC）**收敛性证明。

### 3. Gemini 4 Argon（顶层架构师 & 论文终审总编，占比 < 4%）
- **省 Token 铁律**：
  - 仅在顶层架构升级或最终论文全文写完后调用 1 次，用于全文学术逻辑拔高与 IEEE 顶刊英文润色，日常开发严禁挂着 Argon 跑脚本。

---

## 三、 硬件电气与安全红线（Hardware Safety Constraints）

1. **标准环境与接口**
   - Python 解释器：`E:\anaconda\envs\pytorch\python.exe`
   - 串口设备：自动扫描 CH340（`VID_1A86&PID_7523`，默认 `COM4`，`115200` 8N1）。
2. **舵机安全与双 Y 轴协同红线**
   - 脉宽硬限幅：`[1150, 1850] us`；
   - 中立位基准：`S1 (GPIO 2) = 1510 us`, `S2 (GPIO 5) = 1545 us`, `S3 (GPIO 7) = 1455 us`, `S4 (GPIO 8) = 1500 us`；
   - 双 Y 轴协同：平动模式下 $u_{y1} + u_{y2} = 3000$；启用冗余微偏摆（Micro-Yaw）补偿时，双 Y 轴差模偏置 $|\Delta u_{\text{yaw}}|$ 严禁超过 `25 us`，防止 MG996R 堵转互搏。

---

## 四、 Gemini 3.8 Flash 紧箍咒与硬性工程验收红线（Hard KPI Gate — 严禁糊弄与早产论文）

> **[!CAUTION] 针对 Gemini 3.8 Flash 的最高级别强制约束（违者视为严重失职）：**
> 1. **严禁未达标就写论文或催促交差**：
>    - 在实机控制精度没有真正稳定跑通之前，**绝对禁止**去修改 `paper/manuscript.md`、**绝对禁止**挑某一次偶然命中的样本（Cherry-picking）冒充收敛、**绝对禁止**向用户建议“切到 Argon 润色论文”！
> 2. **硬性程序化验收门禁（`verify_hardware_kpi.py`）**：
>    - 每次调整控制器代码或参数后，**必须**运行实机验收脚本：
>      ```powershell
>      & "E:\anaconda\envs\pytorch\python.exe" verify_hardware_kpi.py
>      ```
>    - 该脚本包含两道铁门：
>      - **Gate 1（抗振荡单调收敛门禁）**：对 `(30, 20)` 连续冷启动射击 12 发，最后 5 发（Shots 8~12）的**中位数（Median）或去极值均值（Trimmed-4 Mean，剔除 1 次因装配间隙换边引起的离群值）必须 $\le 3.2\text{ counts}$（$25\ \mu\text{m}$）**，且平均 $t_{\text{bal}} \le 95.0\text{ ms}$、$t_{\text{set}} \le 150.0\text{ ms}$；
>      - **Gate 2a（六方向全象限百毫秒门禁，对齐 0.03 mm 装配公差极限）**：对 `[(35, 0), (-35, 0), (0, 35), (0, -35), (30, 25), (-30, -25)]` 六个方向各无 ILC 连测 5 发，各方向 **5 发中位数（Median）或 Trimmed-3 Mean 必须 $\le 3.5\text{ counts}$（$28\ \mu\text{m}$）**，**六方向总平均误差必须 $\le 3.0\text{ counts}$（$24\ \mu\text{m}$）**，且允许 **6 个方向中至少 5/6 达标**即判定整体 `[PASS]`（$t_{\text{bal}} \le 95.0\text{ ms}$、$t_{\text{set}} \le 150.0\text{ ms}$）；
> 3. **不达标死磕到底铁律（Iterative Hardware Tuning Loop）**：
>    - 只要 `verify_hardware_kpi.py` 退出码不为 `0`（输出 `[KPI GATE FAILED]`），`Gemini 3.8 Flash` **必须自动分析日志中哪个方向或哪一发超差**（例如检查 `-X` 反向齿隙偏置 `b_offset_x`、正负向不对称增益、早刹时间 `t_brake` 或闭环微调时机），修改 `ballistic_controller.py` 后立即重跑 `verify_hardware_kpi.py`，循环死磕直到 `[ALL HARDWARE KPI GATES PASSED]` 为止！
>    - **唯有 `verify_hardware_kpi.py` 100% 通过后**，才允许运行 `run_paper_experiments.py` 和 `plot_paper_figures.py` 更新最终论文数据。

---

## 五、 论文资产与黄金基线快照（Paper Assets & Frozen Baseline Master Map）

> **跨会话快速检索协议**：一旦用户提及“论文 / 投稿 / Overleaf / LaTeX / 审稿回复 / 实验数据”，Agent 必须直接读取 `paper/README.md` 即可掌握全部事实，严禁复原早期已证伪假设。

1. **封板代码与实机黄金基线（Frozen at commit `45ea2aa`，禁止修改）**
   - 封板文件：`rect_manifold_calibrator.py`、`servo_driver.py`、`firmware_ws63/servo_test.c`、`verify_hardware_kpi.py`、`calib_rect_manifold.json`、`experiments/*.csv`。
   - **已确立核心架构（In-Flight Optical-Feedback Optimal-Time Ballistic Control, IF-OPB）**：
     - 硬件与固件：`100 Hz` 载波周期（$T_{\text{pwm}} = 10000\ \mu\text{s}$）+ $T_{\text{low,min}} = 4500\ \mu\text{s}$ 电容恢复保护 + `0.1 µs` TCXO 定时 + 原子多相发包 + 短臂 $Y$ 连杆降敏 $-49.6\%$（$4.332 \to 2.182\ \text{cnt}/\mu\text{s}$）+ $u_{y1}+u_{y2}=3000\ \mu\text{s}$；
     - 控制律：Active Sandwich Preload（$4\text{--}20\ \mu\text{s}$）$\to$ Stage-1 纯位置时间门限截断（$p_\parallel \ge r_{\text{cut}} L$ 且 $t \ge t_{\min} \in \{36, 52\}\ \text{ms}$）$\to$ 短程 $L \le 70$ 冻结跨发 ILC $\to$ Stage-2 笛卡尔符号防反向（$e_i \cdot (p_{\text{target},i} - p_{\text{start},i}) \le 0$ 钳位）与 $B\to\text{eq}$ 回撤 $Y$ 轴锁。
     - **严禁误恢复的早期证伪设计**：200 Hz PWM、Micro-Yaw 差模补偿、$6\ \text{ms}$ 光学速度微分切换、$L \le 70$ 跨发 ILC。

2. **终稿交付物与核心实测指标（commit `be8b8b2`）**
   - 作者与单位：`Qiuhan Liu`，`School of Mechanical and Electronic Engineering, Wuhan University of Technology, Wuhan 430070, China`。
   - 文档与源码：`paper/README.md`（总索引）、`paper/manuscript.md`（去 AI 精修洁净稿）、`paper/references.bib`（15 篇实查 BibTeX）、`paper/latex/main.tex`（`IEEEtran` 双栏工程）、`paper/paper_overleaf_submission.zip`（Overleaf 一键包；仅待作者上传 `fig0_hardware_photo.jpg` 替换 Fig. 1 占位框）。
   - 核心指标：Protocol I `(30, 20)` 12 发 Last-5 Trim-4 Mean **$1.83\ \text{cnt}$ ($14.5\ \mu\text{m}$)**；Protocol II 6 方向 30 发 Trim-3 Mean **$2.39\ \text{cnt}$ ($19.0\ \mu\text{m}$)**、$t_{\text{bal}} = 83.6\ \text{ms}$、$t_{\text{set}} = 98.2\ \text{ms}$（`5/6 PASS`）；较 PID（$101.04\ \text{cnt}$ / $1244.9\ \text{ms}$）误差降 **$97.63\%$**、提速 **$12.68\times$**。
