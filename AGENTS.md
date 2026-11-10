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
