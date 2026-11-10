# AGENTS.md — 3-RRR 3200 DPI 鼠标精密定位系统 Agent 行为准则与工程规范

> **状态**：强制生效（Always-On Workspace Rule）
> **适用对象**：本工作区内运行的所有 AI Agent（Gemini 3.8 Flash、DeepSeek R1、Antigravity 等）

---

## 一、 防丢失与 Git / GitHub 强制同步铁律（Zero-Data-Loss Protocol）

1. **禁止破坏性删除（No Destructive Deletes）**
   - 严禁在未备份的情况下执行清空目录、`Remove-Item -Recurse -Force`、`git reset --hard` 或直接覆盖历史标定参数文件（`calib_params.json`）。
   - 每次生成新的标定参数或模型权重时，必须先将旧版本按时间戳归档至 `backups/` 目录（如 `backups/calib_params_YYYYMMDD_HHMMSS.json`）。

2. **双端代码同仓备份（Host + WS63 Firmware Co-Location）**
   - 凡是修改海思 WS63 固件（`D:\ws63\src\application\samples\custom\servo_test.c`），必须**同步拷贝一份**到本仓库的 `firmware_ws63/` 目录下纳入 Git 版本控制，决不允许固件代码孤立在 Git 仓库之外。

3. **每步必提交、每阶段必推送（Step-by-Step Git & GitHub Sync）**
   - 每完成一个独立子步骤（参见 `PROJECT_PLAN.md` 中的阶段节点），Agent **必须立即执行 Git 提交并推送至 GitHub**：
     ```powershell
     powershell -ExecutionPolicy Bypass -File .\scripts\git_sync.ps1 "feat/fix/calib: <简明步骤说明>"
     ```
   - 若远端仓库尚未配置或网络暂时不可达，必须完成本地 `git commit` 并在回复末尾显式提醒用户检查远端推送状态。

---

## 二、 多模型算力分工与协作规范（Gemini 3.8 Flash vs. DeepSeek R1）

为了最大化工程效率并严格节省用户的 **DeepSeek R1** 额度，所有 Agent 必须遵守以下分工边界：

### 1. Gemini 3.8 Flash（高频工程主力，承担 85%+ 任务）
- **职责范围**：
  - Python 上位机工程代码编写、重构、调试与运行；
  - Windows RawInput 3200 DPI 鼠标底层数据采集与 `COM4` 串口通信测试；
  - 自动化硬件探针（死区测定、正反齿隙测定、空间网格采样）的执行与数据清洗；
  - 海思 WS63 固件 C 代码修改、编译指令执行与日志排查；
  - Git 提交、GitHub 同步、文档更新与误差可视化绘图。
- **行为要求**：
  - 直接动手执行工具调用，少说空话，代码必须模块化、带完整类型注解和硬件异常捕获（确保串口异常退出时自动发送 `C\n` 归中指令）。

### 2. DeepSeek R1（低频首席数学架构师，严格省着用，全周期仅调用 2~3 次）
- **职责范围**：
  - **严禁**让 DeepSeek R1 编写串口通信、文件读写、UI 界面或执行常规报错调试；
  - **仅在以下 2 个关键数学节点调用**：
    1. **节点 A（拿到实测探针数据后）**：输入 `hardware_probe.py` 实测出的死区、正反齿隙与网格雅可比数据，推导**非对称 Prandtl-Ishlinskii 齿隙逆补偿方程**与**二阶非线性雅可比耦合矩阵参数**；
    2. **节点 B（闭环微步精调阶段）**：推导针对 MG996R 舵机死区的**瞬态脉冲激发（Impulse-Retract）+ 变增益防积分饱和（Anti-Windup）闭环控制律**及李雅普诺夫稳定边界（防止 3200 DPI 下出现极限环微幅震荡）。
- **交互方式**：
  - 由 Gemini 3.8 Flash 将实测数据打包成高密度、结构化的数学 Prompt（写入 `prompts/deepseek_r1_prompt.md`），交由 DeepSeek R1 一次性给出数学闭式解与核心算法伪代码，再由 Gemini 3.8 Flash 落地为工程代码。

---

## 三、 硬件电气与机械安全红线（Hardware Safety Constraints）

1. **标准运行环境**
   - 统一使用 Anaconda PyTorch 环境解释器：
     `E:\anaconda\envs\pytorch\python.exe`
   - 默认串口：`COM4`（CH340，`VID_1A86&PID_7523`，波特率 `115200`，8N1），代码中必须具备自动扫描 CH340 端口的能力。

2. **舵机脉宽与双 Y 轴防互搏约束（Critical Mechanical Safety）**
   - **引脚定义**：
     - `S1`（X轴 水平推拉）：`GPIO 2`，中立位 `1510 us`
     - `S2`（Y1轴 右上垂直主提）：`GPIO 5`，中立位 `1545 us`
     - `S3`（Y2轴 右下垂直反装辅助）：`GPIO 7`，中立位 `1455 us`
     - `S4`（Aux 备用）：`GPIO 8`，中立位 `1500 us`
   - **安全限幅**：自动标定与常规闭环控制中，脉宽严格限制在 `[1150, 1850] us` 内，严禁下发超出机械止挡的极限脉宽。
   - **双 Y 轴协同铁律**：`S2 (Y1)` 与 `S3 (Y2)` 共同驱动动平台垂直方向且反向安装，必须满足几何协同约束 $u_{y1} + u_{y2} \approx 3000 + \delta_{\text{coord}}(u_x, u_y)$，两轴非对称差模偏置 $|\Delta u_{\text{diff}}|$ 严禁超过 `35 us`，防止两颗 MG996R 产生内力互搏导致堵转烧毁或拉杆脱扣。

