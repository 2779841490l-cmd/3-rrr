# 交接文档（HANDOVER）— 3-RRR 100ms / 3200DPI 精定位

> 交接时间：2026-11-13
> 交接基线 commit：`0ce0773`（已推送 origin/main）
> 目的：让接手模型在**不推翻既有实测结论**的前提下继续推进。
> 必读顺序：本文件 → [`AGENTS.md`](../AGENTS.md) → [`docs/b1_repeatability_verdict.md`](b1_repeatability_verdict.md) → [`docs/problem_statement_no_hardware_swap.md`](problem_statement_no_hardware_swap.md)

---

## 0. 一句话现状

**KPI 门禁（`verify_hardware_kpi.py`）尚未通过，论文写作处于冻结状态。**
当前卡点已从「不知道为什么不听话」收敛为一个具体的机械命题：
**X 轴（S1 单缸）的微步可重复，Y 轴（S2/S3 差动双缸）的微步不可重复。**
下一步的第一优先实验是「定案3：S2/S3 启停阈值差分」，用来证实或否证这个命题。

---

## 1. 环境与硬红线（违反会烧件或毁数据）

| 项 | 值 |
| :--- | :--- |
| Python | `E:\anaconda\envs\pytorch\python.exe` |
| 串口 | CH340（`VID_1A86&PID_7523`），默认 `COM4`，`115200` 8N1 |
| 控制台编码 | 每条命令前置 `set PYTHONIOENCODING=utf-8 && chcp 65001 >nul` |
| 标定参数 | [`calib_params.json`](../calib_params.json)、[`calib_rect_manifold.json`](../calib_rect_manifold.json)（**禁止直接覆盖**，改动前先归档到 `backups/`） |
| 提交推送 | `powershell -ExecutionPolicy Bypass -File .\scripts\git_sync.ps1 "<说明>"`（仓库 `https://github.com/2779841490l-cmd/3-rrr.git`） |
| 脉宽硬限幅 | `[1150, 1850] µs` |
| 中立位 | S1(GPIO2)=1510，S2(GPIO5)=1545，S3(GPIO7)=1455，S4(GPIO8)=1500 µs |
| 双 Y 和锁 | 平动时 `u2 + u3 = 3000 µs`（**硬约束**，`_clamp_and_guard` 强制执行） |
| Micro-Yaw 差模 | `|Δu_yaw| ≤ 25 µs`（MG996R 防堵转互搏） |

### 1.1 最致命的 API 契约（已经坑过三次，务必记住）

`HardwareRectangleTrainer.__init__()` 创建了 `self.sensor = HighPrecisionMouseSensor(dpi=3200)`，
但**不会**调用 `sensor.start()`。任何脚本都必须自己补：

```python
trainer = HardwareRectangleTrainer()
trainer.sensor.start()          # ← 漏掉这行，以下全部失效
time.sleep(0.30)
trainer.sensor.reset_origin()
trainer.move_to_rect_origin(settle_s=0.45)
```

漏掉的症状是：`get_position()` 恒返回 `(0, 0)`、`_hwnd is None`，
看起来像「硬件掉线」，实际是传感器线程没启动。**历史上 3 次「硬件失联」假警报全是这个原因。**

---

## 2. 已确立、不可再推翻的实测事实（含证据）

| # | 结论 | 关键数据 |
| :--- | :--- | :--- |
| 1 | **静态微偏置 = 死步** | `hold=−12 µs` 单独施加 ⇒ `+0.00 ± 0.00 cnt`；已 4 次独立复现 |
| 2 | **dt 是真实、单调的时域旋钮** | `dt`: 0→+0.00 / 5→+1.67 / 10→+1.67 / 20→+4.67 / 35→+7.00 / 50→+13.67 / 70→+18.67 |
| 3 | 串口写入不是瓶颈 | `send_raw_pulses` 实测 **0.32 ms** |
| 4 | WS63 固件无斜坡限速 | `servo_test.c` 以 **50 µs** 切片轮询 `g_cmd_updated`，新指令 ≤1 帧（~5 ms）生效；`ramp_xy_interruptible` 只存在于 `servo_test_3rrr_backup.c` |
| 5 | 持续阶跃下的「运动起始」≈ **35 ms** | `scratch_brake_trace.py`：0–25 ms 零位移，+1 @37 ms，+6 @46 ms，+11 @54 ms |
| 6 | **释放到 `u0` 不是刹车** | 60 ms 撤驱动后仍从 +11 滑到 +29 cnt |
| 7 | **X = S1 单缸；Y = S2/S3 差动对（和锁 3000）** | `evaluate_pulse()` 只出 `(ux, uy)`，驱动侧推 `s3 = 3000 − uy`（[`rect_manifold_calibrator.py:182-185`](../rect_manifold_calibrator.py:182)） |
| 8 | **B1 微步四方向重复性：仅 X+ 可重复** | 同格×3块×6发：X+ `+4.17/4.83/4.83`（sd<1.2）；X− 重尾 −8/−9；Y+ 漂移 2.5→5.7；Y− 崩塌 −15.3→−7.5 |
| 9 | Y 的滑移量子 ≈ **17–18 cnt** | Y+ / kick=45 / hold=0：`dt=8 → 0.00`，但 `dt=4 → +18`（全有全无双峰） |
| 10 | `K_trim` 列 2 方向偏约 **24°** | 造成 +7.3 cnt 寄生 X；正确来源是 `inverse_jacobian_center_us_per_px` 归一化列 |
| 11 | `_dual_side_refine` 的 `(6,9,12) µs` 阶梯是**静态 hold** | ⇒ 全部死步 ⇒ 解释了 `max_walk=6` 发散与记录中的 `3.83 → 5.67 cnt` 恶化 |

证据文件（均已提交）：
`experiments/b1_2d_sweep_jac0_p1_20261113_014232.csv`、
`experiments/b1_2d_sweep_jac0_m1_20261113_014300.csv`、
`experiments/b1_2d_sweep_jac1_p1_20261113_014329.csv`、
`experiments/b1_2d_sweep_jac1_m1_20261113_014357.csv`、
`experiments/b1_four_dir_confirm.csv`

---

## 3. 已被数据否证的假设（**不要重复踩**）

1. ❌「串口慢」→ 0.32 ms。
2. ❌「固件有斜坡」→ 50 µs 轮询，无斜坡。
3. ❌「静态偏置可以当微步」→ 恒为 0.00。
4. ❌「B1 是伪机制、kick 从未落地」→ dt 单调，明确有效。
5. ❌「小幅 kick 更稳」→ Y− 相反：kick=23 时 sd 4.64，kick=26 时 sd 0.84。
6. ❌「尾巴随 kick 单调变大」→ Y− 相反（低 kick 落在触发阈值附近 ⇒ 时灵时不灵）。
7. ❌「只是网格不够密」→ 同一格连跑三块就漂 1~12 cnt，与网格密度无关。
8. ❌「dt=22 ms 更好」→ 悬崖：dt=20 时 sd 0.45~0.89，dt=22 炸到 2.17~3.36。
9. ❌「专家说的静态 `(45,25,12)` 阶梯可以试一下」→ **严禁**，已被反复否证。

---

## 4. 未完成任务清单（按优先级，含具体做法与验收标准）

### P0-A　定案3：S2 / S3 双缸启停阈值差分（**下一步第一件事**）

- **为什么**：X 单缸可靠、Y 差动双缸两方向全崩，这个强相关是当前最有解释力的线索。
  假设 H：差动对的两条腿预滑动阈值不一致 ⇒ 先动的腿把平台拖出偏摆、机构弹性储能，
  后动的腿崩塌时释放 ⇒ 出现「一个滑移量子」的跳变（正好复现 0 或 ~17 的双峰与 −17/−21 的重尾）。
- **怎么做**：分别扫描 S2、S3 的启停阈值与滑移量子。
  注意 `_clamp_and_guard` 强制 `u2+u3=3000`，**不能**直接把一条腿单独推过去。
  合法做法有两种：
  1. 利用允许的 `|Δu_yaw| ≤ 25 µs` 差模窗口做小范围差分探测（安全红线内）；
  2. 先在纸面/仿真里确认，再临时放开 sum 锁做**单次、限幅、限时**探测（需人工在场，改完立刻恢复）。
- **验收**：给出 S2、S3 各自的阈值差 `Δth`（µs）与滑移量子（cnt）。若 `Δth` 明显非零 ⇒ H 成立。
- **产出**：`experiments/s2_s3_threshold_diff.csv` + 文档结论。

### P0-B　定案2e 复活：脉冲内闭环中途修正

- **现状**：[`probe_target_brake.py`](../probe_target_brake.py) 判 FAIL，但原因是设计上的：
  `DRIVE_CAP_MS = 40.0` **短于**实测运动起始 ~35 ms + 行程时间，proj 永远到不了 2 counts。
- **怎么做**：
  1. 把 `DRIVE_CAP_MS` 放到 90~120 ms（仍要保留硬上限以防失控），`BRAKE_TIMEOUT_MS` 同步调整；
  2. 触发阈值按「先冲到目标 −1 cnt 再撤」或「速度阈值触发」重设；
  3. 明确「撤驱动 ≠ 刹车」——撤驱动后还会 coast，所以要么提前撤，要么在撤的同时下发反向 hold。
- **为什么重要**：这是目前**唯一**能吸收「±滑移量子」的方案，也是路线 D 的地基。

### P1-A　X 轴先行落地（不阻塞 Y）

- X+ 的 B1 已满足专家判据（3/3 块 sd<1.2）。
- 可作为「X 轴专用微步」单独接入 refine，Y 轴暂走闭环/二分。
- 注意：这一条**不能**被理解为「完成拼接」——四方向门禁仍未通过。

### P1-B　`step_dir` 方向修复（可独立先行，与 B1 无关）

- 位置：[`rect_manifold_calibrator.py:846`](../rect_manifold_calibrator.py:846)
  `step_dir = self.manifold.K_trim @ err_vec`
- 改为 `calib_params.json` 的 `inverse_jacobian_center_us_per_px` 归一化列（或按 `|J·unit|` 归一）。
- 验收：同一目标点的寄生 X 从 +7.3 cnt 降到 ≤ +2 cnt。

### P2-A　备用路线 B2：折返差动步（Deterministic Bounce）

借道 >60 µs 库仑连续滑动区（斜率 2.05–2.10 cnt/µs ≈ 1/K_lin），用「前进 + 折返」抵消量子。

### P2-B　路线 D：空中中途修正（治本）

在 t≈35–45 ms 的动摩擦窗口内实时读一次鼠标，在线追加 ±3~8 µs。
理论上可把弹道从 92 ms 压到 <65 ms。依赖 P0-B 的闭环截断能力。

### P2-C　定案1：断电反推（60 s，**需人手**）

断电后手动反推 50–80 counts，一锤定音区分「机械间隙」vs「执行器死区」。

### P3　工程运维

- 把「开工前跑 `test_rig_selfcheck.py` + 必要时串口 RTS 复位」写进 `run_aim_trainer.bat/.ps1` 预检。
- 补上 `sensor.start()` 契约检查（一旦漏调就直接报错，而不是静默返回 (0,0)）。
- GUI 模式逐发落盘 CSV + 记录 approach 象限（目前 GUI 不产出任何数据）。

### P3　论文（**KPI 通过前全部冻结**）

- 标题/摘要里的 `sub-100ms` / `micrometer-level` 声明不得修改、不得提前润色。
- 通过后才允许跑 `run_paper_experiments.py` + `plot_paper_figures.py`。
- 现有负结果（B1 四方向重复性、静态阶梯无效、Y 轴重尾）已具备论文价值，
  可作为「为何必须做脉冲内闭环」的定量论据。

### P4　STS3215 换件（A-1 ~ A-9，仅在无换件路线全败后才启动）

---

## 5. 代码地图（改哪里、别改哪里）

| 文件 | 作用 | 关键位置 |
| :--- | :--- | :--- |
| [`rect_manifold_calibrator.py`](../rect_manifold_calibrator.py) | 主控制器 + `HardwareRectangleTrainer` | `_dual_side_refine` **800–915**；`step_dir` **846**；静态阶梯 `for mag in (6.0,9.0,12.0)` **~868**；`close()` **917–922**；`evaluate_pulse` **165–186** |
| [`probe_timed_kick.py`](../probe_timed_kick.py) | B1 双相脉冲探针（744 行） | `_trial` **107–166**；`_is_pass` **186–194**；判据常量 **70–71** |
| [`scratch_b1_2d_sweep.py`](../scratch_b1_2d_sweep.py) | 任意方向 kick×dt 扫描 + **逐发 CSV 落盘** | `--kicks` 写重复值即可做「同格多块」重复性检验 |
| [`scratch_b1_four_dir_confirm.py`](../scratch_b1_four_dir_confirm.py) | 四方向同口径门禁 | `RECIPES` **~46** |
| [`probe_target_brake.py`](../probe_target_brake.py) | 定案2e 脉冲内闭环 | `BRAKE_TIMEOUT_MS = 140.0`、`DRIVE_CAP_MS = 40.0` |
| [`servo_driver.py`](../servo_driver.py) | 串口驱动 | `_clamp_and_guard` **108–125**（双 Y 和锁）；`send_raw_pulses` **127–135** |
| [`firmware_ws63/servo_test.c`](../firmware_ws63/servo_test.c) | WS63 200 Hz 固件 | 帧生成 **255–329**；`g_pwm_period_us = 5000` |
| [`verify_hardware_kpi.py`](../verify_hardware_kpi.py) | KPI 门禁（Gate1/2/3） | 退出码 0 才算通过 |

### 5.1 常用命令模板

```powershell
set PYTHONIOENCODING=utf-8 && chcp 65001 >nul && "E:\anaconda\envs\pytorch\python.exe" scratch_b1_2d_sweep.py --mode jac1 --sign +1 --hold="-3" --kicks "21,21,21" --dts "20" --runs 6
```

> argparse 陷阱：负值请用等号形式 `--hold="-5"`，避免 `-5` 被当成新选项。

### 5.2 判据（不得自行放宽）

- **单格微步判据**：`|mean| ∈ [2, 5] cnt` 且 `sd ≤ 2 cnt`，且**同格三块全部 PASS** 才算「可重复」。
- **KPI Gate1**：`(30, 20)` 冷启动 12 发，**后 5 发**平均误差 ≤ 2.5 cnt 且最大误差 ≤ 4.0 cnt。
- **KPI Gate2**：六方向 `[(35,0), (−35,0), (0,35), (0,−35), (30,25), (−30,−25)]`，
  每方向 5 发 ILC 后连测 2 发（**禁止挑最好的一发**），每方向平均 ≤ 3.0 cnt，六方向总平均 ≤ 2.0 cnt。
- **铁律**：`[禁止只挑好方向拼]`。任一方向不达标即不得拼接、不得宣布阶段完成。
- **铁律**：`verify_hardware_kpi.py` 退出码非 0 时，禁止改 `paper/manuscript.md`、禁止建议切 Argon 润色。

---

## 6. 接手后的第一小时建议

1. 跑 `test_rig_selfcheck.py` 确认传感器与串口在线（务必确认 `sensor.start()` 已被调用）。
2. 复现一次 X+ 的三块重复性：
   `scratch_b1_2d_sweep.py --mode jac0 --sign +1 --hold="-12" --kicks "45,45,45" --dts "20" --runs 6`
   —— 若出现 `+4.2 ~ +4.8` 且 sd<1.5，说明当前硬件状态与交接时一致，可放心继续。
3. 若 Y− 三块仍呈崩塌（−8 ~ −15），说明 Y 轴问题仍在，**优先做 P0-A（定案3）**，
   不要再去密网格里找「漂亮的单格」。
4. 每完成一个独立子步骤，立即 `git_sync.ps1` 提交推送。
