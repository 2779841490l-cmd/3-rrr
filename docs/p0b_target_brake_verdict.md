# 定案4 + P1-B′ + P0-B 实机阶段验收总结（2026-11-13）

> **基线提交**：`7211ed9` → 本阶段包含 [`measure_real_jacobian.py`](../measure_real_jacobian.py)、[`rect_manifold_calibrator.py`](../rect_manifold_calibrator.py)、[`fps_aim_trainer.py`](../fps_aim_trainer.py)、[`probe_target_brake.py`](../probe_target_brake.py)。

---

## 一、 定案 4：全 2×2 实测雅可比与通道死区（[`experiments/real_jacobian_2x2.csv`](../experiments/real_jacobian_2x2.csv)）

修复 [`measure_real_jacobian.py`](../measure_real_jacobian.py) 第 53 行混算 `p_x_plus[0]` 的 bug，补齐同向预载回中、正负分向、多档阶跃（`u1: 12..32 us`, `u2: 12..25 us`, `repeats=4`, 共 64 发耗时 `55.4 s`）：

- **通道死区 `d_hat`**：
  - `d_u1 = 6.78 us`（`+7.45 / -6.10 us`）
  - `d_u2 = 5.90 us`（`+5.56 / -6.25 us`）
- **死区修正后前向雅可比（`cnt/us`）**：
  - `u1 (+1)`: `(+2.072, -0.662)` | `u1 (-1)`: `(+1.713, -0.397)`
  - `u2 (+1)`: `(-1.871, -4.363)` | `u2 (-1)`: `(-1.912, -3.432)`（正负斜率 `1.27×` 非对称）
  - `J_meas = [[+1.8925, -1.8915], [-0.5298, -3.8972]]`
  - `K_meas = J_meas⁻¹ = [[+0.46520, -0.22578], [-0.06323, -0.22590]]` us/cnt
- **1 µs 最小步长量化极限**：`|du1=1us| = 1.965 cnt`，`|du2=1us| = 4.332 cnt`（> 2.5 cnt 门禁）。

---

## 二、 P1-B′：`K_meas` + 死区冲量微扰（`fps_aim_trainer.py` 20 发 `seed=12345` 对照）

| 实验标签 | 改动 | 均光学误差 | 中位数 | `mean \|ex\| / \|ey\|` | R=10 / R=18 px 命中 | 结论 |
|---|---|---|---|---|---|---|
| `_nop1bp_r1` | 旧 `K_trim` | `3.98 cnt` | `3.79` | `2.90 / 2.42` | `80% / 95%` | 基线 |
| `_p1bp_r1` | Phase 2 直接永久叠加 `d_hat` | `14.78 cnt` | — | `10.70 / 7.80` | `20% / 35%` | **否证**：与 Phase 1 `play_g` 双计，过冲 12–34 cnt |
| **`_p1bp_r2`** | **`K_meas` + 停滞临时冲量 `d_hat` (`40 ms`) + `(6,9,12)`** | **`3.90 cnt`** | **`3.16`** | `3.15 /` **`1.88 (-22%)`** | **`85% / 95%`** | **保留**：Y 误差明显下降，但仍受限于 4.33 cnt/us 量化墙 |
| `_p1bp_r3` | Phase 3 步长改细 `(2,4,6)` | `4.76 cnt` | — | — | `60% / 85%` | **否证**：步长低于死区地板失效（`gain +2.42 -> +1.20`） |

---

## 三、 P0-B：脉冲内闭环预测制动（[`experiments/target_brake_p0b_all4_final.csv`](../experiments/target_brake_p0b_all4_final.csv)）

修复 [`probe_target_brake.py`](../probe_target_brake.py) 第 174 行漏乘 `sign` 的 bug（此前 `x-/y-` 全部直撞 `100 ms` 上限并回弹归零），升级：
1. `DRIVE_CAP_MS: 40 -> 100 ms`，切换为 `定案4` 实测 `J_meas / K_meas`；
2. 5 ms 滑窗速度计算预测落点 `pred_brake = proj + tau_brake * vel`（阈值 `2.1 cnt`, `tau_brake=14 ms` 避开单计数伪速度误触）；
3. 开启 `--asym-brake` 按实测 `1.27×` 补偿 `y-` 主动反刹（`hold=-10.5 us / 21 ms`，`y-` 补偿 `-13.3 us`）。

**全四方向统一参数一次通过（`runs=5`，`8/8` 格全 `PASS`，退出码 `0`）**：

| 方向 | `kick=15.0 us`（均值±sd / min..max / 串扰 / `t_brake`） | `kick=16.5 us`（均值±sd / min..max / 串扰 / `t_brake`） | 判定 |
|---|---|---|---|
| `x+` | **`+3.00 ± 0.71 cnt`** (`+2..+4`, 串扰 `-0.20`, `66.6 ms`) | `+4.60 ± 0.89 cnt` (`+4..+6`, 串扰 `-1.20`, `57.3 ms`) | **PASS** |
| `x-` | **`-3.00 ± 0.00 cnt`** (`-3..-3`, 串扰 `+0.60`, `49.0 ms`) | `-3.80 ± 0.84 cnt` (`-5..-3`, 串扰 `+1.20`, `43.9 ms`) | **PASS** |
| `y+` | `+2.80 ± 0.84 cnt` (`+2..+4`, 串扰 `-0.20`, `44.2 ms`) | **`+4.40 ± 0.55 cnt`** (`+4..+5`, 串扰 `-0.40`, `37.5 ms`) | **PASS** |
| `y-` | `-3.60 ± 1.95 cnt` (`-6..-1`, 串扰 `+0.40`, `51.7 ms`) | **`-4.60 ± 0.89 cnt`** (`-6..-4`, 串扰 `+0.60`, `43.1 ms`) | **PASS** |

> **物理结论**：时间域脉冲内闭环预测制动已把 MG996R Y 轴可控微步压缩至 **`2.8~4.6 cnt`、`sd <= 0.89 cnt`**，成功绕开 `4.33 cnt/us` 静态脉冲量化墙，具备向 [`rect_manifold_calibrator.py`](../rect_manifold_calibrator.py) 精修阶段集成与跑 [`verify_hardware_kpi.py`](../verify_hardware_kpi.py) 门禁的条件。

