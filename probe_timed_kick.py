r"""probe_timed_kick.py — 定案2d：B1 双相限时脉冲（Kick-and-Hold）单变量验证

为什么做这个探针
----------------
定案2c 已经把真实物理底牌掀开了（见 docs/problem_statement_no_hardware_swap.md §5.1）：

    | 指令变化量 du       | 实测位移 dx            | 定性                   |
    |--------------------|------------------------|------------------------|
    |  0 ~ 20 us         |  恒为 0 cnt            | 静默储能区（预滑动）   |
    | 20 ~ 50 us         |  中位 15、均值 29.3+-26.6 | Stribeck 粘滑崩塌区 |
    | 60 ~ 80 us         |  斜率 2.05~2.10 cnt/us | 库仑连续滑动区         |

所以「静态保持一个中等偏移」是必败的：偏移 < 20 us 就完全不动，
偏移落在 20~50 us 就吃到 +-26 cnt 的随机暴冲。
=> 绝对不能把 _dual_side_refine 的 walk 阶梯改成静态 (45, 25, 12) us。

本探针验证的替代方案 B1（双相限时脉冲）
----------------------------------------
    t = 0            : 下发 u0 + du_kick           （kick 相，越过 20 us 静默区）
    t = dt_kick      : 立刻下发 u0 + du_hold       （catch 相，du_hold 只有 +2~+6 us，
                                                    低于静默区 => 伺服不再出力，
                                                    平台靠自身阻尼/摩擦刹停）
    t = dt+settle    : 读一次鼠标，得稳态净位移 dx

物理预期：kick 相只维持 15~30 ms（= 3~6 个 PWM 周期，200 Hz => 5 ms/周期），
位移来不及涨到饱和的 90 cnt，而是停在 2~5 cnt 量级；
catch 相把驱动切断，从而砍掉 2c 里那条「越障后随机暴冲」的长尾。

判定标准（专家给定的硬指标）
----------------------------
只要存在一组 (du_kick, dt_kick) 能稳定产出
        均值 dx ∈ [2, 5] cnt  且  标准差 <= 2 cnt
即判定通过，可把该双相脉冲拼进
rect_manifold_calibrator.py 的 _dual_side_refine()，替换静态 (6, 9, 12) us 阶梯。

对照组（三个，缺一不可）
------------------------
    REF-静默区    kick= 0, dt=20ms, hold=+4  -> 必须 ~0 cnt（证明基线没有漂）
    REF-不回拉    kick=45, dt=20ms, hold=45  -> 与 B1 同 dt，但不回拉，
                                                用来证明「刹车」才是有效的那个变量
    REF-全长保持  kick=45, dt=400ms, hold=+4 -> 复现 §5.1 的随机暴冲（~10~40 cnt）

硬件耗时：约 90~110 秒（比专家估的 30 秒长，因为每发都加了一次确定性预载，
          否则静止基线不可复现，整个对照就失去意义）。
"""

import argparse
import csv
import math
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple

from rect_manifold_calibrator import HardwareRectangleTrainer

# ---------------- 时序常量 ----------------
PRELOAD_US = 35.0          # 与 move_to_rect_origin() 同款反向预载量
PRELOAD_HOLD_S = 0.22      # 预载保持时间（与 move_to_rect_origin 一致）
ORIGIN_SETTLE_S = 0.36     # 预载回位后静置，让齿面锁定
SETTLE_S = 0.46            # catch 相之后静置，等平台刹停
DRIFT_S = 0.26             # 额外观察窗，量化刹停后是否还有爬行（蠕变）

DEFAULT_KICKS_US = (35.0, 45.0, 55.0)
DEFAULT_DTS_MS = (15.0, 20.0, 25.0, 30.0)
DEFAULT_HOLDS_US = (4.0,)          # 阶段 A 只跑一个 hold，省时间
STAGE_B_HOLDS_US = (2.0, 6.0)      # 阶段 B 补另一个 hold 值

# 通过判据
PASS_MEAN_LO, PASS_MEAN_HI = 2.0, 5.0
PASS_STD_MAX = 2.0

# 中心雅可比（计数/µs）与逆雅可比（µs/计数），取自 calib_params.json。
# 用途：把 u 空间的单位向量 v 映射到"屏幕上实际会走的方向"，从而只测该方向的位移。
JAC_CENTER_CNT_PER_US = ((1.18995, -1.69540), (-0.25616, -2.42069))
JAC_INV_CENTER_US_PER_CNT = ((0.73026, -0.51146), (-0.07728, -0.35898))


def _screen_dir_for_vec(vx: float, vy: float) -> Tuple[float, float]:
    """u 空间单位向量 -> 屏幕（计数）方向的单位向量。"""
    ex = JAC_CENTER_CNT_PER_US[0][0] * vx + JAC_CENTER_CNT_PER_US[0][1] * vy
    ey = JAC_CENTER_CNT_PER_US[1][0] * vx + JAC_CENTER_CNT_PER_US[1][1] * vy
    n = math.hypot(ex, ey)
    if n < 1e-9:
        return 1.0, 0.0
    return ex / n, ey / n


# ---------------- 底层时序 ----------------
def _send(trainer, ux: float, uy: float) -> float:
    """下发 raw 脉冲，返回串口写入耗时（秒）。"""
    t0 = time.perf_counter()
    trainer.driver.send_raw_pulses(float(ux), float(uy))
    return time.perf_counter() - t0


def _spin_until(t_target: float):
    """精确等到 t_target。Windows 上 time.sleep 对 5~30 ms 分辨率不足，必须自旋收尾。"""
    while True:
        rem = t_target - time.perf_counter()
        if rem <= 0.0:
            return
        if rem > 0.0030:
            time.sleep(rem - 0.0020)


def _trial(
    trainer,
    u0x: float,
    u0y: float,
    kick_us: float,
    dt_ms: float,
    hold_us: float,
    sign: float = 1.0,
    axis: str = "x",
    vec: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """一次「预载 -> 静止基线 -> kick(限时) -> hold(刹车) -> 稳态读数」。

    axis='x' 动 S1（u0x）；axis='y' 动 S2（u0y，固件自动反解 S3=3000-S2）。
    vec 给了就按 u 空间矢量 kick（= _dual_side_refine 里 `unit` 的真实用法），
    此时 axis 仅作标签用。
    """

    def _u(delta: float) -> Tuple[float, float]:
        if vec is not None:
            return u0x + delta * vec[0], u0y + delta * vec[1]
        if axis == "x":
            return u0x + delta, u0y
        return u0x, u0y + delta

    # 1) 确定性预载：反向 35us 再回 u0，使齿面锁定状态可复现
    _send(trainer, *_u(-PRELOAD_US * sign))
    time.sleep(PRELOAD_HOLD_S)
    _send(trainer, *_u(0.0))
    time.sleep(ORIGIN_SETTLE_S)

    trainer.sensor.recenter_cursor()
    p0 = trainer.get_rect_position()

    # 2) kick 相：下发后立即开始计时，期间绝不读传感器
    t_kick = time.perf_counter()
    w_kick = _send(trainer, *_u(sign * kick_us))
    _spin_until(t_kick + dt_ms / 1000.0)

    # 3) catch 相：立刻拉回 hold，切断驱动，靠阻尼刹停
    t_hold = time.perf_counter()
    w_hold = _send(trainer, *_u(sign * hold_us))
    dt_meas_ms = (t_hold - t_kick) * 1000.0

    # 4) 稳态读数 + 蠕变观察窗
    time.sleep(SETTLE_S)
    p1 = trainer.get_rect_position()
    time.sleep(DRIFT_S)
    p2 = trainer.get_rect_position()

    return {
        "axis": axis,
        "dx_cnt": float(p1[0]) - float(p0[0]),
        "dy_cnt": float(p1[1]) - float(p0[1]),
        "drift_x": float(p2[0]) - float(p1[0]),
        "drift_y": float(p2[1]) - float(p1[1]),
        "dt_meas_ms": dt_meas_ms,
        "w_kick_ms": w_kick * 1000.0,
        "w_hold_ms": w_hold * 1000.0,
    }


def _cell_stats(vals: List[float]) -> Tuple[float, float, float, float]:
    """返回 (mean, stdev, min, max)；n<2 时 stdev 记 nan。"""
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else float("nan")
    return m, sd, min(vals), max(vals)


def _fmt(v: float, nd: int = 2) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:+.{nd}f}"


def _fmt_sd(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.2f}"


def _is_pass(mean: float, sd: float) -> bool:
    # 注意：必须取绝对值。判据是「受控微步的幅度」，正负方向都是同样的好步长。
    # （v1 曾用正区间直接比较，导致 -X 方向的 PASS 格被误标为 fail。）
    return (
        math.isfinite(mean)
        and math.isfinite(sd)
        and PASS_MEAN_LO <= abs(mean) <= PASS_MEAN_HI
        and sd <= PASS_STD_MAX
    )


def _run_cell(
    trainer,
    u0x: float,
    u0y: float,
    tag: str,
    kick_us: float,
    dt_ms: float,
    hold_us: float,
    sign: float,
    runs: int,
    rows: List[Dict],
    axis: str = "x",
    vec: Optional[Tuple[float, float]] = None,
    escreen: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    dxs: List[float] = []
    dys: List[float] = []
    driftxs: List[float] = []
    driftys: List[float] = []
    dts_meas: List[float] = []
    for i in range(1, runs + 1):
        r = _trial(trainer, u0x, u0y, kick_us, dt_ms, hold_us, sign, axis=axis, vec=vec)
        dxs.append(r["dx_cnt"])
        dys.append(r["dy_cnt"])
        driftxs.append(r["drift_x"])
        driftys.append(r["drift_y"])
        dts_meas.append(r["dt_meas_ms"])
        rows.append(
            {
                "tag": tag,
                "axis": axis,
                "sign": sign,
                "kick_us": kick_us,
                "dt_cmd_ms": dt_ms,
                "dt_meas_ms": round(r["dt_meas_ms"], 2),
                "hold_us": hold_us,
                "run_id": i,
                "dx_cnt": round(r["dx_cnt"], 2),
                "dy_cnt": round(r["dy_cnt"], 2),
                "drift_x_cnt": round(r["drift_x"], 2),
                "drift_y_cnt": round(r["drift_y"], 2),
                "w_kick_ms": round(r["w_kick_ms"], 2),
                "w_hold_ms": round(r["w_hold_ms"], 2),
            }
        )

    # 主轴 = 沿被激励方向的实际位移；副轴 = 垂直方向串扰
    if vec is not None:
        ex, ey = escreen if escreen is not None else _screen_dir_for_vec(*vec)
        prim_vals = [dxs[i] * ex + dys[i] * ey for i in range(len(dxs))]
        cross_vals = [-dxs[i] * ey + dys[i] * ex for i in range(len(dxs))]
    else:
        prim_vals = dxs if axis == "x" else dys
        cross_vals = dys if axis == "x" else dxs
    prim_drifts = driftxs if axis == "x" else driftys
    m, sd, lo, hi = _cell_stats(prim_vals)
    mm, msd, _, _ = _cell_stats(cross_vals)
    dm, dsd, _, _ = _cell_stats(prim_drifts)
    tm, _, _, _ = _cell_stats(dts_meas)
    cell = {
        "tag": tag,
        "axis": axis,
        "sign": sign,
        "kick_us": kick_us,
        "dt_cmd_ms": dt_ms,
        "hold_us": hold_us,
        "n": float(len(prim_vals)),
        "mean": m,
        "sd": sd,
        "min": lo,
        "max": hi,
        "cross_mean": mm,
        "cross_sd": msd,
        "drift_mean": dm,
        "drift_sd": dsd,
        "dt_meas_ms": tm,
        "pass": _is_pass(m, sd),
    }
    flag = "PASS" if cell["pass"] else ("----" if not math.isfinite(m) else "    ")
    print(
        f"  [{flag}] {tag:<12} {axis}{'+' if sign > 0 else '-'} "
        f"kick={kick_us:5.1f}us dt={dt_ms:5.1f}ms hold={hold_us:+4.1f}us "
        f"-> d{axis}={_fmt(m)} +-{_fmt_sd(sd)} cnt (min {_fmt(lo)} / max {_fmt(hi)})  "
        f"串扰={_fmt(mm)} +-{_fmt_sd(msd)}  drift={_fmt(dm)}  dt_meas={tm:5.1f}ms"
    )
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description="定案2d · B1 双相限时脉冲 Kick-and-Hold 验证")
    ap.add_argument("--runs", type=int, default=3, help="每格重复次数（默认 3）")
    ap.add_argument("--stage", choices=("a", "b", "all"), default="all")
    ap.add_argument("--both", action="store_true", help="同时测 +X 与 -X 两个方向")
    ap.add_argument("--kicks", type=str, default=",".join(str(int(k)) for k in DEFAULT_KICKS_US))
    ap.add_argument("--dts", type=str, default=",".join(str(int(d)) for d in DEFAULT_DTS_MS))
    ap.add_argument(
        "--holds",
        type=str,
        default="",
        help="catch 相回拉量，逗号分隔，可为负；给了就按 kicks x dts x holds 全栅格跑",
    )
    ap.add_argument(
        "--axis",
        choices=("x", "y", "both"),
        default="x",
        help="踢哪个执行器: x=S1, y=S2, both=都测",
    )
    ap.add_argument(
        "--vec",
        type=str,
        default="",
        help="u 空间单位向量 vx,vy（如 K_trim@e 归一化）；给了就按该矢量做双相脉冲，"
        "这正是 _dual_side_refine 里 unit 的真实用法。此时 --axis 被忽略",
    )
    ap.add_argument("--no-refs", action="store_true", help="跳过 3 个对照组")
    args = ap.parse_args()

    kicks = [float(s) for s in args.kicks.split(",") if s.strip()]
    dts = [float(s) for s in args.dts.split(",") if s.strip()]
    holds_cli: Optional[List[float]] = (
        [float(s) for s in args.holds.split(",") if s.strip()] if args.holds.strip() else None
    )
    holds_a = holds_cli if holds_cli else list(DEFAULT_HOLDS_US)
    signs = (1.0, -1.0) if args.both else (1.0,)
    if args.vec.strip():
        _p = [float(s) for s in args.vec.split(",") if s.strip()]
        if len(_p) != 2:
            raise SystemExit("--vec 需要两个数: vx,vy")
        _n = math.hypot(_p[0], _p[1])
        if _n < 1e-9:
            raise SystemExit("--vec 不能是零向量")
        vec: Optional[Tuple[float, float]] = (_p[0] / _n, _p[1] / _n)
        escreen: Optional[Tuple[float, float]] = _screen_dir_for_vec(*vec)
        axes = ("v",)
    else:
        vec = None
        escreen = None
        axes = ("x", "y") if args.axis == "both" else (args.axis,)

    trainer = HardwareRectangleTrainer()
    manifold = trainer.manifold
    trainer.sensor.start()

    rows: List[Dict] = []
    cells: List[Dict] = []
    t_run0 = time.perf_counter()

    try:
        u0 = manifold.evaluate_kinematic_pulse(0.0, 0.0)
        u0x, u0y = float(u0[0]), float(u0[1]) + float(manifold.hyst_y_pos)

        print("=" * 108)
        print("  定案2d · B1 双相限时脉冲（Kick-and-Hold）—— 目标：可控 2~5 cnt 微步")
        print("=" * 108)
        print(
            f"  raw 基线 u0 = ({u0x:.2f}, {u0y:.2f}) us   预载 {PRELOAD_US:.0f} us / "
            f"{PRELOAD_HOLD_S:.2f}s   PWM 200Hz(周期 5ms)   runs={args.runs}   "
            f"方向={'±' + '/'.join(axes) + ' 双向' if args.both else '+' + '/'.join(axes)}"
        )
        if vec is not None:
            print(
                f"  矢量模式 v=({vec[0]:+.4f},{vec[1]:+.4f})  屏幕方向 e≈"
                f"({escreen[0]:+.3f},{escreen[1]:+.3f})  kicks={kicks} us   "
                f"dts={dts} ms   holds={holds_a} us"
            )
        else:
            print(
                f"  轴={'/'.join(axes)}   kicks={kicks} us   dts={dts} ms   "
                f"holds={holds_a} us   stage={args.stage}"
            )
        print(
            f"  判定: 均值 ∈ [{PASS_MEAN_LO:.0f}, {PASS_MEAN_HI:.0f}] cnt 且 标准差 <= "
            f"{PASS_STD_MAX:.1f} cnt  => PASS"
        )
        print("-" * 108)

        # ---------------- 对照组 ----------------
        if not args.no_refs:
            print("\n[0] 对照组（证明变量确实是「限时」与「回拉」）")
            refs = [
                ("REF-静默区", 0.0, 20.0, 4.0),
                ("REF-不回拉", 45.0, 20.0, 45.0),
                ("REF-全长保持", 45.0, 400.0, 4.0),
            ]
            for tag, k, d, h in refs:
                for ax in axes:
                    for s in signs:
                        cells.append(
                            _run_cell(
                                trainer,
                                u0x,
                                u0y,
                                tag,
                                k,
                                d,
                                h,
                                s,
                                args.runs,
                                rows,
                                axis=ax,
                                vec=vec,
                                escreen=escreen,
                            )
                        )

        # ---------------- 阶段 A：kick x dt 栅格 ----------------
        stage_a: List[Dict] = []
        if args.stage in ("a", "all"):
            print(f"\n[1] 阶段 A：kick x dt x hold 栅格（holds={holds_a} us）")
            for ax in axes:
                for k in kicks:
                    for d in dts:
                        for h in holds_a:
                            for s in signs:
                                c = _run_cell(
                                    trainer,
                                    u0x,
                                    u0y,
                                    "A",
                                    k,
                                    d,
                                    h,
                                    s,
                                    args.runs,
                                    rows,
                                    axis=ax,
                                    vec=vec,
                                    escreen=escreen,
                                )
                                stage_a.append(c)
                                cells.append(c)

        # ---------------- 阶段 B：在候选格上扫 hold ----------------
        if args.stage in ("b", "all") and stage_a and holds_cli is None:
            cand = [
                c
                for c in stage_a
                if math.isfinite(c["mean"])
                and 0.5 <= c["mean"] <= 12.0
                and math.isfinite(c["sd"])
                and c["sd"] <= 4.0
            ]
            cand.sort(key=lambda c: (c["sd"], abs(c["mean"] - 3.5)))
            cand = cand[:3]
            if cand:
                print(
                    f"\n[2] 阶段 B：对 {len(cand)} 个候选格补扫 hold ∈ "
                    f"{STAGE_B_HOLDS_US}（筛选依据：均值 0.5~12 cnt 且 sd <= 4 cnt）"
                )
                for c in cand:
                    for h in STAGE_B_HOLDS_US:
                        cells.append(
                            _run_cell(
                                trainer,
                                u0x,
                                u0y,
                                "B",
                                c["kick_us"],
                                c["dt_cmd_ms"],
                                h,
                                c["sign"],
                                args.runs,
                                rows,
                                axis=c["axis"],
                                vec=vec,
                                escreen=escreen,
                            )
                        )
            else:
                print(
                    "\n[2] 阶段 B：阶段 A 里没有任何格落在 0.5~12 cnt 且 sd <= 4 cnt，"
                    "hold 扫无意义（见下面的判读建议）"
                )

        # ---------------- 落盘 ----------------
        os.makedirs("experiments", exist_ok=True)
        csv_path = "experiments/timed_kick.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "tag", "axis", "sign", "kick_us", "dt_cmd_ms", "dt_meas_ms",
                    "hold_us", "run_id", "dx_cnt", "dy_cnt", "drift_x_cnt",
                    "drift_y_cnt", "w_kick_ms", "w_hold_ms",
                ],
            )
            w.writeheader()
            w.writerows(rows)

        # ---------------- 判读 ----------------
        elapsed = time.perf_counter() - t_run0
        lines: List[str] = []
        lines.append("=" * 108)
        lines.append("  定案2d · B1 双相限时脉冲（Kick-and-Hold）判读")
        lines.append("=" * 108)
        lines.append(f"  raw 基线 u0=({u0x:.2f},{u0y:.2f})us  runs={args.runs}  用时 {elapsed:.0f}s")
        lines.append("")

        by_tag: Dict[str, List[Dict]] = {}
        for c in cells:
            by_tag.setdefault(c["tag"], []).append(c)

        def _pick(tag: str, ax: str, s: float) -> Optional[Dict]:
            return next(
                (c for c in by_tag.get(tag, []) if c["sign"] == s and c["axis"] == ax),
                None,
            )

        def _dir_label(ax: str, s: float) -> str:
            return f"{ax}{'+' if s > 0 else '-'}"

        # 对照组的意义（逐轴逐向，避免两轴数据混在一起判读）
        lines.append("[0] 对照组读法")
        ref_pairs: Dict[Tuple[str, float], Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]] = {}
        for ax in axes:
            for s in signs:
                lab = _dir_label(ax, s)
                ref_quiet = _pick("REF-静默区", ax, s)
                ref_nobrake = _pick("REF-不回拉", ax, s)
                ref_full = _pick("REF-全长保持", ax, s)
                ref_pairs[(ax, s)] = (ref_quiet, ref_nobrake, ref_full)
                if ref_quiet:
                    ok = abs(ref_quiet["mean"]) <= 1.0 and (
                        not math.isfinite(ref_quiet["sd"]) or ref_quiet["sd"] <= 1.0
                    )
                    lines.append(
                        f"  [{lab}] REF-静默区  kick=0         : d{ax}={_fmt(ref_quiet['mean'])}"
                        f" +-{_fmt_sd(ref_quiet['sd'])} cnt"
                        f"  -> {'基线干净（与 §5.1 的 <20us 恒为 0 一致）' if ok else '!! 基线有漂移，本次数据需谨慎'}"
                    )
                if ref_full:
                    big = abs(ref_full["mean"]) > 20.0
                    lines.append(
                        f"  [{lab}] REF-全长保持  kick=45/400ms : d{ax}={_fmt(ref_full['mean'])}"
                        f" +-{_fmt_sd(ref_full['sd'])} cnt"
                        f"  -> {'大幅位移，复现 §5.1 的越障暴冲' if big else '本次位移偏小，需核对预载/齿面状态'}"
                        f"（目标区间只有 {PASS_MEAN_LO:.0f}~{PASS_MEAN_HI:.0f} cnt）"
                    )
                if ref_nobrake:
                    line = (
                        f"  [{lab}] REF-不回拉  kick=45/20ms  : d{ax}={_fmt(ref_nobrake['mean'])}"
                        f" +-{_fmt_sd(ref_nobrake['sd'])} cnt"
                    )
                    if ref_full and abs(ref_full["mean"]) > 1e-9:
                        ratio = abs(ref_nobrake["mean"]) / abs(ref_full["mean"])
                        line += (
                            f"  -> 与「全长保持」同 kick 只差保持时长，位移比 {ratio:.2f}x"
                            f"（比值越小 => 限时截断越有效）"
                        )
                    lines.append(line)
        lines.append("")

        # 逐方向排序（正负步长都以「幅度」评价，故按 |mean| 接近 3.5 挑）
        valid = [c for c in cells if math.isfinite(c["mean"]) and math.isfinite(c["sd"])]
        lines.append("[1] 每个方向按「标准差升序」排序（前 6 名）")
        for ax in axes:
            for s in signs:
                lab = _dir_label(ax, s)
                grp = [c for c in valid if c["axis"] == ax and c["sign"] == s]
                if not grp:
                    continue
                grp.sort(key=lambda c: (c["sd"], abs(abs(c["mean"]) - 3.5)))
                lines.append(
                    f"  --- {lab} ---  tag  kick(us)  dt(ms)  hold(us)   mean+-sd (cnt)"
                    "           min/max        判定"
                )
                for c in grp[:6]:
                    lines.append(
                        f"     {c['tag']:<4} {c['kick_us']:5.1f}   {c['dt_cmd_ms']:5.1f}   "
                        f"{c['hold_us']:+6.1f}   "
                        f"{_fmt(c['mean']):>7} +-{_fmt_sd(c['sd']):<5}   "
                        f"{_fmt(c['min']):>7}/{_fmt(c['max']):<7}  "
                        f"{'PASS' if c['pass'] else ('REF' if str(c['tag']).startswith('REF') else 'fail')}"
                    )
        lines.append("")

        # --- 可达最小非零步长：B1 的核心能力指标（逐方向，按 |mean| 取最小） ---
        lines.append("[1b] B1 可达最小非零步长（逐方向）")
        floor_note = False
        for ax in axes:
            for s in signs:
                lab = _dir_label(ax, s)
                grp = [
                    c
                    for c in valid
                    if c["axis"] == ax
                    and c["sign"] == s
                    and not str(c["tag"]).startswith("REF")
                    and abs(c["mean"]) >= 1.0
                ]
                if not grp:
                    lines.append(f"  [{lab}] 没有 >=1 cnt 的非零格（全是静默区，或全都取了负号）")
                    continue
                nz = min(grp, key=lambda c: abs(c["mean"]))
                lines.append(
                    f"  [{lab}] 最小非零步长 = {_fmt(nz['mean'])} +-{_fmt_sd(nz['sd'])} cnt"
                    f"  (kick={nz['kick_us']:.0f}us dt={nz['dt_cmd_ms']:.0f}ms hold={nz['hold_us']:+.0f}us)"
                )
                if abs(nz["mean"]) > PASS_MEAN_HI:
                    floor_note = True
        lines.append(
            "    说明: 若同一 kick 在 15/20/25/30 ms 下均值几乎不变，则位移由 kick 幅度决定、"
            "与保持时长无关。"
        )
        if floor_note:
            lines.append(
                "     => 截断法存在硬地板：拿不到更小的确定性位移，需转 B2 折返差动步 / D 空中修正。"
            )
        lines.append("")

        # --- 双稳态检测：一端≈0（没越障）但另一端很大（越障后暴冲） ---
        bimodal: List[Dict] = []
        for c in valid:
            if str(c["tag"]).startswith("REF"):
                continue
            lo, hi = sorted([c["min"], c["max"]])
            if abs(lo) <= 0.5 and abs(hi) >= 8.0:
                bimodal.append(c)
        if bimodal:
            lines.append("[1c] 双稳态告警（部分试次完全不越障，越障了则一口吃掉大位移）")
            for c in bimodal:
                lines.append(
                    f"  {_dir_label(c['axis'], c['sign'])} kick={c['kick_us']:.0f}us "
                    f"dt={c['dt_cmd_ms']:.0f}ms hold={c['hold_us']:+.0f}us"
                    f"  -> min={_fmt(c['min'])} max={_fmt(c['max'])} 均值{_fmt(c['mean'])}"
                    f"  sd={_fmt_sd(c['sd'])}  => 仍是「全有或全无」，不是连续可调"
                )
            lines.append(
                "     含义: 该区域内提高 dt / kick 只能改变「越障概率」，不能改变「越障后位移幅度」。"
            )
            lines.append("")

        # 结论（逐方向）
        passers = [c for c in cells if c["pass"]]
        lines.append("=" * 108)
        if passers:
            lines.append("[结论 TIMED-KICK-PASS]  B1 成立，可拼进 _dual_side_refine()")
            all_dirs_ok = True
            for ax in axes:
                for s in signs:
                    lab = _dir_label(ax, s)
                    grp = [c for c in passers if c["axis"] == ax and c["sign"] == s]
                    if not grp:
                        all_dirs_ok = False
                        lines.append(
                            f"  [{lab}] 无 PASS 格 -> 该方向需单独调参"
                            f"（建议 --axis {ax} --holds=\"-8,-10,-12,-14,-16\" --both 复测）"
                        )
                        continue
                    grp.sort(key=lambda c: (c["sd"], abs(abs(c["mean"]) - 3.5)))
                    best = grp[0]
                    lines.append(
                        f"  [{lab}] 最优: kick={best['kick_us']:.0f}us / 保持 "
                        f"{best['dt_cmd_ms']:.0f}ms / 回拉 {best['hold_us']:+.0f}us"
                        f"  ->  d{ax}={_fmt(best['mean'])} +-{_fmt_sd(best['sd'])} cnt"
                        f" (min {_fmt(best['min'])} / max {_fmt(best['max'])}),"
                        f" 串扰 {_fmt(best['cross_mean'])} cnt, 蠕变 {_fmt(best['drift_mean'])} cnt"
                    )
                    if abs(best["cross_mean"]) > 3.0:
                        lines.append(
                            f"    !! 串扰 {_fmt(best['cross_mean'])} cnt 偏大，拼进控制器时需按 K_lin 反解补偿"
                        )
                    if abs(best["drift_mean"]) > 1.0:
                        lines.append(
                            f"    !! 刹停后仍有 {_fmt(best['drift_mean'])} cnt 蠕变，"
                            f"说明 {SETTLE_S:.2f}s 读数未完全稳定，拼进控制器时要留更长读点延迟"
                        )
            passers.sort(key=lambda c: (c["sd"], abs(abs(c["mean"]) - 3.5)))
            best = passers[0]
            lines.append(
                f"  下一步: 把 _dual_side_refine 的静态 for mag in (6.0, 9.0, 12.0) 阶梯替换为"
                f"「kick {best['kick_us']:.0f}us / 保持 {best['dt_cmd_ms']:.0f}ms / 回拉 {best['hold_us']:+.0f}us」双相脉冲，"
            )
            lines.append("          然后重跑 verify_hardware_kpi.py 的 Gate1/Gate2/Gate3。")
            if not all_dirs_ok:
                lines.append("  !! 仍有方向未通过：先补齐该方向参数，再拼接，禁止只挑好方向拼。")
            lines.append(
                "  备选 PASS 组: "
                + ", ".join(
                    f"({_dir_label(c['axis'], c['sign'])} k={c['kick_us']:.0f},dt={c['dt_cmd_ms']:.0f},"
                    f"h={c['hold_us']:+.0f}) sd={c['sd']:.2f}"
                    for c in passers[1:4]
                )
                if len(passers) > 1
                else "  备选 PASS 组: 无（仅此一组通过，建议 --both 复测另一方向）"
            )
        else:
            lines.append("[结论 TIMED-KICK-FAIL]  没有任何方向/组合落进 2~5 cnt 且 sd<=2 cnt")
            for ax in axes:
                for s in signs:
                    lab = _dir_label(ax, s)
                    grp = [c for c in valid if c["axis"] == ax and c["sign"] == s]
                    if not grp:
                        continue
                    small = [c for c in grp if abs(c["mean"]) < PASS_MEAN_LO]
                    big = [c for c in grp if abs(c["mean"]) > PASS_MEAN_HI]
                    noisy = [
                        c for c in grp if math.isfinite(c["sd"]) and c["sd"] > PASS_STD_MAX
                    ]
                    lines.append(
                        f"  [{lab}] n={len(grp)} 格 -> 偏小 {len(small)} / 偏大 {len(big)} / "
                        f"sd超标 {len(noisy)}"
                    )
            if valid:
                lines.append("  下一步单变量（本脚本已支持，无需改代码）：")
                lines.append(
                    "    (a) 反向制动:  --stage a --no-refs --dts 20 --kicks 45 "
                    "--holds=\"+4,0,-4,-8,-12,-16\" --runs 4"
                )
                lines.append(
                    "        （动摩擦滑动中下发反向 hold，理论上可在 ~2 cnt/us 上做确定性减速）"
                )
                lines.append(
                    "    (b) 亚越障 kick: --stage a --no-refs --kicks 24,28,32 "
                    "--dts 25,40 --holds +4 --runs 4"
                )
                lines.append(
                    "        （kick=35/dt=15 已出现 0,0,11 的双稳态，向下压才有机会进入 2~5 cnt）"
                )
            for key, (rq, rnb, rfl) in ref_pairs.items():
                if rnb and rfl and abs(rnb["mean"] - rfl["mean"]) < 3.0:
                    lines.append(
                        f"  关键提示[{_dir_label(*key)}]: REF-不回拉 与 REF-全长保持 结果接近 => "
                        "「切断驱动」没有起作用，不再是限时截断问题，"
                        "而是执行器本身惯性/回差主导，应转 B2/D"
                    )
        lines.append("=" * 108)

        report = "\n".join(lines)
        print("\n" + report)

        ana_path = "experiments/analysis_timed_kick.txt"
        with open(ana_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n[OK] CSV -> {csv_path}")
        print(f"[OK] 判读 -> {ana_path}")
        return 0

    finally:
        try:
            trainer.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
