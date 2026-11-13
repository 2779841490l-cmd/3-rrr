r"""probe_target_brake.py — 定案2e：传感器触发的目标制动微步（B1 的闭环版）

为什么需要它（全部来自上一轮实测，不是推测）
--------------------------------------------
probe_timed_kick.py 的 B1（大 kick + 定长回拉）实机结论：
  * X 主导方向  : kick=45us / dt=20ms / hold=-12us -> +4.25+-0.50 / -5.00+-0.00 cnt  PASS
  * Y 主导方向  : hold -4 / -8 / -12 全部停在 +13~+19 cnt，直到 -16 才骤降到 -1.50+-2.38
  即 Y 主导方向的「净位移 vs 回拉量」是一条「平台 + 悬崖」曲线，拐点附近 sd 2.4~5.3 cnt。
  => 开环固定幅度在 Y 主导方向不可能稳定拿到 2~5 cnt 且 sd<=2 cnt。

两个已坐实的根因
----------------
1) 步进方向错。_dual_side_refine 用 step_dir = K_trim @ err_vec。
   Y 主导残差时 K_trim 第 2 列 (-0.115,-0.285) 归一为 (-0.3745,-0.9282)，
   而真逆雅可比第 2 列 (-0.51146,-0.35898) 归一为 (-0.8195,-0.5751)，相差约 24 度。
   实测：沿前者走，位移里恒带约 +7.3 cnt 的寄生 X（串扰 -5.8~-15.3 cnt）；
         换到后者后串扰立刻降到 +1.8~+3.0 cnt，且预测屏幕方向 e=(0,+1.000) 完全命中。
2) 步幅错。同一个 mag(us) 在不同方向的屏幕增益差 1.6~2.6 倍，而 mag 是固定微秒数，
   于是 (6,9,12)us 阶梯在 Y 主导残差上实际是 15.6~31.2 cnt 的步长，
   而待修残差只有 3~8 cnt —— 最小步长已经大于残差，精修必然把点走坏。

本脚本的做法（B2/D 之间的最小可用形态：脉冲内闭环）
--------------------------------------------------
把「定长回拉」换成「传感器触发的目标制动」：
    kick（幅度足够越过 Stribeck 门槛）
      -> 每 ~1 ms 轮询一次鼠标原始计数
      -> 一旦沿 e 的位移达到 target_cnt，立刻切到 hold（默认 0us = 回到 u0，断驱动）
      -> 静置读数
方向各向异性、正负号不对称、play 半径差异全部由闭环自动吸收，
不再需要逐方向标定表。

对照组
------
--targets 里给 0.0 等于「kick 后立刻断电」，即 REF：证明不闭环时步长有多大。

判定
----
逐 (方向, 符号) 统计稳态净位移：均值绝对值 ∈ [2,5] cnt 且 sd <= 2.0 cnt 记为 PASS。
只有全部枚举到的方向/符号都拿到 PASS，才允许拼进 _dual_side_refine。
"""

import argparse
import csv
import math
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple

from rect_manifold_calibrator import HardwareRectangleTrainer

# ---------------- 时序常量（与 probe_timed_kick.py 保持一致，便于对比） ----------------
PRELOAD_US = 35.0          # 反向预载量，与 move_to_rect_origin() 同款
PRELOAD_HOLD_S = 0.22
ORIGIN_SETTLE_S = 0.36
SETTLE_S = 0.46            # 断电后静置
DRIFT_S = 0.26             # 蠕变观察窗
BRAKE_TIMEOUT_MS = 140.0   # 轮询超时保护，防止空转到天荒地老
DRIVE_CAP_MS = 100.0       # 【P0-B 升级】原 40ms 短于 MG996R ~35ms 起动延迟，放宽到 100ms 覆盖完整弹道窗
DEFAULT_TAU_BRAKE_MS = 16.0  # 预测落点超前制动时间常数 tau_brake (ms)

# ---------------- 判定 ---------------- 
PASS_MEAN_LO, PASS_MEAN_HI = 2.0, 5.0
PASS_STD_MAX = 2.0

# 实测 2x2 雅可比（计数/us）与逆雅可比（us/计数），取自 定案4 experiments/real_jacobian_2x2.csv
JAC_CENTER_CNT_PER_US = ((1.8925, -1.8915), (-0.5298, -3.8972))
JAC_INV_CENTER_US_PER_CNT = ((0.46520, -0.22578), (-0.06323, -0.22590))


def _norm(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-12:
        return 1.0, 0.0
    return vx / n, vy / n


def _screen_dir(vx: float, vy: float) -> Tuple[float, float]:
    """u 空间单位向量 -> 屏幕（计数）单位方向。"""
    ex = JAC_CENTER_CNT_PER_US[0][0] * vx + JAC_CENTER_CNT_PER_US[0][1] * vy
    ey = JAC_CENTER_CNT_PER_US[1][0] * vx + JAC_CENTER_CNT_PER_US[1][1] * vy
    return _norm(ex, ey)


def _dir_vec(name: str) -> Tuple[float, float]:
    """方向名 -> u 空间单位向量。默认用真逆雅可比列（物理上正确的消残差方向）。"""
    if name == "x":
        # 逆雅可比第 0 列 = 屏幕 +X 对应的 u 方向
        return _norm(JAC_INV_CENTER_US_PER_CNT[0][0], JAC_INV_CENTER_US_PER_CNT[1][0])
    if name == "y":
        # 逆雅可比第 1 列 = 屏幕 +Y 对应的 u 方向
        return _norm(JAC_INV_CENTER_US_PER_CNT[0][1], JAC_INV_CENTER_US_PER_CNT[1][1])
    if name == "xtrim":
        # 仅供对照：_dual_side_refine 现用的 K_trim 第 0 列
        return _norm(0.58, -0.038)
    if name == "ytrim":
        # 仅供对照：_dual_side_refine 现用的 K_trim 第 1 列（已证明方向错 24 度）
        return _norm(-0.115, -0.285)
    raise SystemExit(f"未知方向名: {name}（可选 x/y/xtrim/ytrim）")


# ---------------- 底层时序 ----------------
def _send(trainer, ux: float, uy: float) -> float:
    t0 = time.perf_counter()
    trainer.driver.send_raw_pulses(float(ux), float(uy))
    return time.perf_counter() - t0


def _stats(vals: List[float]) -> Tuple[float, float, float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return m, sd, min(vals), max(vals)


def _is_pass(mean: float, sd: float) -> bool:
    return (
        math.isfinite(mean)
        and math.isfinite(sd)
        and PASS_MEAN_LO <= abs(mean) <= PASS_MEAN_HI
        and sd <= PASS_STD_MAX
    )


def _trial(
    trainer,
    u0x: float,
    u0y: float,
    kick_us: float,
    sign: float,
    vec: Tuple[float, float],
    escreen: Tuple[float, float],
    target_cnt: float,
    hold_us: float,
    timeout_ms: float,
    cap_ms_limit: float = DRIVE_CAP_MS,
    tau_brake_ms: float = DEFAULT_TAU_BRAKE_MS,
    brake_ms: float = 0.0,
) -> Dict[str, float]:
    """一次「预载 -> 静基线 -> kick -> 预测落点触发断电 -> 静置读数」。"""

    def _u(delta: float) -> Tuple[float, float]:
        return u0x + delta * vec[0], u0y + delta * vec[1]

    # 1) 确定性预载：反向 35us 再回 u0，锁定齿面
    _send(trainer, *_u(-PRELOAD_US * sign))
    time.sleep(PRELOAD_HOLD_S)
    _send(trainer, *_u(0.0))
    time.sleep(ORIGIN_SETTLE_S)

    trainer.sensor.recenter_cursor()
    p0 = trainer.get_rect_position()
    esx, esy = escreen

    # 2) kick 相：下发即开始计时
    t0 = time.perf_counter()
    _send(trainer, *_u(sign * kick_us))

    # 3) 轮询：用 ~5ms 滑窗速度计算预测落点 pred_proj = proj + tau_brake * vel_proj
    #    满足 pred_proj >= target_cnt 且已真实起步 (proj >= 0.5) 时立刻触发制动
    proj_brake = 0.0
    pred_brake = 0.0
    vel_brake = 0.0
    n_poll = 0
    hit_cap = 0.0
    cap_ms = min(float(timeout_ms), float(cap_ms_limit), DRIVE_CAP_MS)
    tau_s = max(0.0, float(tau_brake_ms)) / 1000.0
    hist: List[Tuple[float, float]] = [(t0, 0.0)]

    if target_cnt > 0.0:
        while True:
            now = time.perf_counter()
            if (now - t0) * 1000.0 >= cap_ms:
                break
            p = trainer.get_rect_position()
            n_poll += 1
            proj_brake = (
                (float(p[0]) - float(p0[0])) * esx
                + (float(p[1]) - float(p0[1])) * esy
            ) * sign
            hist.append((now, proj_brake))
            # 取 >= 5ms 前的最早有效点算滑窗速度，消除 USB 帧间 0/尖峰噪声
            while len(hist) > 2 and (now - hist[1][0]) >= 0.005:
                hist.pop(0)
            dt_v = now - hist[0][0]
            vel_brake = (proj_brake - hist[0][1]) / dt_v if dt_v >= 0.003 else 0.0
            pred_brake = proj_brake + tau_s * max(0.0, vel_brake)

            if proj_brake >= target_cnt or (proj_brake >= 0.5 and vel_brake > 15.0 and pred_brake >= target_cnt):
                break
            time.sleep(0.0005)
    t_brake_ms = (time.perf_counter() - t0) * 1000.0
    if target_cnt > 0.0 and pred_brake < target_cnt and proj_brake < target_cnt:
        hit_cap = 1.0

    # 4) catch 相：切到 hold（默认 0us => 回到 u0，撤掉驱动；若 hold<0 且 brake_ms>0，则反刹 brake_ms 后回 u0）
    _send(trainer, *_u(sign * hold_us))
    if hold_us != 0.0 and brake_ms > 0.0:
        time.sleep(brake_ms / 1000.0)
        _send(trainer, *_u(0.0))

    # 5) 稳态读数 + 蠕变窗
    time.sleep(SETTLE_S)
    p1 = trainer.get_rect_position()
    time.sleep(DRIFT_S)
    p2 = trainer.get_rect_position()

    dx1 = float(p1[0]) - float(p0[0])
    dy1 = float(p1[1]) - float(p0[1])
    dx2 = float(p2[0]) - float(p0[0])
    dy2 = float(p2[1]) - float(p0[1])
    return {
        "prim": dx1 * esx + dy1 * esy,
        "cross": -dx1 * esy + dy1 * esx,
        "drift": (dx2 - dx1) * esx + (dy2 - dy1) * esy,
        "t_brake_ms": t_brake_ms,
        "n_poll": float(n_poll),
        "proj_brake": proj_brake,
        "pred_brake": pred_brake,
        "vel_brake": vel_brake,
        "hit_cap": hit_cap,
    }


def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="定案2e · 传感器触发目标制动微步（脉冲内闭环）")
    ap.add_argument("--dirs", type=str, default="x,y", help="方向名，逗号分隔：x,y,xtrim,ytrim")
    ap.add_argument("--both", action="store_true", default=True, help="正负双向都测（默认开）")
    ap.add_argument("--one-way", dest="both", action="store_false", help="只测 +v")
    ap.add_argument("--kicks", type=str, default="30,45", help="kick 幅度 us，逗号分隔")
    ap.add_argument("--targets", type=str, default="0.0,2.0,3.0", help="触发阈值 cnt，0=kick 后立即断电（REF 对照）")
    ap.add_argument("--hold-us", type=float, default=0.0, help="断电时的 hold 量 us（0=回 u0 撤驱动，负数=反向制动）")
    ap.add_argument("--runs", type=int, default=4, help="每格重复次数")
    ap.add_argument("--timeout-ms", type=float, default=BRAKE_TIMEOUT_MS, help="轮询超时保护")
    ap.add_argument("--cap-ms", type=float, default=DRIVE_CAP_MS, help="kick 驱动硬上限 ms (<=100)")
    ap.add_argument("--tau-brake-ms", type=float, default=DEFAULT_TAU_BRAKE_MS, help="预测落点超前时间常数 ms (0=仅实测位移触发)")
    ap.add_argument("--brake-ms", type=float, default=0.0, help="若>0且hold_us!=0，施加hold反刹brake_ms毫秒后再回u0")
    ap.add_argument("--asym-brake", action="store_true", help="按定案4正负斜率比(u2+4.36/u2-3.43=1.27x)对y-自动放大反刹hold")
    ap.add_argument("--csv", type=str, default=os.path.join("experiments", "target_brake.csv"), help="输出 CSV 路径")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    kicks = _floats(args.kicks)
    targets = _floats(args.targets)
    signs = (1.0, -1.0) if args.both else (1.0,)

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()   # 关键：HardwareRectangleTrainer.__init__ 不启动传感器线程
    time.sleep(0.30)
    t_start = time.perf_counter()
    try:
        trainer.move_to_rect_origin(settle_s=0.45)
        manifold = trainer.manifold
        u0 = manifold.evaluate_kinematic_pulse(0.0, 0.0)
        u0x, u0y = float(u0[0]), float(u0[1])

        print("=" * 108)
        print("  定案2e · 目标触发制动微步（脉冲内闭环）—— 目标：可控 2~5 cnt，sd <= 2 cnt")
        print("=" * 108)
        print(
            f"  raw 基线 u0 = ({u0x:.2f}, {u0y:.2f}) us   预载 {PRELOAD_US:.0f}us   "
            f"runs={args.runs}   hold={args.hold_us:+.1f}us   cap={args.cap_ms:.0f}ms   tau_brake={args.tau_brake_ms:.1f}ms"
        )
        for d in dirs:
            v = _dir_vec(d)
            e = _screen_dir(*v)
            print(
                f"  方向 {d:>5s}: v=({v[0]:+.4f},{v[1]:+.4f})  e=({e[0]:+.4f},{e[1]:+.4f})  "
                f"屏幕增益 |J*v|={math.hypot(*[JAC_CENTER_CNT_PER_US[i][0]*v[0]+JAC_CENTER_CNT_PER_US[i][1]*v[1] for i in (0,1)]):.3f} cnt/us"
            )
        print(f"  kicks={kicks}  targets={targets}  signs={signs}")
        print("-" * 108)

        rows: List[Dict] = []
        for d in dirs:
            v = _dir_vec(d)
            e = _screen_dir(*v)
            for sign in signs:
                # 定案4 实测 u2+ (4.36 cnt/us) vs u2- (3.43 cnt/us) 存在 1.27x 非对称性，
                # y- 沿 u2+ 加速、沿 u2- 反刹，故开启 --asym-brake 时按实测 1.27x 补偿反刹脉宽
                hold_eff = args.hold_us * 1.27 if (args.asym_brake and d == "y" and sign < 0 and args.hold_us < 0) else args.hold_us
                for kick in kicks:
                    for tg in targets:
                        for r in range(args.runs):
                            res = _trial(
                                trainer, u0x, u0y, kick, sign, v, e,
                                tg, hold_eff, args.timeout_ms,
                                cap_ms_limit=args.cap_ms, tau_brake_ms=args.tau_brake_ms,
                                brake_ms=args.brake_ms,
                            )
                            rows.append(
                                {
                                    "dir": d,
                                    "sign": sign,
                                    "kick_us": kick,
                                    "target_cnt": tg,
                                    "hold_us": round(hold_eff, 2),
                                    "tau_brake_ms": args.tau_brake_ms,
                                    "run_id": r,
                                    "prim_cnt": res["prim"],
                                    "cross_cnt": res["cross"],
                                    "drift_cnt": res["drift"],
                                    "t_brake_ms": res["t_brake_ms"],
                                    "proj_brake_cnt": res["proj_brake"],
                                    "pred_brake_cnt": round(res["pred_brake"], 2),
                                    "vel_brake_cnt_s": round(res["vel_brake"], 1),
                                    "hit_cap": int(res["hit_cap"]),
                                    "n_poll": res["n_poll"],
                                }
                            )
                        grp = [
                            x["prim_cnt"] for x in rows
                            if x["dir"] == d and x["sign"] == sign
                            and x["kick_us"] == kick and x["target_cnt"] == tg
                        ]
                        crg = [
                            x["cross_cnt"] for x in rows
                            if x["dir"] == d and x["sign"] == sign
                            and x["kick_us"] == kick and x["target_cnt"] == tg
                        ]
                        tb = [
                            x["t_brake_ms"] for x in rows
                            if x["dir"] == d and x["sign"] == sign
                            and x["kick_us"] == kick and x["target_cnt"] == tg
                        ]
                        m, sd, lo, hi = _stats(grp)
                        cm, csd, _, _ = _stats(crg)
                        tm, _, _, _ = _stats(tb)
                        flag = "PASS" if _is_pass(m, sd) else "    "
                        print(
                            f"  [{flag}] {d}{'+' if sign > 0 else '-'} kick={kick:5.1f}us "
                            f"target={tg:4.1f}cnt -> prim={m:+7.2f} +-{sd:5.2f} cnt "
                            f"(min {lo:+6.2f}/max {hi:+6.2f})  串扰={cm:+7.2f} +-{csd:5.2f}  "
                            f"t_brake={tm:5.1f}ms"
                        )

        # ---------------- 落盘 ----------------
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        csv_path = args.csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # ---------------- 判读 ----------------
        print()
        print("=" * 108)
        print(f"  定案2e · 目标触发制动判读   用时 {time.perf_counter() - t_start:.0f}s")
        print("=" * 108)

        all_ok = True
        lines = []
        for d in dirs:
            for sign in signs:
                lab = f"{d}{'+' if sign > 0 else '-'}"
                grp = [r for r in rows if r["dir"] == d and r["sign"] == sign]
                # 按 (kick, target) 聚合
                cells: Dict[Tuple[float, float], List[Dict]] = {}
                for r in grp:
                    cells.setdefault((r["kick_us"], r["target_cnt"]), []).append(r)
                cell_stats = []
                for (kick, tg), rs in cells.items():
                    m, sd, lo, hi = _stats([x["prim_cnt"] for x in rs])
                    cm, csd, _, _ = _stats([x["cross_cnt"] for x in rs])
                    tm, _, _, _ = _stats([x["t_brake_ms"] for x in rs])
                    cell_stats.append(
                        {"kick": kick, "target": tg, "mean": m, "sd": sd,
                         "lo": lo, "hi": hi, "cm": cm, "csd": csd, "tm": tm,
                         "pass": _is_pass(m, sd)}
                    )
                cell_stats.sort(key=lambda c: (c["sd"], abs(abs(c["mean"]) - 3.5)))
                print(f"\n  --- {lab} ---")
                print("    kick(us)  target(cnt)   prim(mean+-sd)          min/max        串扰       t_brake  判定")
                for c in cell_stats:
                    print(
                        f"    {c['kick']:7.1f} {c['target']:10.1f}   "
                        f"{c['mean']:+7.2f} +-{c['sd']:5.2f} cnt   "
                        f"{c['lo']:+6.2f}/{c['hi']:+6.2f}   "
                        f"{c['cm']:+7.2f}   {c['tm']:5.1f}ms  "
                        f"{'PASS' if c['pass'] else 'fail'}"
                    )
                passes = [c for c in cell_stats if c["pass"]]
                if passes:
                    b = passes[0]
                    chosen = b
                else:
                    chosen = cell_stats[0] if cell_stats else None
                if chosen is None:
                    continue
                ok = bool(passes)
                all_ok = all_ok and ok
                if ok:
                    print(
                        f"    -> 采用: kick={chosen['kick']:.1f}us / target={chosen['target']:.1f}cnt "
                        f"/ hold={args.hold_us:+.1f}us => {chosen['mean']:+.2f} +-{chosen['sd']:.2f} cnt"
                    )
                else:
                    print(
                        f"    -> 无 PASS 格；最接近: {chosen['mean']:+.2f} +-{chosen['sd']:.2f} cnt "
                        f"(kick={chosen['kick']:.1f}us target={chosen['target']:.1f}cnt)"
                    )
                lines.append(f"{lab}: " + ("PASS" if ok else "FAIL"))

        print()
        print("=" * 108)
        if all_ok:
            print("[结论 TARGET-BRAKE-PASS] 所有枚举方向/符号均有 PASS 格，可拼进 _dual_side_refine")
        else:
            print("[结论 TARGET-BRAKE-FAIL] 仍有方向/符号无 PASS 格")
            print("  !! 先补齐该方向参数，再拼接，禁止只挑好方向拼。")
        for l in lines:
            print("   ", l)
        print("=" * 108)
        print(f"\n[OK] CSV -> {csv_path}")
        return 0 if all_ok else 2
    finally:
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
