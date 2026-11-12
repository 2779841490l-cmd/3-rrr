"""
test_dual_lattice_grid.py — 同口径 5x5 网格 A/B：Phase 3 双侧晶格精修 开 / 关

目的（预登记判据，先写死再看结果）：
  基线（test_band_relaxation.py 同参）：两轮 mean 5.145 cnt，8 轮间 s=1.09，
  逐点 RMSE 8.68 cnt，elapsed 中位数 3.55 s（多点撞 3400 ms 饱和）。

  A/B 必须在**同一场次内**完成（同一 hyst 状态、同一温度），否则热漂移会把
  差值吃掉。故本脚本用 modes 字符串控制先后：
      modes="10" -> 第1轮 dual=ON，第2轮 dual=OFF（默认，推荐）
      modes="01" -> 第1轮 dual=OFF，第2轮 dual=ON（顺序对照，排除单调漂移）
      modes="11" -> 两轮都 ON（看 ON 自身的轮间一致性）

  判据：
    PASS 条件（三条全中）：
      (1) dual=ON 两轮 mean 均 < 4.0 cnt（基线 5.145，需见到 >1 cnt 改善）；
      (2) dual=ON 的 max_err 两轮均 < 15 cnt（基线 max 28.15 由 y=127.5 行造成）；
      (3) elapsed 中位数不高于基线 3.55 s（不允许用时间换精度）。
    若经 dual 修完仍 >= 4.5 cnt，则"同次访问内双侧选优"这条路在本执行器上判死，
    直接进【定案4】换件，不再追加任何学习算法投入。

用法：
  set PYTHONIOENCODING=utf-8 && chcp 65001 >nul
  "E:\\anaconda\\envs\\pytorch\\python.exe" test_dual_lattice_grid.py 10

输出：
  experiments/grid_dual_<ts>_r1.csv / _r2.csv（含 dual_trials / dual_gain_cnt）
退出码: 0 = 全 PASS   2 = 未达标
"""

import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer


FIELDNAMES = [
    "round", "dual_lattice", "start", "target", "final",
    "err_counts", "err_um", "straight_dev_cnt", "straight_dev_pct",
    "ballistic_ms", "elapsed_ms", "dual_trials", "dual_gain_cnt", "u_kin_final",
]

TOL_COUNTS = 1.2          # 与 test_band_relaxation.py 基线完全一致
BASE_MEAN = 5.145         # 基线两轮均值
BASE_MAX = 28.15
BASE_ELAPSED_MED = 3550.0  # ms

PASS_MEAN = 4.0
PASS_MAX = 15.0


def run_one_round(trainer, round_id: int, dual: bool):
    """跑一轮 5x5 蛇形网格，返回 results 列表。"""
    trainer.move_to_rect_origin(settle_s=0.45)
    W, H = 0.85 * trainer.manifold.W, 0.85 * trainer.manifold.H
    xs = np.linspace(-W, W, 5)
    ys = np.linspace(-H, H, 5)

    test_pts = []
    for j, y in enumerate(ys):
        row_x = xs if j % 2 == 0 else xs[::-1]
        for x in row_x:
            test_pts.append((round(float(x), 1), round(float(y), 1)))

    tag = "DUAL" if dual else "BASE"
    print("\n" + "=" * 96)
    print(f"[Round {round_id}] dual_lattice={dual} ({tag})  tol_counts={TOL_COUNTS}")
    print("=" * 96)

    results = []
    errs = []
    for idx, (tx, ty) in enumerate(test_pts, 1):
        r = trainer.move_point_to_point(tx, ty, tol_counts=TOL_COUNTS, dual_lattice=dual)
        errs.append(float(r["err_counts"]))
        results.append(r)
        print(
            f"Pt #{idx:02d}/25 {r['start']} -> Tgt{r['target']} | Final{r['final']} | "
            f"Err: {r['err_counts']:5.2f} cnt ({r['err_um']:5.1f} um) | "
            f"dual_trials: {r.get('dual_trials', 0):2d} gain: {r.get('dual_gain_cnt', 0.0):5.2f} | "
            f"bal: {r['ballistic_ms']:5.1f}ms elapsed: {r['elapsed_ms']:7.1f}"
        )

    mean_err = float(np.mean(errs))
    max_err = float(np.max(errs))
    elapsed_med = float(np.median([r["elapsed_ms"] for r in results]))
    print(f"\n[{tag}] mean_err = {mean_err:.3f} cnt   max_err = {max_err:.2f} cnt   "
          f"elapsed_median = {elapsed_med:.1f} ms")
    print(f"[{tag}] 逐点误差: {[round(e, 2) for e in errs]}")

    os.makedirs("experiments", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"experiments/grid_dual_{ts}_r{round_id}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            row["round"] = round_id
            row["dual_lattice"] = 1 if dual else 0
            w.writerow(row)
    print(f"[{tag}] CSV: {os.path.abspath(out_csv)}")

    return {"dual": dual, "mean": mean_err, "max": max_err,
            "elapsed_med": elapsed_med, "errs": errs, "csv": out_csv}


def main() -> int:
    modes = sys.argv[1] if len(sys.argv) > 1 else "10"
    print("=" * 96)
    print("test_dual_lattice_grid.py — Phase 3 双侧晶格精修 A/B（同场次）")
    print(f"modes = {modes}   ('1'=dual ON, '0'=dual OFF)")
    print(f"基线: mean {BASE_MEAN} / max {BASE_MAX} / elapsed_med {BASE_ELAPSED_MED:.0f} ms")
    print("=" * 96)

    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    t_start = time.perf_counter()
    rounds = []
    try:
        for i, ch in enumerate(modes, 1):
            rounds.append(run_one_round(trainer, i, ch == "1"))
    finally:
        trainer.close()

    total_s = time.perf_counter() - t_start

    print("\n" + "=" * 96)
    print(f"汇总（总耗时 {total_s/60:.1f} min）")
    print("=" * 96)
    print(f"{'round':>5} {'dual':>5} {'mean':>8} {'max':>8} {'elapsed_med':>12}  vs 基线")
    for r in rounds:
        d = r["mean"] - BASE_MEAN
        print(f"{rounds.index(r)+1:>5} {str(r['dual']):>5} {r['mean']:>8.3f} {r['max']:>8.2f} "
              f"{r['elapsed_med']:>12.1f}  {d:+.3f} cnt")

    duals = [r for r in rounds if r["dual"]]
    bases = [r for r in rounds if not r["dual"]]
    if duals and bases:
        dm = float(np.mean([r["mean"] for r in duals]))
        bm = float(np.mean([r["mean"] for r in bases]))
        print(f"\n同场次 A/B: DUAL {dm:.3f}  vs  BASE {bm:.3f}   =>  差 {dm-bm:+.3f} cnt")
    elif duals:
        dm = float(np.mean([r["mean"] for r in duals]))
        print(f"\n仅 DUAL 臂: {dm:.3f} cnt（对比历史基线 {BASE_MEAN}）")

    if not duals:
        print("\n[WARN] 未跑 dual=ON，无法判定。退出码 3。")
        return 3

    ok_mean = all(r["mean"] < PASS_MEAN for r in duals)
    ok_max = all(r["max"] < PASS_MAX for r in duals)
    ok_time = all(r["elapsed_med"] <= BASE_ELAPSED_MED for r in duals)

    print("\n" + "-" * 96)
    print(f"(1) mean < {PASS_MEAN}: {'PASS' if ok_mean else 'FAIL'}   "
          f"(2) max < {PASS_MAX}: {'PASS' if ok_max else 'FAIL'}   "
          f"(3) elapsed_med <= {BASE_ELAPSED_MED:.0f}ms: {'PASS' if ok_time else 'FAIL'}")
    if ok_mean and ok_max and ok_time:
        print("[DUAL LATTICE A/B PASSED] 双侧晶格精修是真实改善，可进入 KPI 门禁复跑。")
        return 0
    print("[DUAL LATTICE A/B FAILED] 双侧晶格精修不足以达标 => 进【定案4】换件。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
