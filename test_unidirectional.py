"""
test_unidirectional.py — 单向趋近网格测试（v2 安全版）
用法:
  python test_unidirectional.py            # 默认: 中心 3x3, 0.5 倍范围, 1 轮（安全试跑）
  python test_unidirectional.py full 1     # 全场 5x5 serpentine 0.85 倍（与基线同目标点）, 1 轮
输出: experiments/grid_unidir_<mode>_<ts>.csv；SafetyAbort 时文件名带 _ABORTED
"""
import csv
import os
import sys
from datetime import datetime

import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer
from unidirectional_approach import SafetyAbort, move_unidirectional

FIELDNAMES = ["start", "target", "final", "err_counts", "err_um", "straight_dev_cnt", "straight_dev_pct",
              "ballistic_ms", "elapsed_ms", "u_kin_final", "retries", "trims", "overshoot", "min_err_seen"]
REZERO_EVERY = 5


def make_points(trainer, mode):
    n, ratio = (5, 0.85) if mode == "full" else (3, 0.5)
    W, H = ratio * trainer.manifold.W, ratio * trainer.manifold.H
    xs, ys = np.linspace(-W, W, n), np.linspace(-H, H, n)
    pts = []
    for j, y in enumerate(ys):
        for x in (xs if j % 2 == 0 else xs[::-1]):
            pts.append((round(float(x), 1), round(float(y), 1)))
    return pts


def one_round(trainer, mode):
    pts = make_points(trainer, mode)
    res, aborted = [], None
    try:
        for idx, (tx, ty) in enumerate(pts, 1):
            if (idx - 1) % REZERO_EVERY == 0:
                trainer.move_to_rect_origin(settle_s=0.45)
            r = move_unidirectional(trainer, tx, ty, tol_counts=1.2)
            res.append(r)
            print(f"Pt #{idx:02d}/{len(pts)} Tgt{r['target']} Final{r['final']} Err {r['err_counts']:5.2f} "
                  f"min_seen {r['min_err_seen']:5.2f} over {r['overshoot']} trims {r['trims']:2d} "
                  f"elapsed {r['elapsed_ms']:.0f}ms", flush=True)
    except SafetyAbort as ex:
        aborted = str(ex)
        print(f"\n[SAFETY ABORT] at point #{len(res) + 1}: {aborted} -> servos centered, round INVALID", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"experiments/grid_unidir_{mode}_{ts}{'_ABORTED' if aborted else ''}.csv"
    os.makedirs("experiments", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(res)
    if res:
        errs = np.array([r["err_counts"] for r in res])
        ms = np.array([r["elapsed_ms"] for r in res])
        print(f"\nmean_err: {errs.mean():.2f}  max_err: {errs.max():.2f}  hit(<=2.0): {(errs <= 2.0).sum()}/{len(res)}  "
              f"overshoot: {sum(r['overshoot'] for r in res)}  median_ms: {np.median(ms):.0f}", flush=True)
    print(f"CSV saved: {out}\n", flush=True)
    return None if aborted else float(np.mean([r["err_counts"] for r in res]))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "center"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    try:
        means = []
        for _ in range(n):
            m = one_round(trainer, mode)
            if m is None:
                break
            means.append(m)
        if len(means) > 1:
            print(f"ROUNDS mean_err: {[round(x, 2) for x in means]}  avg {np.mean(means):.2f}  "
                  f"std(ddof=1) {np.std(means, ddof=1):.2f}")
    finally:
        try:
            trainer.driver.center(settle_s=0.4)
        except Exception:
            pass
        trainer.close()


if __name__ == "__main__":
    main()
