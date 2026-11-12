"""
test_band_relaxation.py — Single 5x5 serpentine grid evaluation (no Gate 1/3, no weight updates).
"""

import csv
import os
from datetime import datetime
import numpy as np

from rect_manifold_calibrator import HardwareRectangleTrainer


FIELDNAMES = [
    "start",
    "target",
    "final",
    "err_counts",
    "err_um",
    "straight_dev_cnt",
    "straight_dev_pct",
    "ballistic_ms",
    "elapsed_ms",
    "u_kin_final",
]


def main():
    trainer = HardwareRectangleTrainer()
    trainer.sensor.start()
    try:
        trainer.move_to_rect_origin(settle_s=0.45)
        W, H = 0.85 * trainer.manifold.W, 0.85 * trainer.manifold.H
        xs = np.linspace(-W, W, 5)
        ys = np.linspace(-H, H, 5)

        test_pts = []
        for j, y in enumerate(ys):
            row_x = xs if j % 2 == 0 else xs[::-1]
            for x in row_x:
                test_pts.append((round(float(x), 1), round(float(y), 1)))

        grid_results = []
        grid_errs = []
        for idx, (tx, ty) in enumerate(test_pts, 1):
            r = trainer.move_point_to_point(tx, ty, tol_counts=1.2)
            grid_errs.append(r["err_counts"])
            grid_results.append(r)
            print(
                f"Pt #{idx:02d}/25 {r['start']} -> Tgt{r['target']} | Final{r['final']} | "
                f"Err: {r['err_counts']:4.2f} cnt ({r['err_um']:4.1f} um) | "
                f"Ballistic: {r['ballistic_ms']:4.1f}ms | elapsed_ms: {r['elapsed_ms']:.1f}"
            )

        os.makedirs("experiments", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = f"experiments/grid_bandfresh_{ts}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(grid_results)

        mean_err = float(np.mean(grid_errs))
        max_err = float(np.max(grid_errs))
        print(f"\nmean_err: {mean_err:.2f} cnt")
        print(f"max_err: {max_err:.2f} cnt")
        print("elapsed_ms per point:", [r["elapsed_ms"] for r in grid_results])
        print(f"CSV saved: {os.path.abspath(out_csv)}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
