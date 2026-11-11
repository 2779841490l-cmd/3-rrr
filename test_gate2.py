import sys
import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

print("=" * 72)
print("  GATE 2 COMPREHENSIVE TEST: 6-DIRECTION ALL-QUADRANT EVALUATION")
print("=" * 72)

directions = [(35, 0), (-35, 0), (0, 35), (0, -35), (30, 25), (-30, -25)]
dir_eval_errors = []

for dx, dy in directions:
    print(f"\nTarget ({dx:+3d}, {dy:+3d}):")
    # 6 warmup ILC learning shots
    for w in range(1, 7):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=True)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        corr = ctrl.ilc.get_correction(dx, dy)
        print(f"  Warmup {w}: Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err: {err:4.2f} cnt | ILC: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

    # 2 consecutive evaluation shots (NO cherry-picking!)
    eval_errs = []
    eval_finals = []
    for ev in range(1, 3):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
        eval_errs.append(res["final_error_counts"])
        eval_finals.append(res["final_delta"])

    avg_dir_err = float(np.mean(eval_errs))
    dir_eval_errors.append(avg_dir_err)
    status_str = "PASS" if avg_dir_err <= 3.0 else "FAIL"
    print(
        f"  [{status_str}] Eval Finals: {eval_finals[0]}, {eval_finals[1]} | "
        f"Mean Err: {avg_dir_err:4.2f} cnt ({avg_dir_err*7.9375:5.1f} um)"
    )

overall_mean = float(np.mean(dir_eval_errors))
print("\n" + "=" * 72)
print(f"Gate 2 Overall 6-Direction Mean Error: {overall_mean:.2f} cnt ({overall_mean*7.9375:.1f} um)")
print(f"Gate 2 Status: {'PASS' if overall_mean <= 2.0 and all(e <= 3.0 for e in dir_eval_errors) else 'FAIL'}")
print("=" * 72)

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

