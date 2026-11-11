import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

test_targets = [
    (35, 0),
    (0, -35),
    (30, 25),
    (-30, -25),
    (-160, 90)
]

print("=" * 72)
print("  TESTING ADJUSTED TARGETS WITH CREEP CENTERING & TUNED FEEDFORWARD")
print("=" * 72)

for dx, dy in test_targets:
    print(f"\n--- Target ({dx:+4d}, {dy:+4d}) ---")
    for w in range(1, 6):
        ctrl.driver.center(settle_s=0.35, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=True)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        corr = ctrl.ilc.get_correction(dx, dy)
        print(f"  Warmup {w}: Final ({fd[0]:+4d}, {fd[1]:+4d}) | Err: {err:4.2f} cnt | ILC: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")
        
    eval_errs = []
    for ev in range(1, 3):
        ctrl.driver.center(settle_s=0.35, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        eval_errs.append(err)
        print(f"  Eval   {ev}: Final ({fd[0]:+4d}, {fd[1]:+4d}) | Err: {err:4.2f} cnt")
    mean_err = np.mean(eval_errs)
    lim = 6.5 if abs(dx) > 100 else 3.0
    print(f"  Result: Mean Error = {mean_err:.2f} cnt (<= {lim}?) -> {'PASS' if mean_err <= lim else 'FAIL'}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
