import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

test_dirs = [(-35, 0), (-30, -25), (35, 0), (0, 35)]
for dx, dy in test_dirs:
    print(f"\n--- Testing ({dx:+3d}, {dy:+3d}) ---")
    for w in range(1, 6):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=True)
        print(f"  Warmup {w}: final={res['final_delta']}, err={res['final_error_counts']:.2f}")

    eval_errs = []
    eval_finals = []
    for ev in range(1, 3):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
        eval_errs.append(res["final_error_counts"])
        eval_finals.append(res["final_delta"])
    avg_e = np.mean(eval_errs)
    print(f"  Result: finals={eval_finals}, mean_err={avg_e:.2f} cnt -> {'PASS' if avg_e <= 3.0 else 'FAIL'}")

ctrl.driver.center(settle_s=0.38, preload=True)
ctrl.close()
