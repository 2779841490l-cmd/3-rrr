import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

print("Testing Target (-35, 0) with FULL 2D ILC update:")

dx, dy = -35, 0
for w in range(1, 7):
    ctrl.driver.center(settle_s=0.38, preload=True)
    res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
    fd = res['final_delta']
    err = res['final_error_counts']
    err_vec = np.array([float(dx - fd[0]), float(dy - fd[1])])
    K = ctrl.poly_jac.evaluate(0, 0)
    # Full 2D update
    du_step = 0.35 * (K @ err_vec) if err > 1.6 else np.array([0.0, 0.0])
    du_step = np.clip(du_step, -3.5, 3.5)
    cur_corr = ctrl.ilc.get_correction(dx, dy)
    new_corr = cur_corr + du_step
    ctrl.ilc.params["target_table"][f"{dx}_{dy}"] = [round(float(new_corr[0]), 3), round(float(new_corr[1]), 3)]
    print(f"  Warmup {w}: Final {fd} | Err: {err:.2f} cnt | ILC: [{new_corr[0]:+.2f}, {new_corr[1]:+.2f}]")

print("Evaluating 2 test shots:")
eval_errs = []
for ev in range(1, 3):
    ctrl.driver.center(settle_s=0.38, preload=True)
    res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
    fd = res['final_delta']
    err = res['final_error_counts']
    eval_errs.append(err)
    print(f"  Eval {ev}: Final {fd} | Err: {err:.2f} cnt ({err*7.9375:.1f} um)")

print(f"Mean Eval Error: {np.mean(eval_errs):.2f} cnt (Gate 2 limit is <= 3.0 cnt)")

ctrl.driver.center(settle_s=0.38, preload=True)
ctrl.close()

