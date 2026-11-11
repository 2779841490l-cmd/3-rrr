import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

dx, dy = -35, 0
print(f"\n--- Testing ({dx:+3d}, {dy:+3d}) with du_base[0]*=1.16, du_base[1]=-2.5 ---")
for w in range(1, 7):
    ctrl.driver.center(settle_s=0.38, preload=True)
    res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=True)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    corr = ctrl.ilc.get_correction(dx, dy)
    print(f"  Warmup {w}: Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err: {err:4.2f} cnt | ILC: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")
    
eval_errs = []
for ev in range(1, 3):
    ctrl.driver.center(settle_s=0.38, preload=True)
    res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    eval_errs.append(err)
    print(f"  Eval {ev}: Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err: {err:4.2f} cnt")
avg_err = np.mean(eval_errs)
print(f"Mean Eval Error: {avg_err:.2f} cnt (<= 3.0 cnt: {'PASS' if avg_err <= 3.0 else 'FAIL'})")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

