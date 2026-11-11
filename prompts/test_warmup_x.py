import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params['target_table'] = {}
ctrl.sensor.start()

print("Testing 6 warmup + 2 eval shots on (+35, 0) and (-35, 0)...")
for target_x in [35, -35]:
    for w in range(6):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(target_x, 0, timeout_ms=100.0, enable_ilc_update=True)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        corr = ctrl.ilc.get_correction(target_x, 0)
        print(f"  Warmup {w+1} ({target_x:+3d}, 0) -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err {err:4.1f} cnt | ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

    eval_errs = []
    for ev in range(2):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(target_x, 0, timeout_ms=100.0, enable_ilc_update=False)
        fd = res["final_delta"]
        err = res["final_error_counts"]
        eval_errs.append(err)
        print(f"Eval {ev+1} ({target_x:+3d}, 0) -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err {err:4.1f} cnt")
    print(f"==> Target ({target_x:+3d}, 0) Eval Mean: {sum(eval_errs)/len(eval_errs):.2f} cnt\n")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

