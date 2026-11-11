import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()

ctrl.sensor.start()

directions = [(35, 0), (-35, 0), (0, 35), (0, -35), (30, 25), (-30, -25)]
dir_eval_errors = []
print("Testing Gate 2 (6 directions: 5 warmup + 2 eval)...")

for dx, dy in directions:
    # 5 warmup
    for w in range(1, 6):
        ctrl.driver.center(settle_s=0.45, preload=True)
        ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=True)
    
    # 2 eval
    eval_errs = []
    eval_finals = []
    for ev in range(1, 3):
        ctrl.driver.center(settle_s=0.45, preload=True)
        res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
        eval_errs.append(res["final_error_counts"])
        eval_finals.append(res["final_delta"])
    
    avg_dir_err = float(np.mean(eval_errs))
    dir_eval_errors.append(avg_dir_err)
    corr = ctrl.ilc.get_correction(dx, dy)
    status_str = "PASS" if avg_dir_err <= 3.0 else "FAIL"
    print(
        f"  [{status_str}] Target ({dx:+3d}, {dy:+3d}) | Finals: {eval_finals[0]}, {eval_finals[1]} | "
        f"Mean Err: {avg_dir_err:4.2f} cnt ({avg_dir_err*7.9375:5.1f} um) | ILC: [{corr[0]:+5.2f}, {corr[1]:+5.2f}]"
    )

overall_mean = float(np.mean(dir_eval_errors))
print(f"\nGate 2 Overall Mean: {overall_mean:.2f} cnt (limit <= 2.00)")
if overall_mean <= 2.00 and all(e <= 3.00 for e in dir_eval_errors):
    print(">>> GATE 2 PASSED! <<<")
else:
    print(">>> GATE 2 FAILED! <<<")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

