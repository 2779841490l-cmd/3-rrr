import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()

ctrl.sensor.start()
ctrl.driver.center(settle_s=0.45, preload=True)

tx, ty = 30, 20
gate1_errors = []
print("Testing Gate 1 (12 shots)...")
for i in range(1, 13):
    ctrl.driver.center(settle_s=0.45, preload=True)
    res = ctrl.flick_shot(tx, ty, timeout_ms=100.0, enable_ilc_update=True)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    gate1_errors.append(err)
    corr = ctrl.ilc.get_correction(tx, ty)
    print(f"Shot {i:2d}/12 -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err: {err:4.2f} cnt | ILC: [{corr[0]:+5.2f}, {corr[1]:+5.2f}]")

last5 = gate1_errors[-5:]
print(f"Last-5 Mean: {np.mean(last5):.2f} cnt, Max: {np.max(last5):.2f} cnt")
ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

