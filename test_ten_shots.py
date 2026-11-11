import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Testing 10 shots on (30, 20) with ux=30.0, uy=-17.2...")
errs = []
for i in range(1, 11):
    ctrl.driver.center(settle_s=0.45, preload=True)
    ctrl.ilc.params["target_table"]["30_20"] = [30.0 - 22.0, -17.2 - (-16.2)]
    ctrl.ilc.save()
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    err = res['final_error_counts']
    errs.append(err)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} cnt")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
print(f"\nOverall Mean: {np.mean(errs):.2f} cnt")
print(f"Last-5 Mean : {np.mean(errs[-5:]):.2f} cnt")
print(f"Last-5 Max  : {np.max(errs[-5:]):.2f} cnt")

