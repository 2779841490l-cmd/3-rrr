import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Testing ux=5.0, uy=-11.5 on (30, 20)...")
errs = []
for i in range(1, 9):
    ctrl.driver.center(settle_s=0.40, preload=True)
    # ballistic_controller currently has [27.5, -15.2]
    ctrl.ilc.params["target_table"]["30_20"] = [5.0 - 27.5, -11.5 - (-15.2)]
    ctrl.ilc.save()
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    err = res['final_error_counts']
    errs.append(err)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} cnt | Onset: {res['t_onset_ms']}ms | Brake: {res['t_brake_ms']}ms")

ctrl.driver.center(settle_s=0.2, preload=True)
ctrl.close()
print(f"\nMean: {np.mean(errs):.2f} cnt, Last-5 Mean: {np.mean(errs[-5:]):.2f} cnt, Last-5 Max: {np.max(errs[-5:]):.2f} cnt")
