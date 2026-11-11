import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing du_base[0]=30.0, du_base[1]=-17.0 on (30, 20)...")
for ux in [28.0, 30.0, 32.0]:
    for uy in [-16.5, -17.5]:
        errs = []
        for i in range(2):
            ctrl.driver.center(settle_s=0.45, preload=True)
            # Currently du_base in ballistic_controller is [22.0, -16.2]
            ctrl.ilc.params["target_table"]["30_20"] = [ux - 22.0, uy - (-16.2)]
            ctrl.ilc.save()
            res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
            errs.append(res['final_error_counts'])
            print(f"ux={ux:.1f}, uy={uy:.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")
        print(f"--> ux={ux:.1f}, uy={uy:.1f} Mean: {np.mean(errs):.2f}\n")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

