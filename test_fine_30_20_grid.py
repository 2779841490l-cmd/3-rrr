from ballistic_controller import Ballistic100msController
import numpy as np

ctrl = Ballistic100msController()
ctrl.sensor.start()

for delta_ux, delta_uy in [(-2.8, -1.0), (-2.8, -1.4), (-3.2, -1.2), (-3.2, -1.6)]:
    errs = []
    for i in range(3):
        ctrl.driver.center(settle_s=0.45, preload=True)
        ctrl.ilc.params["target_table"]["30_20"] = [delta_ux, delta_uy]
        ctrl.ilc.save()
        res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
        errs.append(res['final_error_counts'])
        print(f"du=({delta_ux:.1f},{delta_uy:.1f}) Shot {i+1} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")
    print(f"--> du=({delta_ux:.1f},{delta_uy:.1f}) Mean Err: {np.mean(errs):.2f}\n")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

