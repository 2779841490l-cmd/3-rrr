from ballistic_controller import Ballistic100msController
import numpy as np

ctrl = Ballistic100msController()
ctrl.sensor.start()

errs = []
for i in range(3):
    ctrl.driver.center(settle_s=0.45, preload=True)
    ctrl.ilc.params["target_table"]["35_0"] = [36.0 - 36.0, -9.5 - (-7.8)]
    ctrl.ilc.save()
    res = ctrl.flick_shot(35, 0, timeout_ms=100.0, enable_ilc_update=False)
    errs.append(res['final_error_counts'])
    print(f"Shot {i+1} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
print(f"Mean error for (+35, 0): {np.mean(errs):.2f} cnt")
