import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Testing 5 shots with du_base=[23.0, -16.0]...")
errs = []
for i in range(1, 6):
    ctrl.driver.center(settle_s=0.45, preload=True)
    # du_base in ballistic_controller is [27.5, -15.2]
    # so we add [23.0 - 27.5, -16.0 - (-15.2)] = [-4.5, -0.8]
    ctrl.ilc.params["target_table"]["30_20"] = [-4.5, -0.8]
    ctrl.ilc.save()
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    err = res['final_error_counts']
    errs.append(err)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} cnt | Onset: {res['t_onset_ms']:4.1f}ms | Brake: {res['t_brake_ms']:4.1f}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
print(f"Mean error: {np.mean(errs):.2f} cnt")

