import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Testing 8 shots on (30, 20) with du_base=[28.0, -14.4]...")
errs = []
for i in range(1, 9):
    ctrl.driver.center(settle_s=0.45, preload=True)
    # du_base in ballistic_controller is currently [23.5, -15.5]
    # so we add [28.0 - 23.5, -14.4 - (-15.5)] = [+4.5, +1.1]
    ctrl.ilc.params["target_table"]["30_20"] = [4.5, 1.1]
    ctrl.ilc.save()
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    err = res['final_error_counts']
    errs.append(err)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} cnt | Onset: {res['t_onset_ms']:4.1f}ms | Brake: {res['t_brake_ms']:4.1f}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
print(f"\nMean: {np.mean(errs):.2f} cnt, Max: {np.max(errs):.2f} cnt")
