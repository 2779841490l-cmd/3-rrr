import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.ilc.save()
ctrl.sensor.start()

print("Testing with boost_ratio = 1.0 and stiction_kick = 0 on (30, 20)...")
# Let's test flick_shot with u_boost = u_lock!
# When u_boost = u_lock, there is zero overdrive overshoot!
# The command is cleanly sent to u_lock and held for 100ms!
# Let's see how repeatable the trajectory is!

errs = []
for i in range(1, 11):
    ctrl.driver.center(settle_s=0.45, preload=True)
    # We test with du_base = [23.0, -16.0]
    # In ballistic_controller, let's see what happens when flick_shot runs:
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    err = res['final_error_counts']
    errs.append(err)
    print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} cnt | Onset: {res['t_onset_ms']}ms | Brake: {res['t_brake_ms']}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
print(f"Mean error: {np.mean(errs):.2f} cnt")

