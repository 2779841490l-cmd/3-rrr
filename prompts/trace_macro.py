import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params['target_table'] = {}
ctrl.sensor.start()

print("Testing 6 ILC warmup + 2 eval shots on (160, -90)...")
for i in range(6):
    ctrl.driver.center(settle_s=0.40, preload=True)
    res = ctrl.flick_shot(160, -90, timeout_ms=100.0, enable_ilc_update=True)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    corr = ctrl.ilc.get_correction(160, -90)
    print(f"Warmup {i+1}: Final ({fd[0]:+4d}, {fd[1]:+4d}) | Err {err:4.1f} cnt | ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

for ev in range(2):
    ctrl.driver.center(settle_s=0.40, preload=True)
    res = ctrl.flick_shot(160, -90, timeout_ms=100.0, enable_ilc_update=False)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    tb = res["t_brake_ms"]
    to = res["t_onset_ms"]
    print(f"Eval {ev+1}: Final ({fd[0]:+4d}, {fd[1]:+4d}) | Err {err:4.1f} cnt | Brake: {tb:4.1f}ms | Onset: {to:4.1f}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
