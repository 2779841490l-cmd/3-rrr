import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params["target_table"] = {}
ctrl.sensor.start()

print("Testing 6 ILC shots on (-35, 0)...")
for i in range(6):
    ctrl.driver.center(settle_s=0.38, preload=True)
    res = ctrl.flick_shot(-35, 0, timeout_ms=100.0, enable_ilc_update=True)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    corr = ctrl.ilc.get_correction(-35, 0)
    print(f"Shot {i+1}: Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err {err:4.1f} cnt | ILC_u: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
