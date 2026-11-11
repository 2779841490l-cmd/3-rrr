import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

# Test direct nominal shots to all 6 targets with the refined K
# [dx, dy]
targets = [(35, 0), (-35, 0), (0, 35), (0, -35), (30, 25), (-30, -25)]

print("Testing direct shots on all 6 directions...")
for dx, dy in targets:
    ctrl.driver.center(settle_s=0.40, preload=True)
    res = ctrl.flick_shot(dx, dy, timeout_ms=100.0, enable_ilc_update=False)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    print(f"Target ({dx:+3d}, {dy:+3d}) -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err {err:4.1f} cnt")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

