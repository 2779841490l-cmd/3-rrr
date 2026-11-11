import sys
import time
import numpy as np
from ballistic_controller import Ballistic100msController

def test_full_system():
    print("=" * 74)
    print("  FAST INTEGRATION TEST: GATE 1 + GATE 2 + GATE 3")
    print("=" * 74)
    
    ctrl = Ballistic100msController()
    ctrl.ilc.params["target_table"] = {}
    ctrl.sensor.start()
    
    # 1. Gate 1: 12 shots on (+30, +20)
    print("\n--- Testing Gate 1 (12 Shots on +30, +20) ---")
    g1_errs = []
    for i in range(1, 13):
        ctrl.driver.center(settle_s=0.35, preload=True)
        res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=True)
        err = res["final_error_counts"]
        g1_errs.append(err)
        corr = ctrl.ilc.get_correction(30, 20)
        print(f"  Shot {i:2d}/12 | Final {res['final_delta']} | Err: {err:4.2f} cnt | ILC: [{corr[0]:+5.1f}, {corr[1]:+5.1f}]")
    
    l5 = g1_errs[-5:]
    mean_l5 = np.mean(l5)
    max_l5 = np.max(l5)
    print(f"Gate 1: Last-5 Mean = {mean_l5:.2f} (<=2.50?), Max = {max_l5:.2f} (<=4.00?)")
    
    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.close()

if __name__ == "__main__":
    test_full_system()

