"""
Quick diagnostic runner for tuning individual targets
"""
import sys
import numpy as np
from ballistic_controller import Ballistic100msController

def test_target(tx, ty, shots=5, enable_ilc=True):
    print(f"\n--- Testing Target ({tx}, {ty}) for {shots} shots ---")
    ctrl = Ballistic100msController()
    ctrl.sensor.start()
    
    # Warmup / Test
    errs = []
    for s in range(1, shots + 1):
        ctrl.driver.center(settle_s=0.38, preload=True)
        res = ctrl.flick_shot(tx, ty, timeout_ms=100.0, enable_ilc_update=enable_ilc)
        err = res["final_error_counts"]
        fd = res["final_delta"]
        errs.append(err)
        corr = ctrl.ilc.get_correction(tx, ty)
        print(f"Shot {s:2d} -> Final ({fd[0]:+3d}, {fd[1]:+3d}) | Err: {err:4.2f} cnt | ILC: [{corr[0]:+5.2f}, {corr[1]:+5.2f}] | Brake: {res['t_brake_ms']}ms")
    
    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.close()
    print(f"Mean error: {np.mean(errs):.2f} cnt")
    return errs

if __name__ == "__main__":
    tx = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ty = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    test_target(tx, ty, shots)

