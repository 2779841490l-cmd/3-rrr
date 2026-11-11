import ctypes
import time
import numpy as np
from ballistic_controller import Ballistic100msController

# Temporarily disable Windows mouse acceleration (EPP)
SPI_SETMOUSE = 0x0004
SPI_GETMOUSE = 0x0003
orig_params = (ctypes.c_int * 3)()
ctypes.windll.user32.SystemParametersInfoW(SPI_GETMOUSE, 0, orig_params, 0)
zero_params = (ctypes.c_int * 3)(0, 0, 0)
ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, zero_params, 0)
print(f"Disabled mouse acceleration: original={list(orig_params)} -> zero={list(zero_params)}")

try:
    ctrl = Ballistic100msController()
    ctrl.sensor.start()
    
    print("\nTesting 10 shots on (30, 20) with 100% LINEAR mouse input (No EPP)...")
    errs = []
    for i in range(1, 11):
        ctrl.driver.center(settle_s=0.40, preload=True)
        res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
        err = res['final_error_counts']
        errs.append(err)
        print(f"Shot {i:2d} -> Final {res['final_delta']} | Err: {err:4.2f} cnt | Onset: {res['t_onset_ms']}ms | Brake: {res['t_brake_ms']}ms")
    
    print(f"\nOverall Mean: {np.mean(errs):.2f} cnt")
    print(f"Last-5 Mean : {np.mean(errs[-5:]):.2f} cnt")
    print(f"Last-5 Max  : {np.max(errs[-5:]):.2f} cnt")
    
    ctrl.driver.center(settle_s=0.2, preload=True)
    ctrl.close()

finally:
    # Always restore original settings
    ctypes.windll.user32.SystemParametersInfoW(SPI_SETMOUSE, 0, orig_params, 0)
    print("Restored original mouse acceleration settings.")

