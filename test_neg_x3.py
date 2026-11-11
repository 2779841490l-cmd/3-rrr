import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing Target (-35, 0) with b_off_x = -27.5, test_s2 = -5.0:")
for du_x in [-38.0, -42.0, -46.0]:
    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.sensor.read_delta_and_clear()
    x0, y0 = ctrl.sensor.get_position()
    
    b_off_x = -27.5
    stk_x = -4.5
    u_boost_x = ctrl.driver.cur_s1 + du_x * 1.16 + b_off_x + stk_x
    u_lock_x = ctrl.driver.cur_s1 + du_x + b_off_x
    
    du_y = -5.2
    b_off_y = -4.2
    stk_y = -3.2
    u_boost_y = ctrl.driver.cur_s2 + du_y * 1.16 + b_off_y + stk_y
    u_lock_y = ctrl.driver.cur_s2 + du_y + b_off_y
    
    ctrl.driver.send_raw_pulses(u_boost_x, u_boost_y)
    t_start = time.perf_counter()
    while (time.perf_counter() - t_start) < 0.038:
        time.sleep(0.001)
    ctrl.driver.send_raw_pulses(u_lock_x, u_lock_y)
    while (time.perf_counter() - t_start) < 0.100:
        time.sleep(0.001)
        
    x1, y1 = ctrl.sensor.get_position()
    print(f"du_x = {du_x:5.1f} -> Final: ({x1-x0:+3d}, {y1-y0:+3d}) | Err to (-35,0): {np.hypot(-35-(x1-x0), 0-(y1-y0)):.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

