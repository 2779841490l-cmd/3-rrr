import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing Target (-35, 0) with NEGATIVE S2 pulses:")
K = ctrl.poly_jac.evaluate(0, 0)

for test_s2 in [-2.0, -5.0, -8.0, -10.0]:
    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.sensor.read_delta_and_clear()
    x0, y0 = ctrl.sensor.get_position()
    
    du_x = -36.0  # boost nominal X slightly so it reaches -35
    b_off_x = -22.0
    u_boost_x = ctrl.driver.cur_s1 + du_x * 1.16 + b_off_x - 4.5
    u_lock_x = ctrl.driver.cur_s1 + du_x + b_off_x
    
    du_y = test_s2
    b_off_y = min(5.0, du_y * (5.0 / 10.0)) if du_y >= 0 else max(-4.2, du_y * (4.2 / 8.0))
    stk_y = 3.6 * np.sign(du_y) if abs(du_y) > 0.5 else 0.0
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
    print(f"test_s2 = {test_s2:+5.1f} us -> Final: ({x1-x0:+3d}, {y1-y0:+3d})")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

