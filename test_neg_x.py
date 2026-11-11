import time
import numpy as np
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing Target (-35, 0) with different S2 kinematic compensations:")
# What does the calibrated 2nd-order Jacobian give for (-35, 0)?
K = ctrl.poly_jac.evaluate(0, 0)
print(f"K = \n{K}")
nom_du = K @ np.array([-35.0, 0.0])
print(f"Nominal du = {nom_du}")

for test_s2 in [0.0, float(nom_du[1]), 12.0, 16.0]:
    ctrl.driver.center(settle_s=0.35, preload=True)
    # Let's fire shot with test_s2 applied to du_base[1]
    # We will test final delta
    ctrl.sensor.read_delta_and_clear()
    x0, y0 = ctrl.sensor.get_position()
    
    # Send custom pulse
    du_x = nom_du[0]
    b_off_x = max(-21.5, du_x * (21.5 / 15.0))
    u_boost_x = ctrl.driver.cur_s1 + du_x * 1.16 + b_off_x - 4.2
    u_lock_x = ctrl.driver.cur_s1 + du_x + b_off_x
    
    du_y = test_s2
    b_off_y = min(5.0, du_y * (5.0 / 10.0)) if du_y >= 0 else max(-4.2, du_y * (4.2 / 8.0))
    stk_y = 3.6 * np.sign(du_y) if abs(du_y) > 0.5 else 0.0
    u_boost_y = ctrl.driver.cur_s2 + du_y * 1.16 + b_off_y + stk_y
    u_lock_y = ctrl.driver.cur_s2 + du_y + b_off_y
    
    ctrl.driver.send_raw_pulses(u_boost_x, u_boost_y)
    t_start = time.perf_counter()
    while (time.perf_counter() - t_start) < 0.035:
        time.sleep(0.001)
    ctrl.driver.send_raw_pulses(u_lock_x, u_lock_y)
    while (time.perf_counter() - t_start) < 0.100:
        time.sleep(0.001)
        
    x1, y1 = ctrl.sensor.get_position()
    print(f"test_s2 = {test_s2:+5.1f} us -> Final: ({x1-x0:+3d}, {y1-y0:+3d})")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

