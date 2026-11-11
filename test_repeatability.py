import time
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Testing mechanical repeatability of center(preload=True):")
p_start = ctrl.sensor.get_position()
for i in range(5):
    ctrl.driver.center(settle_s=0.35, preload=True)
    ctrl.sensor.read_delta_and_clear()
    time.sleep(0.1)
    p_cur = ctrl.sensor.get_position()
    # Now simulate a shot displacement
    ctrl.driver.send_raw_pulses(ctrl.driver.NEUTRAL_S1 + 25, ctrl.driver.NEUTRAL_S2 - 25)
    time.sleep(0.1)
    ctrl.driver.center(settle_s=0.35, preload=True)
    time.sleep(0.05)
    dx, dy = ctrl.sensor.read_delta_and_clear()
    print(f"Cycle {i+1}: Net shift after return to center = ({dx:+2d}, {dy:+2d})")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

