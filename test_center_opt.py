import time
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

def test_centering_repeatability(name, center_fn):
    print(f"\n--- Testing Centering Strategy: {name} ---")
    shifts = []
    for i in range(5):
        # Displace platform
        ctrl.driver.send_raw_pulses(ctrl.driver.NEUTRAL_S1 + 30, ctrl.driver.NEUTRAL_S2 - 25)
        time.sleep(0.12)
        # Apply centering
        center_fn()
        time.sleep(0.05)
        dx, dy = ctrl.sensor.read_delta_and_clear()
        shifts.append((dx, dy))
        print(f"  Cycle {i+1}: net shift = ({dx:+2d}, {dy:+2d})")
    xs = [s[0] for s in shifts[1:]]
    ys = [s[1] for s in shifts[1:]]
    print(f"  -> Std dev: X={np.std(xs):.2f}, Y={np.std(ys):.2f} counts")

import numpy as np

# Strategy A: Current preload (-45us step)
def strat_a():
    ctrl.driver.center(settle_s=0.35, preload=True)

# Strategy B: Smooth 2-stage approach: neutral - 22us (0.2s), then neutral (0.35s)
def strat_b():
    d = ctrl.driver
    d.send_raw_pulses(d.NEUTRAL_S1 - 20.0, d.NEUTRAL_S2 - 20.0)
    time.sleep(0.20)
    d.send_raw_pulses(d.NEUTRAL_S1, d.NEUTRAL_S2)
    time.sleep(0.35)

# Strategy C: Triple nudge approach: step to neutral-15, then neutral, with 0.4s settle
def strat_c():
    d = ctrl.driver
    d.send_raw_pulses(d.NEUTRAL_S1, d.NEUTRAL_S2)
    time.sleep(0.10)
    d.send_raw_pulses(d.NEUTRAL_S1 - 16.0, d.NEUTRAL_S2 - 16.0)
    time.sleep(0.20)
    d.send_raw_pulses(d.NEUTRAL_S1, d.NEUTRAL_S2)
    time.sleep(0.38)

test_centering_repeatability("Current Preload (-45us)", strat_a)
test_centering_repeatability("Gentle Preload (-20us)", strat_b)
test_centering_repeatability("Triple Nudge (-16us)", strat_c)

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

