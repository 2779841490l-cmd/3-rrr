import time
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("Monitoring sensor position after center(settle_s=0.35, preload=True):")
for trial in range(3):
    ctrl.driver.center(settle_s=0.35, preload=True)
    t0 = time.perf_counter()
    p0 = ctrl.sensor.get_position()
    readings = []
    for _ in range(20):
        time.sleep(0.025)
        p = ctrl.sensor.get_position()
        elapsed = (time.perf_counter() - t0) * 1000.0
        readings.append((elapsed, p[0]-p0[0], p[1]-p0[1]))
    print(f"Trial {trial+1} drift over 500ms after center:")
    for el, dx, dy in readings[::4]:
        print(f"   t={el:5.1f}ms: delta=({dx:+2d}, {dy:+2d})")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

