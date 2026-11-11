import time
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.ilc.params['target_table'] = {}
ctrl.sensor.start()

print("=" * 72)
print("  TESTING PROGRESS THRESHOLD = 0.75 FOR (30, 20)")
print("=" * 72)

for i in range(1, 10):
    ctrl.driver.center(settle_s=0.35, preload=True)
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    fd = res["final_delta"]
    err = res["final_error_counts"]
    ton = res["t_onset_ms"]
    tb = res["t_brake_ms"]
    print(f"Shot {i}: final={fd}, err={err:4.2f} cnt, ton={ton:4.1f}ms, tb={tb:4.1f}ms, boost_dur={tb-ton:4.1f}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()
