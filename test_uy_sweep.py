import time
from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

for uy in [-15.0, -16.5, -18.0, -19.5]:
    ctrl.driver.center(settle_s=0.45, preload=True)
    # Temporarily override du_base in flick_shot by monkeypatching or testing
    # Let's test by setting an ILC correction!
    # Remember: du_nom = du_base + du_ilc
    # In ballistic_controller: du_base for (30, 20) is [24.8, -15.0]
    # So if we set du_ilc to [ux - 24.8, uy - (-15.0)], it effectively sets du_nom!
    ctrl.ilc.params["target_table"]["30_20"] = [-1.0, uy - (-15.0)]
    ctrl.ilc.save()
    res = ctrl.flick_shot(30, 20, timeout_ms=100.0, enable_ilc_update=False)
    print(f"uy={uy:.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f} cnt, Brake: {res['t_brake_ms']}ms")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

