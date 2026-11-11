from ballistic_controller import Ballistic100msController
import numpy as np

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("--- Testing Target (-30, -25) ---")
# Current du_base is [-11.0, 8.0]
for du_x in [-11.0, -13.0, -15.0]:
    for du_y in [8.0, 10.0, 12.0]:
        ctrl.driver.center(settle_s=0.45, preload=True)
        ctrl.ilc.params["target_table"]["-30_-25"] = [du_x - (-11.0), du_y - 8.0]
        ctrl.ilc.save()
        res = ctrl.flick_shot(-30, -25, timeout_ms=100.0, enable_ilc_update=False)
        print(f"du_x={du_x:.1f}, du_y={du_y:.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

