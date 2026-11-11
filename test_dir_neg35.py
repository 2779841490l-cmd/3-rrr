from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("--- Testing Target (-35, 0) with positive uy ---")
# Currently in ballistic_controller:
# elif delta_x == -35 and delta_y == 0:
#     du_base[0] *= 1.06
#     du_base[1] = -2.2
# K @ [-35, 0] gives du_base[0] ~ -36.3
# So du_base[0] ~ -38.5, du_base[1] = -2.2
for delta_uy in [2.2, 4.2, 6.2, 8.2]:
    ctrl.driver.center(settle_s=0.45, preload=True)
    # With delta_uy, uy becomes -2.2 + delta_uy
    ctrl.ilc.params["target_table"]["-35_0"] = [0.0, delta_uy]
    ctrl.ilc.save()
    res = ctrl.flick_shot(-35, 0, timeout_ms=100.0, enable_ilc_update=False)
    print(f"uy={-2.2+delta_uy:.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

