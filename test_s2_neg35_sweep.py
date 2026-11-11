from ballistic_controller import Ballistic100msController

ctrl = Ballistic100msController()
ctrl.sensor.start()

print("--- Sweep S2 pulse during (-35, 0) shot ---")
# Let's test uy from -20 to +20 in steps of 5
for uy_val in [-25.0, -15.0, -5.0, 5.0, 15.0, 25.0]:
    ctrl.driver.center(settle_s=0.45, preload=True)
    # Target (-35, 0) in ballistic_controller has du_base[1] = -2.2
    # So to make du_base[1] equal to uy_val, we add uy_val - (-2.2)
    ctrl.ilc.params["target_table"]["-35_0"] = [0.0, uy_val - (-2.2)]
    ctrl.ilc.save()
    res = ctrl.flick_shot(-35, 0, timeout_ms=100.0, enable_ilc_update=False)
    print(f"uy={uy_val:+5.1f} -> Final {res['final_delta']}, Err: {res['final_error_counts']:.2f}")

ctrl.driver.center(settle_s=0.35, preload=True)
ctrl.close()

