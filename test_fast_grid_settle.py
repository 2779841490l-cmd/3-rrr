import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    
    # Decoupled Empirical Jacobian
    # [du_x] = [ 1.10   -0.09 ] [dx]
    # [du_y]   [-0.05   -0.35 ] [dy]
    K_emp = np.array([
        [1.10, -0.09],
        [-0.05, -0.35]
    ], dtype=float)

    def goto(tx, ty):
        t0 = time.perf_counter()
        p0 = trainer.get_rect_position()
        dx = tx - p0[0]
        dy = ty - p0[1]
        
        ux, uy = trainer.manifold.evaluate_pulse(tx, ty, dir_hint=(dx, dy))
        trainer.driver.send_raw_pulses(ux, uy)
        
        # Settle wait
        time.sleep(0.13)
        p_last = trainer.get_rect_position()
        for _ in range(8):
            time.sleep(0.015)
            p_now = trainer.get_rect_position()
            if np.hypot(p_now[0] - p_last[0], p_now[1] - p_last[1]) <= 1.5:
                break
            p_last = p_now
            
        cur_ux, cur_uy = ux, uy
        for _ in range(2):
            p_m = trainer.get_rect_position()
            e = np.array([tx - p_m[0], ty - p_m[1]], dtype=float)
            if np.hypot(e[0], e[1]) <= 2.0:
                break
            du = K_emp @ e
            # Only kick if error is significant and du is small (stiction)
            kx = np.sign(du[0]) * 3.0 if abs(e[0]) > 2.0 and abs(du[0]) < 3.0 else 0.0
            ky = np.sign(du[1]) * 2.0 if abs(e[1]) > 2.0 and abs(du[1]) < 2.0 else 0.0
            cur_ux = float(np.clip(cur_ux + du[0] + kx, 1150, 1850))
            cur_uy = float(np.clip(cur_uy + du[1] + ky, 1150, 1850))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.055)
            
        pf = trainer.get_rect_position()
        ef = float(np.hypot(tx - pf[0], ty - pf[1]))
        ms = (time.perf_counter() - t0) * 1000.0
        return ef, ms, pf

    test_pts = [
        (100.0, 80.0),
        (-100.0, 80.0),
        (-100.0, -80.0),
        (100.0, -80.0),
        (0.0, 0.0),
        (180.0, 0.0),
        (-180.0, 0.0),
        (0.0, 120.0),
        (0.0, -120.0),
    ]

    errs = []
    times = []
    print("\n--- Testing 9 Grid Points Precision Positioning ---")
    for tx, ty in test_pts:
        ef, ms, pf = goto(tx, ty)
        errs.append(ef)
        times.append(ms)
        print(f"  Target ({tx:+6.1f}, {ty:+6.1f}) -> Final ({pf[0]:+6.1f}, {pf[1]:+6.1f}) | Err: {ef:4.2f} cnt ({ef*7.9375:4.1f} um) | {ms:5.1f} ms")

    print(f"\n[Result] Mean Error: {np.mean(errs):.2f} cnt ({np.mean(errs)*7.9375:.1f} um) | Max Error: {np.max(errs):.2f} cnt | Mean Time: {np.mean(times):.1f} ms")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

