import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Moving to origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    
    # Accurate empirical K:
    # du_x = 0.70 * dx - 0.28 * dy
    # du_y = -0.10 * dx - 0.38 * dy
    # with backlash kick
    K_trim = np.array([
        [0.72, -0.28],
        [-0.10, -0.38]
    ], dtype=float)

    def goto_point(tx, ty, tol=2.0, max_time_ms=350.0):
        t0 = time.perf_counter()
        p_cur = trainer.get_rect_position()
        dx = tx - p_cur[0]
        dy = ty - p_cur[1]
        
        # Primary stroke
        ux_cmd, uy_cmd = trainer.manifold.evaluate_pulse(tx, ty, dir_hint=(dx, dy))
        trainer.driver.send_raw_pulses(ux_cmd, uy_cmd)
        
        # Phase 1: Wait for primary stroke to settle using zero-velocity detection
        # Servo takes at least 100ms
        time.sleep(0.12)
        p_last = trainer.get_rect_position()
        for _ in range(10):
            time.sleep(0.015)
            p_now = trainer.get_rect_position()
            v_cnt = np.hypot(p_now[0] - p_last[0], p_now[1] - p_last[1])
            p_last = p_now
            if v_cnt <= 1.5:
                break
                
        p_after_primary = trainer.get_rect_position()
        e_primary = np.hypot(tx - p_after_primary[0], ty - p_after_primary[1])
        
        # Phase 2: Up to 3 deadbeat micro-trims
        for trim_iter in range(3):
            p_meas = trainer.get_rect_position()
            err_vec = np.array([tx - p_meas[0], ty - p_meas[1]], dtype=float)
            err_norm = float(np.hypot(err_vec[0], err_vec[1]))
            if err_norm <= tol:
                break
                
            du = K_trim @ err_vec
            
            # Backlash & stiction kick
            # If error is positive in x, horn must move +x; if reversing, kick
            kick_x = 0.0
            if abs(err_vec[0]) > 1.5:
                kick_x = 4.0 if du[0] > 0 else -4.0
            kick_y = 0.0
            if abs(err_vec[1]) > 1.5:
                kick_y = 2.0 if du[1] > 0 else -2.0
                
            ux_cmd = float(np.clip(ux_cmd + du[0] + kick_x, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            uy_cmd = float(np.clip(uy_cmd + du[1] + kick_y, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            trainer.driver.send_raw_pulses(ux_cmd, uy_cmd)
            
            # Wait for trim pulse to physically execute (60ms)
            time.sleep(0.060)
            
        p_final = trainer.get_rect_position()
        e_final = float(np.hypot(tx - p_final[0], ty - p_final[1]))
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "start": (round(float(p_cur[0]), 1), round(float(p_cur[1]), 1)),
            "target": (round(tx, 1), round(ty, 1)),
            "final": (round(float(p_final[0]), 1), round(float(p_final[1]), 1)),
            "primary_err": round(e_primary, 1),
            "final_err": round(e_final, 2),
            "final_err_um": round(e_final * 7.9375, 1),
            "elapsed_ms": round(elapsed, 1),
        }

    targets = [
        (150.0, 100.0),
        (-150.0, 100.0),
        (-150.0, -100.0),
        (150.0, -100.0),
        (0.0, 0.0),
        (200.0, 0.0),
        (-200.0, 0.0),
        (0.0, 150.0),
        (0.0, -150.0),
        (0.0, 0.0),
    ]

    print("\n--- Testing Point-to-Point Settling & Deadbeat Trims ---")
    errs = []
    for tx, ty in targets:
        r = goto_point(tx, ty)
        errs.append(r["final_err"])
        print(f"Target ({tx:+6.1f}, {ty:+6.1f}) | Primary Err: {r['primary_err']:4.1f} cnt -> Final Err: {r['final_err']:4.2f} cnt ({r['final_err_um']:4.1f} um) | FinalPos: {r['final']} | {r['elapsed_ms']:5.1f} ms")

    print(f"\nMean Final Error: {np.mean(errs):.2f} cnt ({np.mean(errs)*7.9375:.1f} um) | Max Error: {np.max(errs):.2f} cnt ({np.max(errs)*7.9375:.1f} um)")

finally:
    trainer.move_to_rect_origin(settle_s=0.3)
    trainer.close()

