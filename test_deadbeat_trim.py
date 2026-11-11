import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

trainer = HardwareRectangleTrainer()
trainer.sensor.start()
try:
    print("Moving to origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    p0 = trainer.get_rect_position()
    print(f"Origin position: {p0}")

    test_targets = [
        (100.0, 0.0),
        (0.0, 100.0),
        (-100.0, 0.0),
        (0.0, -100.0),
        (150.0, 120.0),
        (-150.0, -120.0),
        (200.0, 150.0),
        (-200.0, -150.0),
        (0.0, 0.0),
    ]

    for tx, ty in test_targets:
        t_start = time.perf_counter()
        p_cur = trainer.get_rect_position()
        dx = tx - p_cur[0]
        dy = ty - p_cur[1]
        
        # 1. Primary command
        ux, uy = trainer.manifold.evaluate_pulse(tx, ty, dir_hint=(dx, dy))
        trainer.driver.send_raw_pulses(ux, uy)
        
        # Wait for primary stroke to settle using zero-velocity detection
        # MG996R takes ~150ms for 200 counts
        time.sleep(0.14)
        last_pos = trainer.get_rect_position()
        for _ in range(8):
            time.sleep(0.015)
            curr_pos = trainer.get_rect_position()
            v_norm = np.hypot(curr_pos[0] - last_pos[0], curr_pos[1] - last_pos[1])
            last_pos = curr_pos
            if v_norm < 1.5:
                break
        
        p_primary = trainer.get_rect_position()
        e_primary = np.array([tx - p_primary[0], ty - p_primary[1]], dtype=float)
        
        # 2. Deadbeat trim rounds (up to 2 rounds)
        cur_ux, cur_uy = ux, uy
        for trim_idx in range(2):
            p_meas = trainer.get_rect_position()
            err_vec = np.array([tx - p_meas[0], ty - p_meas[1]], dtype=float)
            if np.hypot(err_vec[0], err_vec[1]) <= 2.0:
                break
            
            # Local inverse jacobian
            gx = tx + float(trainer.manifold.origin_offset_xy[0])
            gy = ty + float(trainer.manifold.origin_offset_xy[1])
            K_loc = trainer.manifold.ws.local_inverse_jacobian(gx, gy)
            du = K_loc @ err_vec
            
            # Stiction kick if small
            kick_x = np.sign(du[0]) * 3.5 if abs(err_vec[0]) > 2.0 and abs(du[0]) < 3.5 else 0.0
            kick_y = np.sign(du[1]) * 2.0 if abs(err_vec[1]) > 2.0 and abs(du[1]) < 2.0 else 0.0
            
            cur_ux = float(np.clip(cur_ux + du[0] + kick_x, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            cur_uy = float(np.clip(cur_uy + du[1] + kick_y, WS63ServoDriver.SAFE_MIN_US, WS63ServoDriver.SAFE_MAX_US))
            trainer.driver.send_raw_pulses(cur_ux, cur_uy)
            time.sleep(0.045)
            
        p_final = trainer.get_rect_position()
        e_final = np.array([tx - p_final[0], ty - p_final[1]], dtype=float)
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        
        print(f"Target ({tx:>6.1f}, {ty:>6.1f}) | Primary Err: {np.hypot(e_primary[0], e_primary[1]):4.1f} cnt -> Final Err: {np.hypot(e_final[0], e_final[1]):4.2f} cnt ({np.hypot(e_final[0], e_final[1])*7.9375:4.1f} um) | FinalPos: ({p_final[0]:.1f}, {p_final[1]:.1f}) | {total_time_ms:5.1f} ms")

finally:
    trainer.move_to_rect_origin(settle_s=0.3)
    trainer.close()

