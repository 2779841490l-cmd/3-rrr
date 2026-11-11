import json
import time
import numpy as np
from scipy.interpolate import RBFInterpolator
from rect_manifold_calibrator import HardwareRectangleTrainer
from servo_driver import WS63ServoDriver

# 1. Load real hardware 49 samples
with open("experiments/real_rect_grid_samples.json", "r") as f:
    samples = json.load(f)

pts_xy = np.array([[s["x"], s["y"]] for s in samples], dtype=float)
pts_u = np.array([[s["ux"], s["uy"]] for s in samples], dtype=float)

# Fit RBF Thin-Plate Spline (x, y) -> (ux, uy)
rbf_direct = RBFInterpolator(pts_xy, pts_u, kernel="thin_plate_spline", smoothing=0.1)

print("Direct Real-Hardware RBF fitted on 49 samples.")

# Test on hardware:
trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    
    test_targets = [
        (0.0, 0.0),
        (80.0, 60.0),
        (-80.0, 60.0),
        (-80.0, -60.0),
        (80.0, -60.0),
        (100.0, 0.0),
        (-100.0, 0.0),
        (0.0, 80.0),
        (0.0, -80.0),
        (0.0, 0.0),
    ]

    print("\n--- Testing Single-Shot Open-Loop Accuracy of Direct Real RBF ---")
    open_errs = []
    final_errs = []
    
    for tx, ty in test_targets:
        u_pred = rbf_direct(np.array([[tx, ty]], dtype=float))[0]
        ux = float(np.clip(u_pred[0], 1150.0, 1850.0))
        uy = float(np.clip(u_pred[1], 1150.0, 1850.0))
        
        # Send open loop
        trainer.driver.send_raw_pulses(ux, uy)
        time.sleep(0.20)
        p_open = trainer.get_rect_position()
        e_open = float(np.hypot(tx - p_open[0], ty - p_open[1]))
        open_errs.append(e_open)
        
        # Micro deadbeat trim (1 round)
        err_vec = np.array([tx - p_open[0], ty - p_open[1]], dtype=float)
        if np.hypot(err_vec[0], err_vec[1]) > 2.0:
            dux = 0.68 * err_vec[0]
            duy = -0.38 * err_vec[1]
            ux_trim = float(np.clip(ux + dux, 1150.0, 1850.0))
            uy_trim = float(np.clip(uy + duy, 1150.0, 1850.0))
            trainer.driver.send_raw_pulses(ux_trim, uy_trim)
            time.sleep(0.12)
            
        p_final = trainer.get_rect_position()
        e_final = float(np.hypot(tx - p_final[0], ty - p_final[1]))
        final_errs.append(e_final)
        
        print(f"Target ({tx:+6.1f}, {ty:+6.1f}) | OpenLoop: {e_open:4.1f} cnt -> Final: {e_final:4.2f} cnt ({e_final*7.9375:4.1f} um) | FinalPos: ({p_final[0]:+5.1f}, {p_final[1]:+5.1f})")

    print(f"\n[Summary] Mean Open-Loop Error: {np.mean(open_errs):.2f} cnt")
    print(f"[Summary] Mean Final Error:     {np.mean(final_errs):.2f} cnt ({np.mean(final_errs)*7.9375:.1f} um)")
    print(f"[Summary] Max Final Error:      {np.max(final_errs):.2f} cnt ({np.max(final_errs)*7.9375:.1f} um)")

finally:
    trainer.move_to_rect_origin(settle_s=0.2)
    trainer.close()

