import time
import json
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    ux0 = trainer.manifold.u0_star_x
    uy0 = trainer.manifold.u0_star_y
    print(f"Base origin pulse: ux0={ux0:.1f}, uy0={uy0:.1f}")

    # 6x6 grid in (ux, uy) space
    dux_vals = np.linspace(-45.0, 45.0, 7)
    duy_vals = np.linspace(-50.0, 50.0, 7)
    
    samples = []
    print(f"Sampling {len(dux_vals) * len(duy_vals)} grid points on real hardware...")
    
    for j, duy in enumerate(duy_vals):
        # serpentine to minimize transit
        row_dux = dux_vals if j % 2 == 0 else dux_vals[::-1]
        for dux in row_dux:
            cmd_ux = float(np.clip(ux0 + dux, 1150.0, 1850.0))
            cmd_uy = float(np.clip(uy0 + duy, 1150.0, 1850.0))
            trainer.driver.send_raw_pulses(cmd_ux, cmd_uy)
            time.sleep(0.18)
            pos = trainer.get_rect_position()
            samples.append({
                "dux": round(dux, 1),
                "duy": round(duy, 1),
                "ux": round(cmd_ux, 1),
                "uy": round(cmd_uy, 1),
                "x": round(float(pos[0]), 1),
                "y": round(float(pos[1]), 1),
            })
            print(f"  du=({dux:+5.1f}, {duy:+5.1f}) us -> pos=({pos[0]:+6.1f}, {pos[1]:+6.1f}) cnt")

    with open("experiments/real_rect_grid_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\nSaved {len(samples)} real hardware calibration points to experiments/real_rect_grid_samples.json")

finally:
    trainer.move_to_rect_origin(settle_s=0.3)
    trainer.close()

