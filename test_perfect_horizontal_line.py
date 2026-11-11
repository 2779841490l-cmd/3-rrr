import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer

trainer = HardwareRectangleTrainer()
trainer.sensor.start()

try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    ux0 = trainer.manifold.u0_star_x
    uy0 = trainer.manifold.u0_star_y

    x_targets = np.linspace(-150.0, 150.0, 11)
    
    print("\nTracing Pure Horizontal Line y=0, x in [-150, 150]:")
    for x in x_targets:
        ux = ux0 + 0.65 * x
        uy = uy0 - 0.05 * x
        trainer.driver.send_raw_pulses(ux, uy)
        time.sleep(0.12)
        pos = trainer.get_rect_position()
        print(f"  Target ({x:+6.1f},   0.0) -> Act ({pos[0]:+6.1f}, {pos[1]:+6.1f}) | y_jump={pos[1]:+5.1f} cnt | uy={uy:.1f}, ux={ux:.1f}")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

