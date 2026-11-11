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

    # Test pure vertical line y from -150 to +150
    # with coupling k_xy = -0.36
    y_targets = np.linspace(-150.0, 150.0, 11)
    
    print("\nTracing Pure Vertical Line x=0, y in [-150, 150]:")
    for y in y_targets:
        ux = ux0 - 0.36 * y
        uy = uy0 - 0.38 * y
        trainer.driver.send_raw_pulses(ux, uy)
        time.sleep(0.12)
        pos = trainer.get_rect_position()
        print(f"  Target (  0.0, {y:+6.1f}) -> Act ({pos[0]:+6.1f}, {pos[1]:+6.1f}) | x_tilt={pos[0]:+5.1f} cnt | uy={uy:.1f}, ux={ux:.1f}")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

