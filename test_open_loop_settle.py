import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer

trainer = HardwareRectangleTrainer()
trainer.sensor.start()
try:
    print("Moving to origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    p0 = trainer.get_rect_position()
    print(f"Origin position: {p0}")

    test_targets = [
        (0.0, 0.0),
        (100.0, 0.0),
        (0.0, 100.0),
        (-100.0, 0.0),
        (0.0, -100.0),
        (150.0, 120.0),
        (-150.0, -120.0),
    ]

    for tx, ty in test_targets:
        ux, uy = trainer.manifold.evaluate_pulse(tx, ty)
        trainer.driver.send_raw_pulses(ux, uy)
        time.sleep(0.35)  # wait for servo to fully settle
        p = trainer.get_rect_position()
        err_x = tx - p[0]
        err_y = ty - p[1]
        dist = np.hypot(err_x, err_y)
        print(f"Target ({tx:>6.1f}, {ty:>6.1f}) -> Act ({p[0]:>6.1f}, {p[1]:>6.1f}) | Err: ({err_x:>5.1f}, {err_y:>5.1f}), Norm: {dist:5.1f} cnt | Pulse ({ux:.1f}, {uy:.1f})")

finally:
    trainer.move_to_rect_origin(settle_s=0.3)
    trainer.close()

