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

    print("\nCommanding 50 us step in ux and uy...")
    t0 = time.perf_counter()
    trainer.driver.send_raw_pulses(ux0 + 50.0, uy0 + 50.0)
    
    trace = []
    while (time.perf_counter() - t0) < 0.400:
        now = time.perf_counter()
        p = trainer.get_rect_position()
        trace.append(((now - t0) * 1000.0, float(p[0]), float(p[1])))
        time.sleep(0.005)
        
    print(f"Recorded {len(trace)} samples over 400 ms.")
    # Print every 20ms
    for t_ms, x, y in trace[::4]:
        print(f"  t={t_ms:5.1f} ms | x={x:+6.1f}, y={y:+6.1f}")
        
    # Find onset time (when x moves >= 3 counts)
    onset_t = next((t for t, x, y in trace if np.hypot(x, y) >= 3.0), None)
    # Find 95% settle time (when x reaches 95% of final)
    x_final, y_final = trace[-1][1], trace[-1][2]
    dist_final = np.hypot(x_final, y_final)
    settle_t = next((t for t, x, y in trace if np.hypot(x, y) >= 0.95 * dist_final), None)
    
    print(f"\nOnset time: {onset_t:.1f} ms")
    print(f"95% Settle time: {settle_t:.1f} ms")
    print(f"Final reached: ({x_final:.1f}, {y_final:.1f}), Norm={dist_final:.1f} counts")

finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

