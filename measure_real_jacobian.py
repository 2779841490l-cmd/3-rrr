import time
import numpy as np
from rect_manifold_calibrator import HardwareRectangleTrainer

trainer = HardwareRectangleTrainer()
trainer.sensor.start()
try:
    print("Zeroing at origin...")
    trainer.move_to_rect_origin(settle_s=0.6)
    ux0, uy0 = trainer.manifold.u0_star_x, trainer.manifold.u0_star_y
    print(f"Base pulse: ux0={ux0}, uy0={uy0}")
    
    # 1. Step ux by +25 us
    trainer.driver.send_raw_pulses(ux0 + 25.0, uy0)
    time.sleep(0.3)
    p_x_plus = trainer.get_rect_position()
    
    # Return
    trainer.driver.send_raw_pulses(ux0, uy0)
    time.sleep(0.3)
    
    # 2. Step ux by -25 us
    trainer.driver.send_raw_pulses(ux0 - 25.0, uy0)
    time.sleep(0.3)
    p_x_minus = trainer.get_rect_position()
    
    # Return
    trainer.driver.send_raw_pulses(ux0, uy0)
    time.sleep(0.3)
    
    # 3. Step uy by +25 us
    trainer.driver.send_raw_pulses(ux0, uy0 + 25.0)
    time.sleep(0.3)
    p_y_plus = trainer.get_rect_position()
    
    # Return
    trainer.driver.send_raw_pulses(ux0, uy0)
    time.sleep(0.3)
    
    # 4. Step uy by -25 us
    trainer.driver.send_raw_pulses(ux0, uy0 - 25.0)
    time.sleep(0.3)
    p_y_minus = trainer.get_rect_position()
    
    print(f"ux +25us -> dx={p_x_plus[0]:.1f}, dy={p_x_plus[1]:.1f}")
    print(f"ux -25us -> dx={p_x_minus[0]:.1f}, dy={p_x_minus[1]:.1f}")
    print(f"uy +25us -> dx={p_y_plus[0]:.1f}, dy={p_y_plus[1]:.1f}")
    print(f"uy -25us -> dx={p_y_minus[0]:.1f}, dy={p_y_minus[1]:.1f}")
    
    # Empirical J
    dx_dux = (p_x_plus[0] - p_x_minus[0]) / 50.0
    dy_dux = (p_x_plus[1] - p_x_minus[1]) / 50.0
    dx_duy = (p_x_plus[0] - p_y_minus[0]) / 50.0
    dy_duy = (p_y_plus[1] - p_y_minus[1]) / 50.0
    J_emp = np.array([[dx_dux, dx_duy], [dy_dux, dy_duy]])
    print("\nEmpirical Forward Jacobian J (counts/us):\n", J_emp)
    print("Empirical Inverse Jacobian K = J^-1 (us/count):\n", np.linalg.inv(J_emp))
    
finally:
    trainer.move_to_rect_origin(0.2)
    trainer.close()

