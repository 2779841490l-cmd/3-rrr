import time
import numpy as np
from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver

drv = WS63ServoDriver()
sensor = HighPrecisionMouseSensor()
sensor.start()

print("Testing theoretical matched ballistic shot on (30, 20)...")
# Target (30, 20)
# u_boost = (1532.0, 1520.0)
# u_lock  = (1527.5, 1524.5)

errs = []
tau_brake = 0.020

for i in range(1, 11):
    drv.center(settle_s=0.40, preload=True)
    sensor.read_delta_and_clear()
    x0, y0 = sensor.get_position()
    
    t_start = time.perf_counter()
    drv.send_raw_pulses(1526.5, 1522.2)
    
    t_onset = None
    brake_sent = False
    t_brake = 0.0
    brake_reason = "NONE"
    vel_buf = []
    
    while True:
        t_now = time.perf_counter()
        el = (t_now - t_start) * 1000.0
        cur_x, cur_y = sensor.get_position()
        if t_onset is None and el >= 25.0 and (cur_x != x0 or cur_y != y0):
            t_onset = t_now
        act = (t_now - t_onset) * 1000.0 if t_onset is not None else 0.0
        
        vel_buf.append((t_now, cur_x, cur_y))
        while vel_buf and (t_now - vel_buf[0][0]) > 0.016:
            vel_buf.pop(0)
        dt_b = t_now - vel_buf[0][0]
        vx = (cur_x - vel_buf[0][1]) / dt_b if dt_b > 0.001 else 0.0
        vy = (cur_y - vel_buf[0][2]) / dt_b if dt_b > 0.001 else 0.0
        
        pred_x = cur_x + vx * tau_brake
        pred_y = cur_y + vy * tau_brake
        
        prog = (pred_x - x0) * 30 + (pred_y - y0) * 20
        if not brake_sent and act >= 12.0:
            trigger_reason = None
            if (prog >= 0.78 * (900 + 400)) or (act >= 44.0):
                trigger_reason = "PRED_OPTICAL"
            elif el >= 95.0:
                trigger_reason = "EL_LIMIT"
                
            if trigger_reason:
                drv.send_raw_pulses(1522.5, 1526.2)
                brake_sent = True
                t_brake = el
                brake_reason = trigger_reason
        
        if (t_onset is not None and act >= 100.0) or el >= 115.0:
            if not brake_sent:
                drv.send_raw_pulses(1522.5, 1526.2)
            break
        time.sleep(0.001)
        
    fx, fy = sensor.get_position()
    dx, dy = fx - x0, fy - y0
    err = float(np.hypot(30 - dx, 20 - dy))
    errs.append(err)
    print(f"Shot {i:2d} -> Final ({dx:+2d}, {dy:+2d}) | Err: {err:4.2f} cnt | Onset: {(t_onset-t_start)*1000 if t_onset else -1:.1f}ms | Brake: {t_brake:.1f}ms ({brake_reason})")

drv.center(settle_s=0.2, preload=True)
sensor.stop()
drv.close()

print(f"\nMean: {np.mean(errs):.2f} cnt, Last-5 Mean: {np.mean(errs[-5:]):.2f} cnt, Last-5 Max: {np.max(errs[-5:]):.2f} cnt")
