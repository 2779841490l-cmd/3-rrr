import time
from mouse_sensor import HighPrecisionMouseSensor
from servo_driver import WS63ServoDriver

drv = WS63ServoDriver()
sensor = HighPrecisionMouseSensor()
sensor.start()

print("Testing coordinated kinematics preload...")
for i in range(5):
    # Step 1: neutral
    drv.send_raw_pulses(1510, 1545, 1455, 1500)
    time.sleep(0.10)
    
    # Step 2: Coordinated preload (-X by 30us, -Y by 25us)
    # Notice: -X is S1=1480. -Y is S2=1570, S3=1430 (sum = 3000 exactly!)
    drv.send_raw_pulses(1480, 1570, 1430, 1500)
    time.sleep(0.20)
    
    # Step 3: Return to neutral (+X by 30us, +Y by 25us)
    drv.send_raw_pulses(1510, 1545, 1455, 1500)
    time.sleep(0.40)
    
    sensor.read_delta_and_clear()
    t0 = time.time()
    drift = [0, 0]
    while time.time() - t0 < 0.2:
        time.sleep(0.02)
        dx, dy = sensor.read_delta_and_clear()
        drift[0] += dx
        drift[1] += dy
    print(f"Cycle {i+1} -> Settle drift in 0.2s: dx={drift[0]}, dy={drift[1]}")

sensor.stop()
drv.close()

