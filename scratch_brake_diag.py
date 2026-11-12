"""scratch_brake_diag.py — 一次性诊断：确认传感器在收数、驱动在出力、以及 get_rect_position 的口径。"""
import time

from rect_manifold_calibrator import HardwareRectangleTrainer

t = HardwareRectangleTrainer()
try:
    t.move_to_rect_origin(settle_s=0.45)
    s = t.sensor
    u0 = t.manifold.evaluate_kinematic_pulse(0.0, 0.0)
    print("u0_kin          =", u0)
    print("origin_sensor_xy=", getattr(t, "origin_sensor_xy", None))
    print("get_position    =", s.get_position())
    print("get_rect_position=", t.get_rect_position())
    print("event_count     =", getattr(s, "_event_count", None))
    print("rejected_counts =", getattr(s, "rejected_counts", None))
    print("per_device      =", getattr(s, "per_device", None))
    print("device_filter   =", getattr(s, "device_filter", None))

    p0 = t.get_rect_position()
    print("\n-- 下发 u0x+150us，观察 1s --")
    t.driver.send_raw_pulses(float(u0[0]) + 150.0, float(u0[1]))
    for i in range(10):
        time.sleep(0.1)
        print(f"  t={0.1*(i+1):.1f}s  rect={t.get_rect_position()}")
    print("event_count     =", getattr(s, "_event_count", None))
    print("rejected_counts =", getattr(s, "rejected_counts", None))
    print("delta from p0   =", t.get_rect_position() - p0)
finally:
    t.close()
