"""
servo_driver.py - WS63 3-RRR Servo Driver with Play-Operator Backlash & Kinetic Nudge
=====================================================================================
Implements:
1. Auto-discovery of CH340 UART port (COM4 default, 115200 8N1).
2. Dual-Y mechanical anti-binding guard (S2 & S3 coordinated motion).
3. Continuous Asymmetric Play-Operator (Prandtl-Ishlinskii hysteresis element)
   for smooth gear backlash compensation without zero-crossing chatter.
4. Impulse-Retract Micro-Stepping (Sub-Deadband Kinetic Kick) to overcome
   MG996R internal comparator deadband (~5us) and static friction.
"""

import time
import serial
import serial.tools.list_ports
from typing import Optional, Tuple


class AsymmetricPlayOperator:
    """
    Continuous tooth-engagement state machine for a single servo axis.
    Tracks internal gear gap state g(t) in [-b_neg, +b_pos] so direction
    reversals smoothly traverse the exact mechanical backlash without jerking.
    """

    def __init__(self, b_pos: float = 22.0, b_neg: float = 22.0):
        self.b_pos = float(b_pos)
        self.b_neg = float(b_neg)
        self.last_kin_u: Optional[float] = None
        self.gap_offset: float = 0.0

    def reset(self, current_kin_u: float):
        self.last_kin_u = float(current_kin_u)
        self.gap_offset = 0.0

    def compute(self, target_kin_u: float) -> Tuple[float, float]:
        """
        Given ideal kinematic pulse `target_kin_u`, returns:
        (compensated_pulse_u, direction_delta_kin)
        """
        if self.last_kin_u is None:
            self.last_kin_u = float(target_kin_u)
            return float(target_kin_u), 0.0

        du = float(target_kin_u) - self.last_kin_u
        self.last_kin_u = float(target_kin_u)

        if du > 0.0:
            # Moving in positive direction -> drive gap_offset toward +b_pos
            self.gap_offset = min(self.b_pos, self.gap_offset + du)
        elif du < 0.0:
            # Moving in negative direction -> drive gap_offset toward -b_neg
            self.gap_offset = max(-self.b_neg, self.gap_offset + du)

        return float(target_kin_u) + self.gap_offset, du


class WS63ServoDriver:
    """
    High-precision host driver for BearPi-Pico H3863 (WS63) 4-channel servo firmware.
    """

    NEUTRAL_S1 = 1510  # GPIO 2: X-axis horizontal push-pull
    NEUTRAL_S2 = 1545  # GPIO 5: Y1-axis top-right vertical main
    NEUTRAL_S3 = 1455  # GPIO 7: Y2-axis bottom-right vertical inverted
    NEUTRAL_S4 = 1500  # GPIO 8: Aux

    SAFE_MIN_US = 1150
    SAFE_MAX_US = 1850
    MAX_DUAL_Y_MISMATCH_US = 35  # Hard safety limit to prevent Y1/Y2 fighting

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        backlash_x: Tuple[float, float] = (22.0, 22.0),
        backlash_y: Tuple[float, float] = (15.0, 15.0),
        deadband_x: Tuple[float, float] = (5.0, 5.0),
        deadband_y: Tuple[float, float] = (5.0, 5.0),
    ):
        self.port = port or self.auto_detect_port()
        self.baudrate = baudrate
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)

        self.play_x = AsymmetricPlayOperator(b_pos=backlash_x[0], b_neg=backlash_x[1])
        self.play_y = AsymmetricPlayOperator(b_pos=backlash_y[0], b_neg=backlash_y[1])
        self.deadband_x = deadband_x
        self.deadband_y = deadband_y

        self.cur_s1 = float(self.NEUTRAL_S1)
        self.cur_s2 = float(self.NEUTRAL_S2)
        self.cur_s3 = float(self.NEUTRAL_S3)
        self.cur_s4 = float(self.NEUTRAL_S4)

        self.play_x.reset(self.cur_s1)
        self.play_y.reset(self.cur_s2)

    @staticmethod
    def auto_detect_port(default_port: str = "COM4") -> str:
        """Automatically locate CH340 (VID 1A86) serial port if COM number shifts."""
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").upper()
            hwid = (p.hwid or "").upper()
            if "CH340" in desc or "1A86" in hwid:
                return p.device
        return default_port

    def _clamp_and_guard(
        self, s1: float, s2: float, s3: Optional[float] = None, s4: float = 1500.0
    ) -> Tuple[int, int, int, int]:
        if s3 is None:
            s3 = 3000.0 - s2

        # Enforce Dual-Y coordination guard so S2 and S3 never fight each other
        y_sum_err = (s2 + s3) - 3000.0
        if abs(y_sum_err) > self.MAX_DUAL_Y_MISMATCH_US:
            s3 = 3000.0 - s2 + (
                self.MAX_DUAL_Y_MISMATCH_US if y_sum_err > 0 else -self.MAX_DUAL_Y_MISMATCH_US
            )

        u1 = int(round(max(self.SAFE_MIN_US, min(self.SAFE_MAX_US, s1))))
        u2 = int(round(max(self.SAFE_MIN_US, min(self.SAFE_MAX_US, s2))))
        u3 = int(round(max(self.SAFE_MIN_US, min(self.SAFE_MAX_US, s3))))
        u4 = int(round(max(self.SAFE_MIN_US, min(self.SAFE_MAX_US, s4))))
        return u1, u2, u3, u4

    def send_raw_pulses(
        self, s1: float, s2: float, s3: Optional[float] = None, s4: float = 1500.0
    ) -> Tuple[int, int, int, int]:
        """Directly send clamped pulse widths (in microseconds) to WS63."""
        u1, u2, u3, u4 = self._clamp_and_guard(s1, s2, s3, s4)
        cmd = f"P {u1} {u2} {u3} {u4}\n"
        self.ser.write(cmd.encode("ascii"))
        self.cur_s1, self.cur_s2, self.cur_s3, self.cur_s4 = float(u1), float(u2), float(u3), float(u4)
        return u1, u2, u3, u4

    def set_pwm_frequency(self, freq_hz: int = 50):
        """
        Set WS63 PWM output frequency (e.g., 50Hz standard, 100Hz, 200Hz overclock).
        Sends 'F <period_us>\n' to firmware.
        """
        period_us = int(round(1000000.0 / freq_hz))
        period_us = max(3000, min(50000, period_us))
        cmd = f"F {period_us}\n"
        self.ser.write(cmd.encode("ascii"))
        time.sleep(0.02)

    def center(self, settle_s: float = 0.5, preload: bool = True):
        """
        Return all servos to calibrated neutral positions.
        When preload=True, executes a robust 3-step sequence:
        1. Returns to neutral to cancel arbitrary shot displacements
        2. Steps back by (-45us, -45us) with 0.30s settling
        3. Approaches neutral deterministically from the negative side with 0.40s settling.
        Ensures 100% deterministic positive gear flank engagement at complete rest.
        """
        if preload:
            self.send_raw_pulses(self.NEUTRAL_S1, self.NEUTRAL_S2, self.NEUTRAL_S3, self.NEUTRAL_S4)
            time.sleep(0.15)
            self.send_raw_pulses(self.NEUTRAL_S1 - 45.0, self.NEUTRAL_S2 - 45.0)
            time.sleep(0.30)
        self.play_x.reset(self.NEUTRAL_S1)
        self.play_y.reset(self.NEUTRAL_S2)
        self.play_x.last_dir = 1
        self.play_y.last_dir = 1
        self.send_raw_pulses(self.NEUTRAL_S1, self.NEUTRAL_S2, self.NEUTRAL_S3, self.NEUTRAL_S4)
        if settle_s > 0:
            time.sleep(max(0.40, settle_s))

    def move_smooth_raw(
        self, target_s1: float, target_s2: float, duration_s: float = 0.25, rate_hz: float = 50.0
    ):
        """Smoothly ramp raw pulses from current state to target."""
        steps = max(1, int(round(duration_s * rate_hz)))
        start_s1, start_s2 = self.cur_s1, self.cur_s2
        dt = 1.0 / rate_hz
        for i in range(1, steps + 1):
            alpha = i / float(steps)
            u1 = start_s1 + (target_s1 - start_s1) * alpha
            u2 = start_s2 + (target_s2 - start_s2) * alpha
            self.send_raw_pulses(u1, u2)
            time.sleep(dt)

    def step_compensated_with_nudge(
        self,
        kin_s1: float,
        kin_s2: float,
        y_coord_offset: float = 0.0,
        enable_nudge: bool = True,
        frame_dt: float = 0.020,
    ) -> Tuple[int, int, int, int]:
        """
        Execute a kinematic command using:
        1. Continuous Play-Operator backlash compensation
        2. Sub-Deadband Kinetic Kick (Impulse-Retract) when step is inside deadband
        """
        comp_s1, du_x = self.play_x.compute(kin_s1)
        comp_s2, du_y = self.play_y.compute(kin_s2)
        comp_s3 = 3000.0 - comp_s2 + y_coord_offset

        if enable_nudge:
            kick_s1 = comp_s1
            kick_s2 = comp_s2
            need_kick = False

            # Check if X step is small (sub-deadband) and non-zero
            if 0.15 < abs(du_x) < self.deadband_x[0]:
                db_x = self.deadband_x[0] if du_x > 0 else -self.deadband_x[1]
                kick_s1 = comp_s1 + db_x * 0.85
                need_kick = True

            # Check if Y step is small (sub-deadband) and non-zero
            if 0.15 < abs(du_y) < self.deadband_y[0]:
                db_y = self.deadband_y[0] if du_y > 0 else -self.deadband_y[1]
                kick_s2 = comp_s2 + db_y * 0.85
                need_kick = True

            if need_kick:
                kick_s3 = 3000.0 - kick_s2 + y_coord_offset
                self.send_raw_pulses(kick_s1, kick_s2, kick_s3)
                time.sleep(frame_dt)

        return self.send_raw_pulses(comp_s1, comp_s2, comp_s3)

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.center(settle_s=0.1)
                self.ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    drv = WS63ServoDriver()
    print(f"Connected to WS63 on {drv.port}. Centering servos...")
    drv.center(settle_s=0.5)
    print("Servos centered at (1510, 1545, 1455, 1500) us.")
    drv.close()

