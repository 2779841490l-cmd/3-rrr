"""
mouse_sensor.py - 3200 DPI High-Frequency Mouse Displacement Sensor
===================================================================
Attaches to the active Windows Interactive Input Desktop (OpenInputDesktop)
to bypass session isolation, reading high-frequency cursor deltas (dx, dy)
with automatic virtual-canvas recentering so screen edges never clamp counts.
"""

import ctypes
import ctypes.wintypes as wintypes
import threading
import time
from typing import Tuple

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
winmm = ctypes.windll.winmm

# Enable 1ms Windows multimedia timer resolution so 1000Hz polling never sleeps 15.6ms
try:
    winmm.timeBeginPeriod(1)
except Exception:
    pass

# Configure win32 types
user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
user32.OpenInputDesktop.restype = wintypes.HANDLE

user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
user32.SetThreadDesktop.restype = wintypes.BOOL


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def attach_interactive_desktop() -> bool:
    """Ensure current thread is attached to the active interactive input desktop."""
    try:
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            return bool(user32.SetThreadDesktop(hdesk))
    except Exception:
        pass
    return False


class HighPrecisionMouseSensor:
    """
    Captures mouse optical displacement counts (dx, dy) at 3200 DPI.
    Maintains an unconstrained continuous virtual coordinate frame by
    silently recentering the cursor when it nears screen edges.
    """

    def __init__(self, dpi: int = 3200, sample_interval_s: float = 0.002):
        self.dpi = dpi
        self.mm_per_count = 25.4 / float(dpi)
        self.um_per_count = 25400.0 / float(dpi)
        self.sample_interval_s = sample_interval_s

        self._lock = threading.Lock()
        self._accum_x = 0
        self._accum_y = 0
        self._total_x = 0
        self._total_y = 0
        self._event_count = 0

        self._running = False
        self._thread = None
        self._ready_evt = threading.Event()

        # Virtual recenter boundaries
        attach_interactive_desktop()
        self.screen_w = user32.GetSystemMetrics(0) or 1536
        self.screen_h = user32.GetSystemMetrics(1) or 864
        self.center_x = self.screen_w // 2
        self.center_y = self.screen_h // 2
        # Wide 240px safety margin prevents fast macro flicks (>3000 cnt/s) from ever touching screen edges
        self.margin = 240
        self._force_recenter = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._ready_evt.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self._ready_evt.wait(timeout=2.0)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def reset_origin(self):
        """Zero out accumulated coordinates and recenter OS cursor."""
        with self._lock:
            self._accum_x = 0
            self._accum_y = 0
            self._total_x = 0
            self._total_y = 0
            self._event_count = 0
            self._force_recenter = True

    def recenter_cursor(self):
        """
        Safely warps the Windows OS cursor back to screen center (center_x, center_y)
        synchronously inside the polling lock while the robot is stationary before a shot,
        guaranteeing maximum screen headroom and zero edge clipping even on ultra-large flicks.
        """
        with self._lock:
            self._force_recenter = True

    def read_delta_and_clear(self) -> Tuple[int, int]:
        """Return (dx, dy) counts since last call and reset delta accumulator."""
        with self._lock:
            dx, dy = self._accum_x, self._accum_y
            self._accum_x = 0
            self._accum_y = 0
            return dx, dy

    def get_position(self) -> Tuple[int, int]:
        """Return cumulative (X, Y) counts."""
        with self._lock:
            return self._total_x, self._total_y

    def get_position_mm(self) -> Tuple[float, float]:
        """Return cumulative (X, Y) in millimeters."""
        x, y = self.get_position()
        return x * self.mm_per_count, y * self.mm_per_count

    def _poll_loop(self):
        attach_interactive_desktop()
        pt = POINT()
        if not user32.GetCursorPos(ctypes.byref(pt)):
            # Retry attaching once if needed
            time.sleep(0.05)
            attach_interactive_desktop()
            user32.GetCursorPos(ctypes.byref(pt))

        last_x, last_y = pt.x, pt.y
        self._ready_evt.set()

        while self._running:
            if user32.GetCursorPos(ctypes.byref(pt)):
                cur_x, cur_y = pt.x, pt.y
                dx = cur_x - last_x
                dy = cur_y - last_y

                with self._lock:
                    if dx != 0 or dy != 0:
                        self._accum_x += dx
                        self._accum_y += dy
                        self._total_x += dx
                        self._total_y += dy
                        self._event_count += 1

                    need_warp = (
                        self._force_recenter
                        or cur_x < self.margin
                        or cur_x > self.screen_w - self.margin
                        or cur_y < self.margin
                        or cur_y > self.screen_h - self.margin
                    )
                    if need_warp:
                        self._force_recenter = False
                        user32.SetCursorPos(self.center_x, self.center_y)
                        last_x, last_y = self.center_x, self.center_y
                    else:
                        last_x, last_y = cur_x, cur_y

            time.sleep(self.sample_interval_s)


if __name__ == "__main__":
    print("Testing 3200 DPI Interactive Mouse Sensor for 3 seconds...")
    s = HighPrecisionMouseSensor(dpi=3200)
    s.start()
    t0 = time.time()
    while time.time() - t0 < 3.0:
        pos = s.get_position()
        print(f"\rPosition: {pos}", end="")
        time.sleep(0.05)
    s.stop()
    print("\nDone.")
