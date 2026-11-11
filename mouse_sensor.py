"""
mouse_sensor.py - True Hardware 3200 DPI Raw Input Displacement Sensor
=====================================================================
Captures hardware-level 3200 DPI optical sensor displacements (lLastX, lLastY)
via Windows Raw Input (WM_INPUT + RIDEV_INPUTSINK) on the active interactive
desktop.

Why Raw Input instead of GetCursorPos:
1. Completely immune to Windows "Enhance Pointer Precision" nonlinear acceleration.
2. Zero screen-edge clamping: optical counts continue accurately even beyond the
   1536x864 physical desktop boundaries.
3. True physical resolution: 1 count = 25.4 / 3200 mm = 7.9375 micrometers.
"""

import ctypes
import ctypes.wintypes as wintypes
import threading
import time
from typing import Tuple

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
winmm = ctypes.windll.winmm

# Enable 1ms Windows multimedia timer resolution for 1000Hz ballistic loops
try:
    winmm.timeBeginPeriod(1)
except Exception:
    pass

WM_INPUT = 0x00FF
WM_QUIT = 0x0012
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)

user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
user32.OpenInputDesktop.restype = wintypes.HANDLE
user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
user32.SetThreadDesktop.restype = wintypes.BOOL
user32.DefWindowProcW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
]
user32.DefWindowProcW.restype = LRESULT
user32.GetRawInputData.argtypes = [
    ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p,
    ctypes.POINTER(wintypes.UINT), wintypes.UINT
]
user32.GetRawInputData.restype = wintypes.UINT


class _WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class _RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class _RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _RAW_BUTTONS_STRUCT(ctypes.Structure):
    _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]


class _RAW_BUTTONS_UNION(ctypes.Union):
    _fields_ = [("ulButtons", wintypes.ULONG), ("buttons", _RAW_BUTTONS_STRUCT)]


class _RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("u", _RAW_BUTTONS_UNION),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class _RAWINPUT_MOUSE(ctypes.Structure):
    _fields_ = [("header", _RAWINPUTHEADER), ("mouse", _RAWMOUSE)]


def attach_interactive_desktop() -> bool:
    """Attach the calling thread to the active interactive input desktop."""
    try:
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            ok = bool(user32.SetThreadDesktop(hdesk))
            user32.CloseDesktop(hdesk)
            return ok
    except Exception:
        pass
    return False


class HighPrecisionMouseSensor:
    """
    Event-driven hardware 3200 DPI Raw Input optical displacement sensor.
    Records true unaccelerated HID relative counts (1 count = 7.9375 um).
    """

    def __init__(self, dpi: int = 3200, sample_interval_s: float = 0.001):
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
        self._hwnd = None
        self._wndproc_ref = None

        attach_interactive_desktop()
        self.screen_w = user32.GetSystemMetrics(0) or 1536
        self.screen_h = user32.GetSystemMetrics(1) or 864
        self.center_x = self.screen_w // 2
        self.center_y = self.screen_h // 2

    def start(self):
        if self._running:
            return
        self._running = True
        self._ready_evt.clear()
        self._thread = threading.Thread(target=self._raw_input_loop, daemon=True)
        self._thread.start()
        self._ready_evt.wait(timeout=2.0)

    def stop(self):
        self._running = False
        if self._hwnd:
            try:
                user32.PostMessageW(self._hwnd, WM_QUIT, 0, 0)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._hwnd = None

    def reset_origin(self):
        """Zero out accumulated coordinates and optionally recenter OS cursor."""
        with self._lock:
            self._accum_x = 0
            self._accum_y = 0
            self._total_x = 0
            self._total_y = 0
            self._event_count = 0

    def recenter_cursor(self):
        """Move visible OS cursor to screen center cleanly when stationary."""
        try:
            user32.SetCursorPos(self.center_x, self.center_y)
        except Exception:
            pass

    def read_delta_and_clear(self) -> Tuple[int, int]:
        """Return (dx, dy) raw 3200 DPI counts since last call and clear accumulator."""
        with self._lock:
            dx, dy = self._accum_x, self._accum_y
            self._accum_x = 0
            self._accum_y = 0
            return dx, dy

    def get_position(self) -> Tuple[int, int]:
        """Return cumulative raw 3200 DPI (X, Y) counts."""
        with self._lock:
            return self._total_x, self._total_y

    def get_position_mm(self) -> Tuple[float, float]:
        """Return cumulative (X, Y) in millimeters."""
        x, y = self.get_position()
        return x * self.mm_per_count, y * self.mm_per_count

    def _raw_input_loop(self):
        attach_interactive_desktop()
        hinst = kernel32.GetModuleHandleW(None)
        cls_name = f"RRR_RawMouse_{time.time_ns()}"
        buf = (ctypes.c_ubyte * 128)()
        sz = wintypes.UINT(128)
        hdr_sz = ctypes.sizeof(_RAWINPUTHEADER)
        min_sz = ctypes.sizeof(_RAWINPUT_MOUSE)

        def wnd_proc(hwnd, msg, wp, lp):
            if msg == WM_INPUT:
                sz.value = 128
                res = user32.GetRawInputData(
                    ctypes.c_void_p(lp), RID_INPUT, buf, ctypes.byref(sz), hdr_sz
                )
                if res != 0xFFFFFFFF and sz.value >= min_sz:
                    ri = ctypes.cast(buf, ctypes.POINTER(_RAWINPUT_MOUSE)).contents
                    # Only accept true hardware relative optical sensor deltas (ignore absolute/synthetic)
                    if ri.header.dwType == RIM_TYPEMOUSE and (ri.mouse.usFlags & 0x01) == 0:
                        dx = int(ri.mouse.lLastX)
                        dy = int(ri.mouse.lLastY)
                        if dx != 0 or dy != 0:
                            with self._lock:
                                self._accum_x += dx
                                self._accum_y += dy
                                self._total_x += dx
                                self._total_y += dy
                                self._event_count += 1
                return 0
            return user32.DefWindowProcW(hwnd, msg, wp, lp)

        self._wndproc_ref = WNDPROC(wnd_proc)
        wc = _WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(_WNDCLASSEXW)
        wc.lpfnWndProc = self._wndproc_ref
        wc.hInstance = hinst
        wc.lpszClassName = cls_name
        user32.RegisterClassExW(ctypes.byref(wc))

        self._hwnd = user32.CreateWindowExW(
            0, cls_name, "3RRR_RawInput", 0, 0, 0, 0, 0, None, None, hinst, None
        )
        rid = _RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, self._hwnd)
        ok = user32.RegisterRawInputDevices(
            ctypes.byref(rid), 1, ctypes.sizeof(_RAWINPUTDEVICE)
        )
        if not ok:
            print("[WARN] RegisterRawInputDevices failed; check interactive desktop attachment.")
        self._ready_evt.set()

        msg = wintypes.MSG()
        while self._running and user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        if self._hwnd:
            user32.DestroyWindow(self._hwnd)
            self._hwnd = None


if __name__ == "__main__":
    print("Testing True 3200 DPI Raw Input Mouse Sensor for 3 seconds...")
    s = HighPrecisionMouseSensor(dpi=3200)
    s.start()
    s.reset_origin()
    s.recenter_cursor()
    time.sleep(0.05)
    print(f"After recenter_cursor while stationary: pos={s.get_position()}")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        pos = s.get_position()
        mm = s.get_position_mm()
        print(f"\rCounts: {pos} | mm: ({mm[0]:+.3f}, {mm[1]:+.3f})", end="")
        time.sleep(0.05)
    s.stop()
    print("\nDone.")
