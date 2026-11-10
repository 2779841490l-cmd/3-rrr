"""
mouse_sensor.py - 3200 DPI High-Frequency Mouse Optical Displacement Sensor
===========================================================================
Supports two acquisition backends:
1. Windows RawInput API (WM_INPUT / lLastX, lLastY) [RECOMMENDED]:
   - Reads unscaled, unaccelerated raw optical counts directly from the Razer mouse.
   - Immune to Windows 'Enhance Pointer Precision' acceleration curve.
   - Immune to screen boundary clamping (infinite virtual workspace).
2. Cursor Position Delta + Auto-Recenter (GetCursorPos / SetCursorPos):
   - Fallback mode if RawInput message pump is not used.
"""

import ctypes
import ctypes.wintypes as wintypes
import threading
import time
from typing import Tuple

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Configure 64-bit Windows API argument and return types
user32.DefWindowProcW.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = ctypes.c_ssize_t

user32.GetRawInputData.argtypes = [
    wintypes.HANDLE,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_uint,
]
user32.GetRawInputData.restype = ctypes.c_uint

user32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL

user32.RegisterRawInputDevices.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
user32.RegisterRawInputDevices.restype = wintypes.BOOL

# Windows RawInput Constants
WM_INPUT = 0x00FF
WM_DESTROY = 0x0002
WM_QUIT = 0x0012
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
HID_USAGE_PAGE_GENERIC = 0x01
HID_USAGE_GENERIC_MOUSE = 0x02
MOUSE_MOVE_ABSOLUTE = 0x01


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.c_ushort),
        ("usUsage", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", ctypes.c_ulong),
        ("dwSize", ctypes.c_ulong),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWMOUSE_BUTTONS(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", ctypes.c_ushort),
        ("usButtonData", ctypes.c_ushort),
    ]


class RAWMOUSE_UNION(ctypes.Union):
    _fields_ = [
        ("ulButtons", ctypes.c_ulong),
        ("buttons", RAWMOUSE_BUTTONS),
    ]


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("usFlags", ctypes.c_ushort),
        ("u", RAWMOUSE_UNION),
        ("ulRawButtons", ctypes.c_ulong),
        ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long),
        ("ulExtraInformation", ctypes.c_ulong),
    ]


class RAWINPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mouse", RAWMOUSE),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("data", RAWINPUT_UNION),
    ]


WNDPROCTYPE = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM
)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROCTYPE),
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


class HighPrecisionMouseSensor:
    """
    Captures raw 3200 DPI optical displacement counts (dx, dy) in real time.
    Automatically falls back to cursor re-centering if RawInput registration fails.
    """

    def __init__(self, dpi: int = 3200, lock_cursor_center: bool = False):
        self.dpi = dpi
        self.mm_per_count = 25.4 / float(dpi)
        self.um_per_count = 25400.0 / float(dpi)
        self.lock_cursor_center = lock_cursor_center

        self._lock = threading.Lock()
        self._accum_x = 0
        self._accum_y = 0
        self._total_x = 0
        self._total_y = 0
        self._event_count = 0
        self._last_event_ts = 0.0

        self._running = False
        self._raw_input_active = False
        self._thread = None
        self._hwnd = None

        # Screen center for fallback or cursor locking
        self.screen_w = user32.GetSystemMetrics(0) or 1536
        self.screen_h = user32.GetSystemMetrics(1) or 864
        self.center_x = self.screen_w // 2
        self.center_y = self.screen_h // 2

    def start(self):
        if self._running:
            return
        self._running = True
        self._ready_evt = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        self._ready_evt.wait(timeout=2.0)

    def stop(self):
        self._running = False
        if self._hwnd:
            user32.PostMessageW(self._hwnd, WM_QUIT, 0, 0)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def reset_origin(self):
        """Zero out current accumulated position."""
        with self._lock:
            self._accum_x = 0
            self._accum_y = 0
            self._total_x = 0
            self._total_y = 0
            self._event_count = 0

    def read_delta_and_clear(self) -> Tuple[int, int]:
        """Return (dx, dy) counts since last call and clear delta accumulator."""
        with self._lock:
            dx, dy = self._accum_x, self._accum_y
            self._accum_x = 0
            self._accum_y = 0
            return dx, dy

    def get_position(self) -> Tuple[int, int]:
        """Return cumulative (X, Y) in raw counts since last reset_origin()."""
        with self._lock:
            return self._total_x, self._total_y

    def get_position_mm(self) -> Tuple[float, float]:
        """Return cumulative (X, Y) in physical millimeters."""
        x, y = self.get_position()
        return x * self.mm_per_count, y * self.mm_per_count

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            size = ctypes.c_uint(0)
            user32.GetRawInputData(
                wintypes.HANDLE(lparam),
                RID_INPUT,
                None,
                ctypes.byref(size),
                ctypes.sizeof(RAWINPUTHEADER),
            )
            if size.value > 0:
                buf = ctypes.create_string_buffer(size.value)
                if (
                    user32.GetRawInputData(
                        wintypes.HANDLE(lparam),
                        RID_INPUT,
                        buf,
                        ctypes.byref(size),
                        ctypes.sizeof(RAWINPUTHEADER),
                    )
                    == size.value
                ):
                    raw = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
                    if raw.header.dwType == RIM_TYPEMOUSE:
                        mouse = raw.data.mouse
                        if not (mouse.usFlags & MOUSE_MOVE_ABSOLUTE):
                            dx = int(mouse.lLastX)
                            dy = int(mouse.lLastY)
                            if dx != 0 or dy != 0:
                                with self._lock:
                                    self._accum_x += dx
                                    self._accum_y += dy
                                    self._total_x += dx
                                    self._total_y += dy
                                    self._event_count += 1
                                    self._last_event_ts = time.perf_counter()
                                if self.lock_cursor_center:
                                    user32.SetCursorPos(self.center_x, self.center_y)
            return 0
        elif msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _worker_loop(self):
        # Attempt to set up a message-only window for RawInput
        self._wnd_proc_cb = WNDPROCTYPE(self._wnd_proc)
        hinst = kernel32.GetModuleHandleW(None)
        class_name = f"RawMouseSink_{id(self)}"

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = self._wnd_proc_cb
        wc.hInstance = hinst
        wc.lpszClassName = class_name
        user32.RegisterClassExW(ctypes.byref(wc))

        HWND_MESSAGE = wintypes.HWND(-3)
        hwnd = user32.CreateWindowExW(
            0, class_name, "RawMouseWindow", 0, 0, 0, 0, 0, HWND_MESSAGE, None, hinst, None
        )
        self._hwnd = hwnd

        rid = RAWINPUTDEVICE()
        rid.usUsagePage = HID_USAGE_PAGE_GENERIC
        rid.usUsage = HID_USAGE_GENERIC_MOUSE
        rid.dwFlags = RIDEV_INPUTSINK
        rid.hwndTarget = hwnd

        if hwnd and user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
            self._raw_input_active = True
            self._ready_evt.set()
            msg = wintypes.MSG()
            while self._running and user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        else:
            # Fallback to high-rate cursor polling
            self._raw_input_active = False
            self._ready_evt.set()
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            last_x, last_y = pt.x, pt.y
            while self._running:
                if user32.GetCursorPos(ctypes.byref(pt)):
                    dx = pt.x - last_x
                    dy = pt.y - last_y
                    if dx != 0 or dy != 0:
                        with self._lock:
                            self._accum_x += dx
                            self._accum_y += dy
                            self._total_x += dx
                            self._total_y += dy
                            self._event_count += 1
                            self._last_event_ts = time.perf_counter()
                        if self.lock_cursor_center or (
                            pt.x < 100
                            or pt.x > self.screen_w - 100
                            or pt.y < 100
                            or pt.y > self.screen_h - 100
                        ):
                            user32.SetCursorPos(self.center_x, self.center_y)
                            last_x, last_y = self.center_x, self.center_y
                        else:
                            last_x, last_y = pt.x, pt.y
                time.sleep(0.002)


if __name__ == "__main__":
    print("Starting 3200 DPI High-Precision Mouse Sensor test (move mouse for 5 seconds)...")
    sensor = HighPrecisionMouseSensor(dpi=3200)
    sensor.start()
    print(f"Backend mode: {'Windows RawInput (1000Hz Direct Hardware Counts)' if sensor._raw_input_active else 'Cursor Polling Fallback'}")
    t0 = time.time()
    while time.time() - t0 < 5.0:
        x, y = sensor.get_position()
        mm_x, mm_y = sensor.get_position_mm()
        print(f"\rCounts: ({x:+6d}, {y:+6d}) | Physical: ({mm_x:+7.3f} mm, {mm_y:+7.3f} mm)", end="")
        time.sleep(0.05)
    sensor.stop()
    print("\nDone.")

