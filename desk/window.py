"""A see-through, always-on-top window that shows premultiplied RGBA pixels.
Copied from RewriteDesk/window.py. Only drawn pixels catch the mouse (so items
can be dragged); everywhere else clicks fall through to the desktop.

Win32 layered window (UpdateLayeredWindow, per-pixel alpha): pixels with alpha 0
let clicks fall through to whatever is underneath, so only Rewrite himself is
clickable. It never takes keyboard focus (WS_EX_NOACTIVATE) and has no taskbar
button (WS_EX_TOOLWINDOW). Plain ctypes, no extra packages.
"""
import ctypes
import ctypes.wintypes as wt

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

WS_POPUP = 0x80000000
WS_EX_LAYERED, WS_EX_TOPMOST, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80000, 0x8, 0x80, 0x08000000
WS_EX_TRANSPARENT = 0x20
WM_DESTROY, WM_COMMAND, WM_HOTKEY = 0x0002, 0x0111, 0x0312
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_RBUTTONUP, WM_MOUSEWHEEL = 0x0200, 0x0201, 0x0202, 0x0205, 0x020A
WM_LBUTTONDBLCLK, WM_MOUSEACTIVATE, MA_NOACTIVATE = 0x0203, 0x0021, 3
PM_REMOVE, ULW_ALPHA, AC_SRC_OVER, AC_SRC_ALPHA = 1, 2, 0, 1
SW_HIDE, SW_SHOWNOACTIVATE = 0, 4
HWND_TOPMOST = wt.HWND(-1)
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x1, 0x2, 0x10
MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x1, 0x2, 0x4000
TPM_RETURNCMD, TPM_RIGHTBUTTON = 0x100, 0x2
MF_STRING, MF_SEPARATOR, MF_CHECKED, MF_GRAYED, MF_POPUP = 0x0, 0x800, 0x8, 0x1, 0x10
SPI_GETWORKAREA = 0x30
CS_DBLCLKS = 0x8


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("style", wt.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wt.HINSTANCE),
                ("hIcon", wt.HICON), ("hCursor", wt.HANDLE), ("hbrBackground", wt.HBRUSH),
                ("lpszMenuName", wt.LPCWSTR), ("lpszClassName", wt.LPCWSTR), ("hIconSm", wt.HICON)]


user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.CreateWindowExW.restype = wt.HWND
user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID]
user32.UpdateLayeredWindow.argtypes = [wt.HWND, wt.HDC, ctypes.POINTER(wt.POINT), ctypes.POINTER(wt.SIZE), wt.HDC,
                                       ctypes.POINTER(wt.POINT), wt.COLORREF, ctypes.POINTER(BLENDFUNCTION), wt.DWORD]
user32.GetDC.restype = wt.HDC
user32.CreatePopupMenu.restype = wt.HMENU
user32.TrackPopupMenu.argtypes = [wt.HMENU, wt.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.HWND, wt.LPVOID]
user32.AppendMenuW.argtypes = [wt.HMENU, wt.UINT, ctypes.c_size_t, wt.LPCWSTR]
user32.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.UINT]
gdi32.CreateCompatibleDC.restype = wt.HDC
gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
gdi32.CreateDIBSection.restype = wt.HBITMAP
gdi32.CreateDIBSection.argtypes = [wt.HDC, ctypes.c_void_p, wt.UINT, ctypes.POINTER(ctypes.c_void_p), wt.HANDLE, wt.DWORD]
gdi32.SelectObject.restype = wt.HGDIOBJ
gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
kernel32.GetModuleHandleW.restype = wt.HMODULE


def dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        user32.SetProcessDPIAware()


def work_area():
    """The desktop minus the taskbar: (left, top, right, bottom) in pixels."""
    r = wt.RECT()
    user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(r), 0)
    return r.left, r.top, r.right, r.bottom


def cursor_pos():
    p = wt.POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


class OverlayWindow:
    """Events go to `on_event(name, **info)`; menu picks come back as ("menu", id=...)."""
    HOTKEY_ID = 1

    def __init__(self, width, height, on_event, title="ATS Playground"):
        self.on_event = on_event
        self.width, self.height = width, height
        self.x, self.y = 0, 0
        self.visible = True
        self._proc = WNDPROC(self._wndproc)       # kept alive: Windows calls back into it
        hinst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = CS_DBLCLKS
        wc.lpfnWndProc = self._proc
        wc.hInstance = hinst
        wc.hCursor = user32.LoadCursorW(None, 32649)    # the hand cursor over him
        wc.lpszClassName = "ATSPlayground"
        user32.RegisterClassExW(ctypes.byref(wc))
        self.hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            wc.lpszClassName, title, WS_POPUP, 0, 0, width, height, None, None, hinst, None)
        self.screen_dc = user32.GetDC(None)
        self.mem_dc = gdi32.CreateCompatibleDC(self.screen_dc)
        self.bitmap = None
        self._make_bitmap()
        self.hotkey = False     # Ctrl+Alt+R belongs to RewriteDesk; this one opens with Shift+A on the desktop
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        self.alive = True

    def _make_bitmap(self):
        if self.bitmap:
            gdi32.DeleteObject(self.bitmap)
        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth, bmi.biHeight = self.width, self.height     # positive height: bottom row first, like GL
        bmi.biPlanes, bmi.biBitCount = 1, 32
        self.bits = ctypes.c_void_p()
        self.bitmap = gdi32.CreateDIBSection(self.mem_dc, ctypes.byref(bmi), 0, ctypes.byref(self.bits), None, 0)
        gdi32.SelectObject(self.mem_dc, self.bitmap)

    def resize(self, width, height):
        self.width, self.height = int(width), int(height)
        self._make_bitmap()

    def show_pixels(self, rgba, x, y):
        """rgba: (height, width, 4) premultiplied, bottom row first."""
        self.x, self.y = int(x), int(y)
        bgra = rgba[..., [2, 1, 0, 3]]
        buf = bgra.tobytes()
        ctypes.memmove(self.bits, buf, len(buf))
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        pt, size, src = wt.POINT(self.x, self.y), wt.SIZE(self.width, self.height), wt.POINT(0, 0)
        user32.UpdateLayeredWindow(self.hwnd, self.screen_dc, ctypes.byref(pt), ctypes.byref(size), self.mem_dc,
                                   ctypes.byref(src), 0, ctypes.byref(blend), ULW_ALPHA)

    def keep_on_top(self):
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def set_visible(self, on):
        self.visible = on
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE if on else SW_HIDE)

    def menu(self, items):
        """items: (id, label, checked), (id, label, checked, enabled), (label, [items])
        for a submenu, or None for a separator. Returns the picked id or 0."""
        def build(items):
            h = user32.CreatePopupMenu()
            for item in items:
                if item is None:
                    user32.AppendMenuW(h, MF_SEPARATOR, 0, None)
                elif len(item) == 2:
                    label, sub = item
                    user32.AppendMenuW(h, MF_POPUP, build(sub), label)
                else:
                    mid, label, checked, *enabled = item
                    flags = MF_STRING | (MF_CHECKED if checked else 0) | (0 if enabled in ([], [True]) else MF_GRAYED)
                    user32.AppendMenuW(h, flags, mid, label)
            return h
        h = build(items)
        x, y = cursor_pos()
        user32.SetForegroundWindow(self.hwnd)      # needed, or the menu won't close when you click away
        pick = user32.TrackPopupMenu(h, TPM_RETURNCMD | TPM_RIGHTBUTTON, x, y, 0, self.hwnd, None)
        user32.DestroyMenu(h)                      # takes its submenus with it
        return pick

    def pump(self):
        msg = wt.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        return self.alive

    def close(self):
        if self.hotkey:
            user32.UnregisterHotKey(self.hwnd, self.HOTKEY_ID)
        user32.DestroyWindow(self.hwnd)
        self.alive = False

    def _wndproc(self, hwnd, msg, wparam, lparam):
        def xy():
            return ctypes.c_short(lparam & 0xFFFF).value, ctypes.c_short((lparam >> 16) & 0xFFFF).value
        if msg == WM_MOUSEACTIVATE:
            return MA_NOACTIVATE
        if msg == WM_HOTKEY and wparam == self.HOTKEY_ID:
            self.on_event("hotkey")
            return 0
        if msg == WM_LBUTTONDOWN:
            user32.SetCapture(hwnd)
            self.on_event("press", pos=xy())
            return 0
        if msg == WM_LBUTTONUP:
            user32.ReleaseCapture()
            self.on_event("release", pos=xy())
            return 0
        if msg == WM_LBUTTONDBLCLK:
            self.on_event("double", pos=xy())
            return 0
        if msg == WM_MOUSEMOVE:
            self.on_event("move", pos=xy())
            return 0
        if msg == WM_RBUTTONUP:
            self.on_event("right", pos=xy())
            return 0
        if msg == WM_MOUSEWHEEL:
            delta = ctypes.c_short((wparam >> 16) & 0xFFFF).value
            self.on_event("wheel", delta=delta)
            return 0
        if msg == WM_DESTROY:
            self.alive = False
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
