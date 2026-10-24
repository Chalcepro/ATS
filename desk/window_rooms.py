"""Open windows as rooms: each window's top and bottom edges are walls, its left
and right sides are open doorways. The brain (and the player) walk in and out
through the sides; dropping the brain on a window puts it inside; a window that
moves carries whoever is inside with it.

The brain already knows walls from its corridor rungs, so nothing is retrained."""
import ctypes
import ctypes.wintypes as wt

user32 = ctypes.WinDLL("user32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

GWL_EXSTYLE, WS_EX_TOOLWINDOW = -20, 0x80
DWMWA_EXTENDED_FRAME_BOUNDS, DWMWA_CLOAKED = 9, 14
SKIP_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd",
                "ATSPlayground", "RewriteDeskOverlay"}

EnumProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
user32.GetWindowLongW.restype = ctypes.c_long
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]


def open_windows():
    """Visible app windows, front to back, as (hwnd, left, top, right, bottom) in screen pixels."""
    found = []

    def each(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        if user32.GetWindowTextLengthW(hwnd) == 0 or user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, buf, 64)
        if buf.value in SKIP_CLASSES:
            return True
        cloaked = ctypes.c_int(0)       # hidden Store-app windows report visible but are cloaked
        dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), 4)
        if cloaked.value:
            return True
        r = wt.RECT()                   # the visible frame, without the invisible resize border
        if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)):
            user32.GetWindowRect(hwnd, ctypes.byref(r))
        if r.right - r.left > 40 and r.bottom - r.top > 40:
            found.append((hwnd, r.left, r.top, r.right, r.bottom))
        return True

    user32.EnumWindows(EnumProc(each), 0)
    return found


class Rooms:
    """Turns window rectangles into tile rooms and keeps the world's walls in step."""

    def __init__(self, left, top, tile_px, cols, rows):
        self.left, self.top, self.px = left, top, tile_px
        self.cols, self.rows = cols, rows
        self.rooms = {}                 # hwnd -> (x0, y0, x1, y1) in tiles, walls on rows y0 and y1

    def to_tiles(self, l, t, r, b):
        p = self.px
        return ((l - self.left) // p, (t - self.top) // p, (r - 1 - self.left) // p, (b - 1 - self.top) // p)

    def room_of(self, x, y):
        """The window the tile is inside (between its walls), frontmost first."""
        for hwnd, (x0, y0, x1, y1) in self.rooms.items():
            if x0 <= x <= x1 and y0 < y < y1:
                return hwnd
        return None

    def walls(self):
        out = set()
        for x0, y0, x1, y1 in self.rooms.values():
            for x in range(max(x0, 0), min(x1, self.cols - 1) + 1):
                for y in (y0, y1):
                    if 0 <= y < self.rows:
                        out.add((x, y))
        return out

    def refresh(self, rooms=None):
        """Re-read the windows. Returns {hwnd: (dx, dy)} for rooms that moved, in tiles."""
        new = {h: self.to_tiles(*rect) for h, *rect in (open_windows() if rooms is None else rooms)}
        moved = {h: (r[0] - self.rooms[h][0], r[1] - self.rooms[h][1])
                 for h, r in new.items() if h in self.rooms and r[:2] != self.rooms[h][:2]}
        self.rooms = new
        return moved
