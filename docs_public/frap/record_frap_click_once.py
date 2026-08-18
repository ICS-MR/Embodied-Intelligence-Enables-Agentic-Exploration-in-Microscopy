from __future__ import annotations

import ctypes
import json
import sys
import time
from datetime import datetime
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parent / "frap_click_points.json"
PROFILE_PATH = Path(__file__).resolve().parent / "frap_ui_profile.json"
VK_LBUTTON = 0x01
VK_Y = 0x59


def _set_dpi_awareness() -> None:
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _get_cursor_position() -> tuple[int, int]:
    point = POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        raise RuntimeError("GetCursorPos failed")
    return int(point.x), int(point.y)


def _get_pyautogui_position() -> tuple[int, int] | None:
    try:
        import pyautogui
    except Exception:
        return None
    position = pyautogui.position()
    return int(position.x), int(position.y)


def _left_button_is_down() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)


def _key_was_pressed(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x0001)


def _wait_for_y() -> None:
    print("Keep the target app focused. Press global key 'y', then click one target point.")
    while True:
        if _key_was_pressed(VK_Y):
            print("Armed. Waiting for one left-click...")
            return
        time.sleep(0.02)


def _wait_for_left_click() -> dict:
    while _left_button_is_down():
        time.sleep(0.02)
    while True:
        if _left_button_is_down():
            win32_position = _get_cursor_position()
            pyautogui_position = _get_pyautogui_position()
            while _left_button_is_down():
                time.sleep(0.02)
            return {
                "win32_get_cursor_pos": {
                    "x": win32_position[0],
                    "y": win32_position[1],
                },
                "pyautogui_position": None
                if pyautogui_position is None
                else {
                    "x": pyautogui_position[0],
                    "y": pyautogui_position[1],
                },
            }
        time.sleep(0.01)


def _load_window_keyword() -> str:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return str(payload.get("window_title_keyword", "")).strip()


def _find_cellsens_window() -> dict | None:
    try:
        import pygetwindow
    except Exception:
        return None

    keyword = _load_window_keyword()
    matches = [
        window
        for window in pygetwindow.getWindowsWithTitle(keyword)
        if int(getattr(window, "width", 0)) > 0 and int(getattr(window, "height", 0)) > 0
    ]
    if not matches:
        return None
    matches.sort(
        key=lambda window: (
            not bool(getattr(window, "isMinimized", False)),
            int(getattr(window, "width", 0)) * int(getattr(window, "height", 0)),
            bool(getattr(window, "isActive", False)),
        ),
        reverse=True,
    )
    window = matches[0]
    return {
        "title": str(getattr(window, "title", "")),
        "left": int(getattr(window, "left", 0)),
        "top": int(getattr(window, "top", 0)),
        "width": int(getattr(window, "width", 0)),
        "height": int(getattr(window, "height", 0)),
        "is_minimized": bool(getattr(window, "isMinimized", False)),
        "is_active": bool(getattr(window, "isActive", False)),
    }


def _get_screen_size() -> dict:
    width = int(ctypes.windll.user32.GetSystemMetrics(0))
    height = int(ctypes.windll.user32.GetSystemMetrics(1))
    virtual_left = int(ctypes.windll.user32.GetSystemMetrics(76))
    virtual_top = int(ctypes.windll.user32.GetSystemMetrics(77))
    virtual_width = int(ctypes.windll.user32.GetSystemMetrics(78))
    virtual_height = int(ctypes.windll.user32.GetSystemMetrics(79))
    return {
        "width": width,
        "height": height,
        "virtual_left": virtual_left,
        "virtual_top": virtual_top,
        "virtual_width": virtual_width,
        "virtual_height": virtual_height,
    }


def _get_pyautogui_screen_size() -> dict | None:
    try:
        import pyautogui
    except Exception:
        return None
    size = pyautogui.size()
    return {"width": int(size.width), "height": int(size.height)}


def _load_existing_points() -> list[dict]:
    if not OUTPUT_PATH.exists():
        return []
    payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"Expected {OUTPUT_PATH} to contain a JSON list.")


def main() -> int:
    _set_dpi_awareness()
    if "--diagnose" in sys.argv:
        print(f"screen: {_get_screen_size()}")
        print(f"pyautogui_screen: {_get_pyautogui_screen_size()}")
        print(f"win32_cursor: {_get_cursor_position()}")
        print(f"pyautogui_cursor: {_get_pyautogui_position()}")
        print(f"diagnostic selected_window: {_find_cellsens_window()}")
        return 0
    _wait_for_y()
    positions = _wait_for_left_click()
    primary_position = positions["win32_get_cursor_pos"]
    x = int(primary_position["x"])
    y = int(primary_position["y"])
    window = _find_cellsens_window()
    screen = _get_screen_size()
    pyautogui_screen = _get_pyautogui_screen_size()
    record = {
        "coordinate_system": "screen_absolute_physical_pixels",
        "source_api": "Win32 GetCursorPos after per-monitor DPI awareness",
        "x": x,
        "y": y,
        "screen": screen,
        "pyautogui_screen": pyautogui_screen,
        "positions": positions,
        "diagnostics": {
            "window_title_keyword": _load_window_keyword(),
            "selected_window": window,
        },
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    points = _load_existing_points()
    points.append(record)
    OUTPUT_PATH.write_text(
        json.dumps(points, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded absolute screen click: x={x}, y={y}")
    print(f"screen: {screen}")
    print(f"pyautogui_screen: {pyautogui_screen}")
    print(f"positions: {positions}")
    print(f"diagnostic selected_window: {window}")
    print(f"Saved to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
