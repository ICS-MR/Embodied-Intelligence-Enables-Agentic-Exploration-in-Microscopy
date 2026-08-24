#!/usr/bin/env python3
import cv2
import os
import numpy as np
import pandas as pd

# === Global variables ===
points = []
last_computed_area = None

def redraw(img):
    """Draw the current points and instructions."""
    disp = img.copy()
    for i, p in enumerate(points):
        cv2.circle(disp, p, 3, (0, 255, 255), -1)
        if i > 0:
            cv2.line(disp, points[i-1], p, (0, 0, 255), 2)

    if last_computed_area is not None:
        text = f"Area: {last_computed_area:.2f}px"
        cv2.rectangle(disp, (8, 8), (8 + len(text)*9, 35), (0,0,0), -1)
        cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    hint = "Left click: add | z: undo | r: clear | c: calculate | q: confirm"
    cv2.displayOverlay("Select", hint, 1000)
    cv2.imshow("Select", disp)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw(param)

def get_area(img, title="Select Region"):
    """Select a polygon interactively and calculate its area."""
    global points, last_computed_area
    points, last_computed_area = [], None
    cv2.namedWindow("Select", cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("Select", title)
    cv2.setMouseCallback("Select", on_mouse, img)
    cv2.imshow("Select", img)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('z') and points:
            points.pop()
            redraw(img)
        elif k == ord('r'):
            points, last_computed_area = [], None
            redraw(img)
        elif k == ord('c'):
            if len(points) < 3:
                last_computed_area = 0.0
                print("[CALCULATION] Not enough points; area = 0")
            else:
                cnt = np.array(points, np.int32).reshape((-1,1,2))
                last_computed_area = float(abs(cv2.contourArea(cnt)))
                print(f"[CALCULATION] Area = {last_computed_area:.2f}px")
            redraw(img)
        elif k == ord('q'):
            if last_computed_area is None:
                print("[CONFIRMATION] Area treated as 0 because c was not pressed")
                last_computed_area = 0.0
            cv2.destroyWindow("Select")
            return last_computed_area

def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def main(root_dir):
    # === Select the object region first ===
    print("[INITIALIZATION] Select the object region (c to calculate, q to confirm)")
    first_video = None
    for folder, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".avi"):
                first_video = os.path.join(folder, f)
                break
        if first_video:
            break
    if not first_video:
        print("No .avi files found"); return

    frame = get_last_frame(first_video)
    if frame is None:
        print("Unable to read the first frame"); return

    area_block = get_area(frame, title="Select the object region (once for all videos)")
    print(f"[OBJECT AREA] = {area_block:.2f}px\n")

    results = []

    # === Iterate over all videos ===
    for folder, _, files in os.walk(root_dir):
        for f in sorted(files):
            if not f.lower().endswith(".avi"): continue
            path = os.path.join(folder, f)
            print(f"\n[PROCESSING] {path}")
            frame = get_last_frame(path)
            if frame is None: continue
            area_inner = get_area(frame, title=f"{f} - Select the region inside the dashed line")
            ratio = area_inner / area_block if area_block > 0 else 0
            success = "Success" if ratio > 0.6 else "Failure"
            print(f"[RESULT] {f}: inner area={area_inner:.2f}, ratio={ratio:.3f}, status={success}")

            results.append({
                "Filename": path,
                "Object Area": area_block,
                "Area Inside Dashed Line": area_inner,
                "Area Ratio": ratio,
                "Result": success
            })

    # === Output results ===
    if results:
        df = pd.DataFrame(results)
        save_path = os.path.join(root_dir, "area_results.xlsx")
        df.to_excel(save_path, index=False)
        print(f"\n✅ Results saved to {save_path}")
    else:
        print("No results were produced")

if __name__ == "__main__":
    root_dir = "/home/nova/videos/Push_to_target_none"
    main(root_dir)
# import cv2
# import numpy as np
# import sys
# import json
# from typing import List, Tuple

# WINDOW_NAME = "Interactive Polygon (L: left click add, u: undo, c: close & calc, r: reset, s: save, q/ESC: quit)"

# def polygon_area(points: List[Tuple[int, int]]) -> float:
#     """Calculate polygon area in pixels with OpenCV contourArea or the shoelace formula."""
#     if len(points) < 3:
#         return 0.0
#     cnt = np.array(points, dtype=np.int32).reshape((-1,1,2))
#     return abs(cv2.contourArea(cnt))

# class PolygonDrawer:
#     def __init__(self, img: np.ndarray):
#         self.orig = img.copy()
#         self.display = img.copy()
#         self.points: List[Tuple[int,int]] = []
#         self.closed = False
#         self.area = 0.0

#     def reset(self):
#         self.points = []
#         self.closed = False
#         self.area = 0.0
#         self.display = self.orig.copy()
#         self._refresh()

#     def undo(self):
#         if self.closed:
#             # Clear the closed state first if the polygon is already closed.
#             self.closed = False
#             self.area = 0.0
#         if self.points:
#             self.points.pop()
#         self._refresh()

#     def add_point(self, x:int, y:int):
#         if self.closed:
#             # Reset first when clicking again after the polygon has been closed.
#             self.reset()
#         self.points.append((x,y))
#         self._refresh()

#     def close_and_calc(self):
#         if len(self.points) < 3:
#             print("[WARNING] Not enough points to close the polygon; at least three are required.")
#             return
#         self.closed = True
#         self.area = polygon_area(self.points)
#         self._refresh()
#         print(f"Polygon closed. Pixel area = {self.area:.2f}")

#     def save_points(self, filename="polygon_points.json"):
#         payload = {
#             "points": self.points,
#             "area_pixels": float(self.area)
#         }
#         with open(filename, "w") as f:
#             json.dump(payload, f, indent=2)
#         print(f"Point set saved to: {filename}")

#     def _refresh(self):
#         """Redraw the display image."""
#         self.display = self.orig.copy()
#         # Draw a translucent fill if the polygon is closed.
#         if self.closed and len(self.points) >= 3:
#             overlay = self.display.copy()
#             pts = np.array(self.points, dtype=np.int32).reshape((-1,1,2))
#             cv2.fillPoly(overlay, [pts], color=(0,50,0))  # Dark translucent fill example.
#             alpha = 0.35
#             cv2.addWeighted(overlay, alpha, self.display, 1-alpha, 0, self.display)

#         # Draw connecting lines whether or not the polygon is closed.
#         if len(self.points) >= 2:
#             for i in range(len(self.points)-1):
#                 cv2.line(self.display, self.points[i], self.points[i+1], (0,255,0), 2)  # Green solid line for the object.
#             if self.closed:
#                 cv2.line(self.display, self.points[-1], self.points[0], (0,0,255), 2)  # Different color for the closing edge.
#         # Draw points.
#         for p in self.points:
#             cv2.circle(self.display, p, 4, (0,255,255), -1)

#         # Display area text if the polygon is closed.
#         if self.closed:
#             text = f"Area (pixels): {self.area:.2f}"
#             # Draw text with a background in the upper-left corner.
#             (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#             cv2.rectangle(self.display, (5,5), (10+tw, 15+th), (0,0,0), -1)
#             cv2.putText(self.display, text, (8, 15+int(th/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# def mouse_callback(event, x, y, flags, param):
#     drawer: PolygonDrawer = param
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawer.add_point(x, y)
#         # Also print the point coordinates to the console.
#         print(f"Added point: ({x}, {y})")

# def interactive_polygon(image_path: str):
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Unable to read the image. Check the path:", image_path)
#         return

#     drawer = PolygonDrawer(img)
#     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback(WINDOW_NAME, mouse_callback, drawer)

#     print("Interaction guide:")
#     print(" - Left mouse button: add points in order")
#     print(" - u: undo the previous point")
#     print(" - c: close the polygon and calculate its area")
#     print(" - r: reset all points")
#     print(" - s: save points to polygon_points.json")
#     print(" - q / ESC: exit")

#     while True:
#         cv2.imshow(WINDOW_NAME, drawer.display)
#         key = cv2.waitKey(20) & 0xFF
#         if key == ord('u'):
#             drawer.undo()
#             print("Undid the previous point")
#         elif key == ord('c'):
#             drawer.close_and_calc()
#         elif key == ord('r'):
#             drawer.reset()
#             print("Reset complete")
#         elif key == ord('s'):
#             drawer.save_points()
#         elif key == ord('q') or key == 27:
#             print("Exiting")
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: /bin/python ~/Mircomanipulation_ws/test/S_compute.py ~/Mircomanipulation_ws/xx.png")
#     else:
#         interactive_polygon(sys.argv[1])
