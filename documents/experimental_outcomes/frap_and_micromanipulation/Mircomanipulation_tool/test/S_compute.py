#!/usr/bin/env python3
import cv2
import os
import numpy as np
import pandas as pd

# Interactive polygon state.
points = []
last_computed_area = None

def redraw(img):
    """Draw the current polygon and interaction hints."""
    disp = img.copy()
    for i, p in enumerate(points):
        cv2.circle(disp, p, 3, (0, 255, 255), -1)
        if i > 0:
            cv2.line(disp, points[i-1], p, (0, 0, 255), 2)

    if last_computed_area is not None:
        text = f"Area: {last_computed_area:.2f}px"
        cv2.rectangle(disp, (8, 8), (8 + len(text)*9, 35), (0,0,0), -1)
        cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    hint = "Left click: add | z: undo | r: reset | c: calculate | q: confirm"
    cv2.displayOverlay("Select", hint, 1000)
    cv2.imshow("Select", disp)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw(param)

def get_area(img, title="Select region"):
    """Interactively select a polygon and calculate its area."""
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
                print("[CALCULATE] At least three points are required; area=0")
            else:
                cnt = np.array(points, np.int32).reshape((-1,1,2))
                last_computed_area = float(abs(cv2.contourArea(cnt)))
                print(f"[CALCULATE] Area={last_computed_area:.2f}px")
            redraw(img)
        elif k == ord('q'):
            if last_computed_area is None:
                print("[CONFIRM] Area was not calculated; using zero")
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
    # Select the reference object once using the first video.
    print("[INITIALIZE] Select the object region (c: calculate, q: confirm)")
    first_video = None
    for folder, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".avi"):
                first_video = os.path.join(folder, f)
                break
        if first_video:
            break
    if not first_video:
        print("No AVI files found"); return

    frame = get_last_frame(first_video)
    if frame is None:
        print("Unable to read the first frame"); return

    area_block = get_area(frame, title="Select the reference object region")
    print(f"[OBJECT AREA] = {area_block:.2f}px\n")

    results = []

    # Evaluate every video.
    for folder, _, files in os.walk(root_dir):
        for f in sorted(files):
            if not f.lower().endswith(".avi"): continue
            path = os.path.join(folder, f)
            print(f"\n[PROCESS] {path}")
            frame = get_last_frame(path)
            if frame is None: continue
            area_inner = get_area(frame, title=f"{f} - Select the region inside the dashed boundary")
            ratio = area_inner / area_block if area_block > 0 else 0
            success = "success" if ratio > 0.6 else "failure"
            print(f"[RESULT] {f}: inner area={area_inner:.2f}, ratio={ratio:.3f}, status={success}")

            results.append({
                "file": path,
                "object_area": area_block,
                "inner_area": area_inner,
                "area_ratio": ratio,
                "result": success
            })

    # Export results.
    if results:
        df = pd.DataFrame(results)
        save_path = os.path.join(root_dir, "area_results.xlsx")
        df.to_excel(save_path, index=False)
        print(f"\nSaved results to {save_path}")
    else:
        print("No results were produced")

if __name__ == "__main__":
    root_dir = "/home/nova/videos/Push_to_target_none"
    main(root_dir)
