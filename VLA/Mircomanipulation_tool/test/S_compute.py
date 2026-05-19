#!/usr/bin/env python3
import cv2
import os
import numpy as np
import pandas as pd

# === 全局变量 ===
points = []
last_computed_area = None

def redraw(img):
    """绘制当前点集和提示信息"""
    disp = img.copy()
    for i, p in enumerate(points):
        cv2.circle(disp, p, 3, (0, 255, 255), -1)
        if i > 0:
            cv2.line(disp, points[i-1], p, (0, 0, 255), 2)

    if last_computed_area is not None:
        text = f"面积: {last_computed_area:.2f}px"
        cv2.rectangle(disp, (8, 8), (8 + len(text)*9, 35), (0,0,0), -1)
        cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    hint = "左键添加 | z 撤销 | r 清空 | c 计算 | q 确认"
    cv2.displayOverlay("Select", hint, 1000)
    cv2.imshow("Select", disp)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw(param)

def get_area(img, title="选择区域"):
    """交互式选择多边形并计算面积"""
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
                print("[计算] 点数不足，面积=0")
            else:
                cnt = np.array(points, np.int32).reshape((-1,1,2))
                last_computed_area = float(abs(cv2.contourArea(cnt)))
                print(f"[计算] 面积={last_computed_area:.2f}px")
            redraw(img)
        elif k == ord('q'):
            if last_computed_area is None:
                print("[确认] 未按c计算，面积视为0")
                last_computed_area = 0.0
            cv2.destroyWindow("Select")
            return last_computed_area

def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开 {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def main(root_dir):
    # === 先选物块区域 ===
    print("[初始化] 请选择物块区域（按c计算，q确认）")
    first_video = None
    for folder, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".avi"):
                first_video = os.path.join(folder, f)
                break
        if first_video:
            break
    if not first_video:
        print("未找到任何.avi文件"); return

    frame = get_last_frame(first_video)
    if frame is None:
        print("无法读取第一帧"); return

    area_block = get_area(frame, title="请选择物块区域（全局一次）")
    print(f"[物块面积] = {area_block:.2f}px\n")

    results = []

    # === 遍历所有视频 ===
    for folder, _, files in os.walk(root_dir):
        for f in sorted(files):
            if not f.lower().endswith(".avi"): continue
            path = os.path.join(folder, f)
            print(f"\n[处理] {path}")
            frame = get_last_frame(path)
            if frame is None: continue
            area_inner = get_area(frame, title=f"{f} - 请选择虚线内部区域")
            ratio = area_inner / area_block if area_block > 0 else 0
            success = "成功" if ratio > 0.6 else "失败"
            print(f"[结果] {f}: 内部面积={area_inner:.2f}, 比={ratio:.3f}, 状态={success}")

            results.append({
                "文件名": path,
                "物块面积": area_block,
                "虚线内部面积": area_inner,
                "面积比": ratio,
                "结果": success
            })

    # === 输出结果 ===
    if results:
        df = pd.DataFrame(results)
        save_path = os.path.join(root_dir, "area_results.xlsx")
        df.to_excel(save_path, index=False)
        print(f"\n✅ 已保存结果至 {save_path}")
    else:
        print("未得到任何结果")

if __name__ == "__main__":
    root_dir = "/home/nova/视频/Push_to_target_none"
    main(root_dir)
# import cv2
# import numpy as np
# import sys
# import json
# from typing import List, Tuple

# WINDOW_NAME = "Interactive Polygon (L: left click add, u: undo, c: close & calc, r: reset, s: save, q/ESC: quit)"

# def polygon_area(points: List[Tuple[int, int]]) -> float:
#     """使用 OpenCV 的 contourArea 或者 shoelace 算法计算多边形像素面积"""
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
#             # 如果已经闭合，先把闭合状态清除
#             self.closed = False
#             self.area = 0.0
#         if self.points:
#             self.points.pop()
#         self._refresh()

#     def add_point(self, x:int, y:int):
#         if self.closed:
#             # 如果已经闭合，再次点击先重置
#             self.reset()
#         self.points.append((x,y))
#         self._refresh()

#     def close_and_calc(self):
#         if len(self.points) < 3:
#             print("[警告] 点数不足，无法闭合多边形（至少3个点）。")
#             return
#         self.closed = True
#         self.area = polygon_area(self.points)
#         self._refresh()
#         print(f"多边形已闭合，像素面积 = {self.area:.2f}")

#     def save_points(self, filename="polygon_points.json"):
#         payload = {
#             "points": self.points,
#             "area_pixels": float(self.area)
#         }
#         with open(filename, "w") as f:
#             json.dump(payload, f, indent=2)
#         print(f"已保存点集到: {filename}")

#     def _refresh(self):
#         """重绘 display 图像"""
#         self.display = self.orig.copy()
#         # 画半透明填充（若闭合）
#         if self.closed and len(self.points) >= 3:
#             overlay = self.display.copy()
#             pts = np.array(self.points, dtype=np.int32).reshape((-1,1,2))
#             cv2.fillPoly(overlay, [pts], color=(0,50,0))  # 深色半透明（示例）
#             alpha = 0.35
#             cv2.addWeighted(overlay, alpha, self.display, 1-alpha, 0, self.display)

#         # 画连线（不论闭合与否）
#         if len(self.points) >= 2:
#             for i in range(len(self.points)-1):
#                 cv2.line(self.display, self.points[i], self.points[i+1], (0,255,0), 2)  # 绿色实线用于实体物体
#             if self.closed:
#                 cv2.line(self.display, self.points[-1], self.points[0], (0,0,255), 2)  # 用不同颜色表示闭合边（示例）
#         # 画点
#         for p in self.points:
#             cv2.circle(self.display, p, 4, (0,255,255), -1)

#         # 显示面积文本（若闭合）
#         if self.closed:
#             text = f"Area (pixels): {self.area:.2f}"
#             # 在图像左上角写文本（带背景）
#             (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#             cv2.rectangle(self.display, (5,5), (10+tw, 15+th), (0,0,0), -1)
#             cv2.putText(self.display, text, (8, 15+int(th/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# def mouse_callback(event, x, y, flags, param):
#     drawer: PolygonDrawer = param
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawer.add_point(x, y)
#         # 也在控制台输出点坐标
#         print(f"添加点: ({x}, {y})")

# def interactive_polygon(image_path: str):
#     img = cv2.imread(image_path)
#     if img is None:
#         print("无法读取图像，请检查路径：", image_path)
#         return

#     drawer = PolygonDrawer(img)
#     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback(WINDOW_NAME, mouse_callback, drawer)

#     print("交互指南：")
#     print(" - 鼠标左键：添加点（顺序）")
#     print(" - u : 撤销上一个点")
#     print(" - c : 闭合并计算面积")
#     print(" - r : 重置所有点")
#     print(" - s : 保存点集到 polygon_points.json")
#     print(" - q / ESC : 退出")

#     while True:
#         cv2.imshow(WINDOW_NAME, drawer.display)
#         key = cv2.waitKey(20) & 0xFF
#         if key == ord('u'):
#             drawer.undo()
#             print("撤销上一个点")
#         elif key == ord('c'):
#             drawer.close_and_calc()
#         elif key == ord('r'):
#             drawer.reset()
#             print("已重置")
#         elif key == ord('s'):
#             drawer.save_points()
#         elif key == ord('q') or key == 27:
#             print("退出")
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("用法: /bin/python ~/Mircomanipulation_ws/test/S_compute.py ~/Mircomanipulation_ws/xx.png")
#     else:
#         interactive_polygon(sys.argv[1])
