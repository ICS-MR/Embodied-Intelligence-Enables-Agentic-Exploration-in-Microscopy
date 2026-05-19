import gxipy as gx
import numpy as np
import cv2
import signal
import sys
import time
from threading import Thread, Lock
from ctypes import c_ubyte, addressof

class DahengCamera:
    def __init__(self, width=1280, height=960, display_size=(640, 480)):
        self.device_manager = gx.DeviceManager()
        self.cam = None
        self.running = False
        self.width = width
        self.height = height
        self.display_size = display_size

        # 同步与状态控制
        self.cam_lock = Lock()           # 保护对 self.cam 的所有访问
        self.frame_lock = Lock()
        self.latest_frame = None
        self.thread = None
        self.thread_started = False     # 防止重复 start()
        self.converter = None           # 重用转换器（在相机初始化成功后创建）

        # 先尝试初始化一次（若失败会把 self.cam 置 None）
        self._reinitialize_camera()

        # Ctrl+C 安全退出
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError:
            print("Warning: Failed to register signal handler (not in main thread). Ignoring.")

    # ---------- 相机打开/关闭/初始化相关 ----------
    def _reinitialize_camera(self):
        """安全重连：先释放旧设备，再重新打开并start stream"""
        with self.cam_lock:
            self._release_camera_locked()  # 内部假定持有 cam_lock

            try:
                dev_num, dev_info_list = self.device_manager.update_all_device_list()
                if dev_num == 0:
                    raise Exception("未找到相机设备")

                sn = dev_info_list[0].get("sn")

                # open device
                try:
                    self.cam = self.device_manager.open_device_by_sn(sn)
                except Exception as e:
                    # 尝试根据异常信息做替代打开
                    if "already been opened" in str(e) or "repeat open" in str(e).lower():
                        print("[警告] open_device_by_sn 报 already opened，尝试 open_device_by_index")
                        try:
                            self.cam = self.device_manager.open_device_by_index(0)
                        except Exception as e2:
                            raise e2
                    else:
                        raise e

                # 配置参数（放在打开后）
                try:
                    feature = self.cam.get_remote_device_feature_control()
                    if feature.is_writable("Width"):
                        feature.get_int_feature("Width").set(self.width)
                    if feature.is_writable("Height"):
                        feature.get_int_feature("Height").set(self.height)
                except Exception as e:
                    print(f"[警告] 设置 Width/Height 失败: {e}")

                # 创建并缓存转换器（如果 device_manager 支持）
                try:
                    self.converter = self.device_manager.create_image_format_convert()
                    self.converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)
                except Exception:
                    self.converter = None

                # 启动流
                try:
                    self.cam.stream_on()
                except Exception as e:
                    # 如果 stream_on 失败，确保释放并抛出
                    self._release_camera_locked()
                    raise e

                print("[信息] 相机初始化成功")

            except Exception as e:
                print(f"[错误] 相机初始化失败: {e}")
                # 确保 cam 为 None（表示未打开）
                try:
                    self._release_camera_locked()
                except Exception:
                    pass
                self.cam = None
                self.converter = None

    def _release_camera_locked(self):
        """在持有 cam_lock 时释放相机资源（内部使用）"""
        # 注意：调用stream_off、close_device 都必须在 cam_lock 下以避免 get_image 与释放并发
        try:
            if self.cam:
                try:
                    # 先停止流（若支持）
                    try:
                        self.cam.stream_off()
                    except Exception:
                        pass
                    # 再关闭设备
                    try:
                        self.cam.close_device()
                    except Exception:
                        pass
                finally:
                    self.cam = None
            self.converter = None
        except Exception as e:
            print(f"[警告] 相机释放失败 (locked): {e}")

    def close(self):
        """外部调用的关闭，停止线程并释放资源"""
        self.running = False
        # 等待线程退出
        if self.thread and self.thread_started:
            self.thread.join(timeout=1.0)
        # 释放相机
        with self.cam_lock:
            self._release_camera_locked()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def _signal_handler(self, sig, frame):
        print("\n[信息] 收到退出信号，正在释放资源...")
        try:
            self.close()
        except Exception:
            pass
        sys.exit(0)

    # ---------- 帧抓取与转换 ----------
    def _convert_to_numpy(self, raw_image):
        """把 raw_image 转换为 numpy RGB 图像。假定 raw_image 是有效帧"""
        # 使用已缓存的 converter（如果存在），否则临时创建
        converter = self.converter
        created_temp_conv = False
        if converter is None:
            if not self.cam:
                raise Exception("Converter is None and cam is closed, cannot create new converter.")
            converter = self.device_manager.create_image_format_convert()
            converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)
            created_temp_conv = True

        # 针对像素格式设置有效位
        try:
            pixel_format = raw_image.get_pixel_format()
            bit_map = {
                gx.GxPixelFormatEntry.BAYER_RG12: gx.DxValidBit.BIT4_11,
                gx.GxPixelFormatEntry.BAYER_RG10: gx.DxValidBit.BIT2_9,
                gx.GxPixelFormatEntry.BAYER_RG8:  gx.DxValidBit.BIT0_7,
            }
            if pixel_format in bit_map:
                converter.set_valid_bits(bit_map[pixel_format])
            else:
                converter.set_valid_bits(gx.DxValidBit.BIT0_7)

            buffer_size = converter.get_buffer_size_for_conversion(raw_image)
            buffer_array = (c_ubyte * buffer_size)()
            buffer_ptr = addressof(buffer_array)
            converter.convert(raw_image, buffer_ptr, buffer_size, False)

            # 注意 frame_data 的字段名（和你之前的写法一致）
            h = raw_image.frame_data.height
            w = raw_image.frame_data.width
            img_np = np.frombuffer(buffer_array, dtype=np.uint8).reshape(h, w, 3)
            # 若创建了临时 converter，手动释放/忽略（GC 处理）
            return img_np
        finally:
            if created_temp_conv:
                # nothing to explicitly free in python wrapper, just drop ref
                pass

    # ---------- 公开接口：启动/停止采集线程 ----------
    def start(self):
        """启动后台采集线程（可多次调用但只会真正启动一次）"""
        if self.thread_started:
            return
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.thread_started = True

    def _capture_loop(self):
        """仅由后台线程执行：抓取、转换、缩放并缓存。"""
        SHORT_RETRY = 3
        while self.running:
            # 如果 cam 尚未就绪，等待并重试
            with self.cam_lock:
                cam_local = self.cam  # 复制引用以减少锁保持时间
                conv_local = self.converter
            if cam_local is None or conv_local is None:
                time.sleep(0.1)
                continue

            # 尝试一次获取帧（带 timeout），失败则短重试几次，再触发重连
            got_frame = False
            for attempt in range(SHORT_RETRY):
                try:
                    with self.cam_lock:
                        # 每次从 SDK 取帧都在 cam_lock 下
                        if not self.cam:
                            raise Exception("相机已关闭")
                        raw_image = self.cam.data_stream[0].get_image(timeout=1000)
                        if raw_image is None:
                            raise Exception("未获取到图像（raw_image is None）")
                        if raw_image.get_status() != gx.GxFrameStatusList.SUCCESS:
                            raise Exception("图像状态无效（Incomplete frame）")

                        frame = self._convert_to_numpy(raw_image)

                    # 缩放和绘制参考线放到 SDK 锁外，减少相机阻塞时间。
                    resized_frame = self._resize_and_pad(frame)
                    with self.frame_lock:
                        self.latest_frame = resized_frame
                    got_frame = True
                    break
                except Exception as e:
                    if not self.running:
                        break
                    # 记录并短等待后重试
                    print(f"[警告] 第 {attempt+1} 次取帧失败: {e}")
                    time.sleep(0.1)

            if not self.running:
                break

            if not got_frame:
                # 短重试都失败 → 安全重连（在 _safe_reconnect 中也会获取 cam_lock）
                print("[警告] 多次短重试取帧失败，尝试安全重连")
                self._safe_reconnect()
                # 重连后短暂等待
                time.sleep(0.1)
            else:
                # 成功获取帧后按目标显示速率稍作 sleep（避免 100% 占用）
                time.sleep(0.01)

    def get_latest_frame(self):
        """返回最近一帧（由后台线程填充）。可能为 None（尚未有帧）。"""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def cv_show_image(self):
        """显示 OpenCV 窗口，同时保持后台采集线程更新缓存。"""
        self.start()
        print("[信息] 相机后台采集线程已启动...")
        while self.running:
            frame = self.get_latest_frame()
            if frame is not None:
                cv2.imshow("Daheng Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.close()
                break
            time.sleep(0.001)

    def cv_get_image(self):
        """外部需要单帧时从缓存中非阻塞读取（不直接访问 SDK）。"""
        return self.get_latest_frame()

    def _safe_reconnect(self):
        """安全的重连流程：在 cam_lock 下释放并重新初始化"""
        with self.cam_lock:
            # 释放（内部会 stream_off + close）
            try:
                if self.cam:
                    try:
                        self.cam.stream_off()
                    except Exception:
                        pass
                    try:
                        self.cam.close_device()
                    except Exception:
                        pass
                self.cam = None
            except Exception as e:
                print(f"[警告] _safe_reconnect release 部分异常: {e}")

        # 给驱动时间释放资源
        time.sleep(0.2)

        # 重新初始化（内部再次获取 cam_lock）
        try:
            self._reinitialize_camera()
        except Exception as e:
            print(f"[错误] _safe_reconnect 重新初始化异常: {e}")
    
    def _resize_and_pad(self, img):
        h, w = img.shape[:2]
        target_w, target_h = self.display_size

        scale = min(target_w / w, target_h / h)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        top = (target_h - resized.shape[0]) // 2
        bottom = target_h - resized.shape[0] - top
        left = (target_w - resized.shape[1]) // 2
        right = target_w - resized.shape[1] - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        padded = self._draw_l_shape(padded, style="dashed")

        return padded
    
    # def _draw_l_shape(self, img):
    #     """在图像上绘制六个点组成的L型封闭图案"""
    #     h, w = img.shape[:2]

    #     # L型参数（相对比例）
    #     margin = int(min(w, h) * 0.2)
    #     spacing = int(min(w, h) * 0.25)
    #     radius = 2
    #     color = (0, 0, 255)  # 绿色点
    #     thickness = -1

    #     # 从左上角开始绘制 L 形（共6个点）
    #     # 例如：
    #     # (0,0) (1,0) (2,0)
    #     # (0,1)
    #     # (0,2)
    #     # (0,3)
    #     pts = [
    #         (360+margin, margin),
    #         (360+margin + spacing, margin),
    #         (360+margin + spacing, margin + 3 * spacing-45),
    #         (360+margin - spacing+20, margin + 3 * spacing-45),
    #         (360+margin - spacing+20, margin + 2 * spacing-35),
    #         (360+margin, margin + 2 * spacing-35),
    #         (360+margin, margin),
    #     ]

    #     # 绘制点
    #     for (x, y) in pts:
    #         cv2.circle(img, (x, y), radius, color, thickness)

    #     # 封闭L形边框（可选：连线）
    #     for i in range(len(pts) - 1):
    #         cv2.line(img, pts[i], pts[i + 1], (0, 0, 255), 3)

    #     return img

    def _draw_l_shape(self, img, style="dashed"):
        """
        在图像上绘制L型封闭图案
        :param style: "solid" 实线 | "dashed" 虚线
        """
        h, w = img.shape[:2]
        margin = int(min(w, h) * 0.2)
        spacing = int(min(w, h) * 0.25)
        color = (0, 0, 255)
        thickness = 2

        pts = [
            (360 + margin, margin),
            (360 + margin + spacing, margin),
            (360 + margin + spacing, margin + 3 * spacing - 45),
            (360 + margin - spacing + 20, margin + 3 * spacing - 45),
            (360 + margin - spacing + 20, margin + 2 * spacing - 35),
            (360 + margin, margin + 2 * spacing - 35),
            (360 + margin, margin),
        ]

        # 画点
        for (x, y) in pts:
            cv2.circle(img, (x, y), 2, color, -1)

        # 根据 style 绘制不同线条
        if style == "solid":
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 3)
        else:  # dashed
            for i in range(len(pts) - 1):
                self._draw_dashed_line(img, pts[i], pts[i + 1], color, thickness, 10, 6)

        return img

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
        """绘制虚线（由短线+间隔组成）"""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = int(np.hypot(x2 - x1, y2 - y1))
        # 沿线段方向逐段绘制
        for i in range(0, dist, dash_length + gap_length):
            start_ratio = i / dist
            end_ratio = min((i + dash_length) / dist, 1.0)
            xs = int(x1 + (x2 - x1) * start_ratio)
            ys = int(y1 + (y2 - y1) * start_ratio)
            xe = int(x1 + (x2 - x1) * end_ratio)
            ye = int(y1 + (y2 - y1) * end_ratio)
            cv2.line(img, (xs, ys), (xe, ye), color, thickness)
