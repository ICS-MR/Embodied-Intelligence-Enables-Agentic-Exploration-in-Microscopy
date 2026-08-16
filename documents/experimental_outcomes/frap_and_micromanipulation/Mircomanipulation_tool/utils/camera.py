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

        # Synchronization and runtime state.
        self.cam_lock = Lock()           # Protect all access to self.cam.
        self.frame_lock = Lock()
        self.latest_frame = None
        self.thread = None
        self.thread_started = False     # Prevent duplicate start() calls.
        self.converter = None           # Reuse the converter after initialization.

        # Attempt initialization once; failures leave self.cam as None.
        self._reinitialize_camera()

        # Handle Ctrl+C when constructed in the main thread.
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError:
            print("Warning: Failed to register signal handler (not in main thread). Ignoring.")

    # Camera initialization and shutdown
    def _reinitialize_camera(self):
        """Release any old device, reopen the camera, and start streaming."""
        with self.cam_lock:
            self._release_camera_locked()

            try:
                dev_num, dev_info_list = self.device_manager.update_all_device_list()
                if dev_num == 0:
                    raise Exception("No camera device found")

                sn = dev_info_list[0].get("sn")

                # open device
                try:
                    self.cam = self.device_manager.open_device_by_sn(sn)
                except Exception as e:
                    # Fall back to index-based opening when the SDK reports a duplicate open.
                    if "already been opened" in str(e) or "repeat open" in str(e).lower():
                        print("[WARNING] Camera is already open by serial number; trying device index")
                        try:
                            self.cam = self.device_manager.open_device_by_index(0)
                        except Exception as e2:
                            raise e2
                    else:
                        raise e

                # Configure dimensions after opening the device.
                try:
                    feature = self.cam.get_remote_device_feature_control()
                    if feature.is_writable("Width"):
                        feature.get_int_feature("Width").set(self.width)
                    if feature.is_writable("Height"):
                        feature.get_int_feature("Height").set(self.height)
                except Exception as e:
                    print(f"[WARNING] Unable to set camera width and height: {e}")

                # Create and cache an image converter when supported.
                try:
                    self.converter = self.device_manager.create_image_format_convert()
                    self.converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)
                except Exception:
                    self.converter = None

                # Start the camera stream.
                try:
                    self.cam.stream_on()
                except Exception as e:
                    # Release the device before propagating stream startup failures.
                    self._release_camera_locked()
                    raise e

                print("[INFO] Camera initialized successfully")

            except Exception as e:
                print(f"[ERROR] Camera initialization failed: {e}")
                # Ensure a failed initialization leaves no open camera reference.
                try:
                    self._release_camera_locked()
                except Exception:
                    pass
                self.cam = None
                self.converter = None

    def _release_camera_locked(self):
        """Release camera resources while the caller holds cam_lock."""
        # stream_off and close_device must not race with frame acquisition.
        try:
            if self.cam:
                try:
                    # Stop streaming before closing the device.
                    try:
                        self.cam.stream_off()
                    except Exception:
                        pass
                    try:
                        self.cam.close_device()
                    except Exception:
                        pass
                finally:
                    self.cam = None
            self.converter = None
        except Exception as e:
            print(f"[WARNING] Camera release failed while locked: {e}")

    def close(self):
        """Stop the capture thread and release camera resources."""
        self.running = False
        # Wait briefly for the capture thread to exit.
        if self.thread and self.thread_started:
            self.thread.join(timeout=1.0)
        # Release the camera under its lock.
        with self.cam_lock:
            self._release_camera_locked()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def _signal_handler(self, sig, frame):
        print("\n[INFO] Exit signal received; releasing camera resources...")
        try:
            self.close()
        except Exception:
            pass
        sys.exit(0)

    # Frame acquisition and conversion
    def _convert_to_numpy(self, raw_image):
        """Convert a valid raw camera frame to a NumPy RGB image."""
        # Use the cached converter or create a temporary one.
        converter = self.converter
        created_temp_conv = False
        if converter is None:
            if not self.cam:
                raise Exception("Converter is None and cam is closed, cannot create new converter.")
            converter = self.device_manager.create_image_format_convert()
            converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)
            created_temp_conv = True

        # Select valid bits for the source pixel format.
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

            # Read dimensions from the SDK frame metadata.
            h = raw_image.frame_data.height
            w = raw_image.frame_data.width
            img_np = np.frombuffer(buffer_array, dtype=np.uint8).reshape(h, w, 3)
            # Temporary converter wrappers are released by Python's GC.
            return img_np
        finally:
            if created_temp_conv:
                # nothing to explicitly free in python wrapper, just drop ref
                pass

    # Public capture-thread interface
    def start(self):
        """Start the background capture thread once."""
        if self.thread_started:
            return
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.thread_started = True

    def _capture_loop(self):
        """Capture, convert, resize, and cache frames in the background."""
        SHORT_RETRY = 3
        while self.running:
            # Wait until both the camera and converter are ready.
            with self.cam_lock:
                cam_local = self.cam
                conv_local = self.converter
            if cam_local is None or conv_local is None:
                time.sleep(0.1)
                continue

            # Retry frame acquisition briefly before reconnecting.
            got_frame = False
            for attempt in range(SHORT_RETRY):
                try:
                    with self.cam_lock:
                        # Serialize all SDK frame access with camera teardown.
                        if not self.cam:
                            raise Exception("Camera is closed")
                        raw_image = self.cam.data_stream[0].get_image(timeout=1000)
                        if raw_image is None:
                            raise Exception("Camera returned no image")
                        if raw_image.get_status() != gx.GxFrameStatusList.SUCCESS:
                            raise Exception("Camera returned an incomplete frame")

                        frame = self._convert_to_numpy(raw_image)

                    # Resize and draw guides outside the SDK lock.
                    resized_frame = self._resize_and_pad(frame)
                    with self.frame_lock:
                        self.latest_frame = resized_frame
                    got_frame = True
                    break
                except Exception as e:
                    if not self.running:
                        break
                    print(f"[WARNING] Frame acquisition attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1)

            if not self.running:
                break

            if not got_frame:
                print("[WARNING] Frame retries exhausted; attempting a safe reconnect")
                self._safe_reconnect()
                # Give the camera a moment after reconnecting.
                time.sleep(0.1)
            else:
                # Avoid a busy loop after successful acquisition.
                time.sleep(0.01)

    def get_latest_frame(self):
        """Return a copy of the latest frame, or None before the first frame."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def cv_show_image(self):
        """Display cached frames in an OpenCV window."""
        self.start()
        print("[INFO] Background camera capture started...")
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
        """Read one cached frame without accessing the SDK directly."""
        return self.get_latest_frame()

    def _safe_reconnect(self):
        """Release the camera under lock and initialize it again."""
        with self.cam_lock:
            # Stop and close the existing device.
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
                print(f"[WARNING] Camera release during reconnect failed: {e}")

        # Give the driver time to release resources.
        time.sleep(0.2)

        # Reinitialize; the helper acquires cam_lock again.
        try:
            self._reinitialize_camera()
        except Exception as e:
            print(f"[ERROR] Camera reinitialization failed: {e}")
    
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
    
    def _draw_l_shape(self, img, style="dashed"):
        """Draw a closed L-shaped guide using solid or dashed lines."""
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

        # Draw guide vertices.
        for (x, y) in pts:
            cv2.circle(img, (x, y), 2, color, -1)

        # Connect vertices using the selected line style.
        if style == "solid":
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 3)
        else:  # dashed
            for i in range(len(pts) - 1):
                self._draw_dashed_line(img, pts[i], pts[i + 1], color, thickness, 10, 6)

        return img

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
        """Draw a dashed line as alternating segments and gaps."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = int(np.hypot(x2 - x1, y2 - y1))
        # Draw consecutive segments along the line direction.
        for i in range(0, dist, dash_length + gap_length):
            start_ratio = i / dist
            end_ratio = min((i + dash_length) / dist, 1.0)
            xs = int(x1 + (x2 - x1) * start_ratio)
            ys = int(y1 + (y2 - y1) * start_ratio)
            xe = int(x1 + (x2 - x1) * end_ratio)
            ye = int(y1 + (y2 - y1) * end_ratio)
            cv2.line(img, (xs, ys), (xe, ye), color, thickness)
