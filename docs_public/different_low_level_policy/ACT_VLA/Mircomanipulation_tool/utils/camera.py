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

        # Synchronization and state control.
        self.cam_lock = Lock()           # Protect all access to self.cam.
        self.frame_lock = Lock()
        self.latest_frame = None
        self.thread = None
        self.thread_started = False     # Prevent repeated start() calls.
        self.converter = None           # Reuse the converter created after camera initialization.

        # Attempt initialization once; self.cam is set to None on failure.
        self._reinitialize_camera()

        # Exit safely on Ctrl+C.
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError:
            print("Warning: Failed to register signal handler (not in main thread). Ignoring.")

    # ---------- Camera open/close/initialization ----------
    def _reinitialize_camera(self):
        """Reconnect safely by releasing the old device and restarting the stream."""
        with self.cam_lock:
            self._release_camera_locked()  # This internal method assumes cam_lock is held.

            try:
                dev_num, dev_info_list = self.device_manager.update_all_device_list()
                if dev_num == 0:
                    raise Exception("No camera device found")

                sn = dev_info_list[0].get("sn")

                # open device
                try:
                    self.cam = self.device_manager.open_device_by_sn(sn)
                except Exception as e:
                    # Try an alternative open method based on the exception message.
                    if "already been opened" in str(e) or "repeat open" in str(e).lower():
                        print("[WARNING] open_device_by_sn reported 'already opened'; trying open_device_by_index")
                        try:
                            self.cam = self.device_manager.open_device_by_index(0)
                        except Exception as e2:
                            raise e2
                    else:
                        raise e

                # Configure parameters after opening the device.
                try:
                    feature = self.cam.get_remote_device_feature_control()
                    if feature.is_writable("Width"):
                        feature.get_int_feature("Width").set(self.width)
                    if feature.is_writable("Height"):
                        feature.get_int_feature("Height").set(self.height)
                except Exception as e:
                    print(f"[WARNING] Failed to set Width/Height: {e}")

                # Create and cache the converter if supported by device_manager.
                try:
                    self.converter = self.device_manager.create_image_format_convert()
                    self.converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)
                except Exception:
                    self.converter = None

                # Start the stream.
                try:
                    self.cam.stream_on()
                except Exception as e:
                    # Release resources and re-raise if stream_on fails.
                    self._release_camera_locked()
                    raise e

                print("[INFO] Camera initialized successfully")

            except Exception as e:
                print(f"[ERROR] Camera initialization failed: {e}")
                # Ensure cam is None to indicate that it is not open.
                try:
                    self._release_camera_locked()
                except Exception:
                    pass
                self.cam = None
                self.converter = None

    def _release_camera_locked(self):
        """Release camera resources while holding cam_lock (internal use)."""
        # stream_off and close_device must run under cam_lock to avoid racing with get_image.
        try:
            if self.cam:
                try:
                    # Stop the stream first, if supported.
                    try:
                        self.cam.stream_off()
                    except Exception:
                        pass
                    # Then close the device.
                    try:
                        self.cam.close_device()
                    except Exception:
                        pass
                finally:
                    self.cam = None
            self.converter = None
        except Exception as e:
            print(f"[WARNING] Failed to release camera while locked: {e}")

    def close(self):
        """Stop the thread and release resources for an external close call."""
        self.running = False
        # Wait for the thread to exit.
        if self.thread and self.thread_started:
            self.thread.join(timeout=1.0)
        # Release the camera.
        with self.cam_lock:
            self._release_camera_locked()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def _signal_handler(self, sig, frame):
        print("\n[INFO] Exit signal received; releasing resources...")
        try:
            self.close()
        except Exception:
            pass
        sys.exit(0)

    # ---------- Frame capture and conversion ----------
    def _convert_to_numpy(self, raw_image):
        """Convert raw_image to a NumPy RGB image, assuming a valid frame."""
        # Use the cached converter if available; otherwise create one temporarily.
        converter = self.converter
        created_temp_conv = False
        if converter is None:
            if not self.cam:
                raise Exception("Converter is None and cam is closed, cannot create new converter.")
            converter = self.device_manager.create_image_format_convert()
            converter.set_dest_format(gx.GxPixelFormatEntry.RGB8)
            created_temp_conv = True

        # Set valid bits for the pixel format.
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

            # Use the frame_data field names expected by the existing implementation.
            h = raw_image.frame_data.height
            w = raw_image.frame_data.width
            img_np = np.frombuffer(buffer_array, dtype=np.uint8).reshape(h, w, 3)
            # Drop a temporary converter reference and let garbage collection handle it.
            return img_np
        finally:
            if created_temp_conv:
                # nothing to explicitly free in python wrapper, just drop ref
                pass

    # ---------- Public interface: start/stop the capture thread ----------
    def start(self):
        """Start the background capture thread only once across repeated calls."""
        if self.thread_started:
            return
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.thread_started = True

    def _capture_loop(self):
        """Capture, convert, resize, and cache frames in the background thread."""
        SHORT_RETRY = 3
        while self.running:
            # Wait and retry if the camera is not ready.
            with self.cam_lock:
                cam_local = self.cam  # Copy references to reduce lock hold time.
                conv_local = self.converter
            if cam_local is None or conv_local is None:
                time.sleep(0.1)
                continue

            # Capture with a timeout, retry briefly on failure, then reconnect.
            got_frame = False
            for attempt in range(SHORT_RETRY):
                try:
                    with self.cam_lock:
                        # Retrieve each SDK frame while holding cam_lock.
                        if not self.cam:
                            raise Exception("Camera is closed")
                        raw_image = self.cam.data_stream[0].get_image(timeout=1000)
                        if raw_image is None:
                            raise Exception("No image received (raw_image is None)")
                        if raw_image.get_status() != gx.GxFrameStatusList.SUCCESS:
                            raise Exception("Invalid image status (incomplete frame)")

                        frame = self._convert_to_numpy(raw_image)

                    # Resize and draw guides outside the SDK lock to reduce camera blocking.
                    resized_frame = self._resize_and_pad(frame)
                    with self.frame_lock:
                        self.latest_frame = resized_frame
                    got_frame = True
                    break
                except Exception as e:
                    if not self.running:
                        break
                    # Log the failure, wait briefly, and retry.
                    print(f"[WARNING] Frame capture attempt {attempt+1} failed: {e}")
                    time.sleep(0.1)

            if not self.running:
                break

            if not got_frame:
                # Reconnect safely after all short retries fail; _safe_reconnect also acquires cam_lock.
                print("[WARNING] Repeated frame capture failures; attempting a safe reconnect")
                self._safe_reconnect()
                # Wait briefly after reconnecting.
                time.sleep(0.1)
            else:
                # Sleep briefly after a successful frame to avoid 100% CPU usage.
                time.sleep(0.01)

    def get_latest_frame(self):
        """Return the latest background-captured frame, or None if unavailable."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def cv_show_image(self):
        """Display an OpenCV window while the background thread updates the cache."""
        self.start()
        print("[INFO] Camera background capture thread started...")
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
        """Read one frame from the cache without blocking or accessing the SDK directly."""
        return self.get_latest_frame()

    def _safe_reconnect(self):
        """Reconnect safely by releasing and reinitializing under cam_lock."""
        with self.cam_lock:
            # Release resources with stream_off and close.
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
                print(f"[WARNING] Exception while releasing resources in _safe_reconnect: {e}")

        # Give the driver time to release resources.
        time.sleep(0.2)

        # Reinitialize; the internal method acquires cam_lock again.
        try:
            self._reinitialize_camera()
        except Exception as e:
            print(f"[ERROR] Reinitialization failed in _safe_reconnect: {e}")
    
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
    #     """Draw a closed L-shaped pattern made of six points on the image."""
    #     h, w = img.shape[:2]

    #     # L-shape parameters as relative proportions.
    #     margin = int(min(w, h) * 0.2)
    #     spacing = int(min(w, h) * 0.25)
    #     radius = 2
    #     color = (0, 0, 255)  # Green points.
    #     thickness = -1

    #     # Draw an L shape from the upper-left corner using six points.
    #     # For example:
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

    #     # Draw points.
    #     for (x, y) in pts:
    #         cv2.circle(img, (x, y), radius, color, thickness)

    #     # Close the L-shaped border, optionally by connecting lines.
    #     for i in range(len(pts) - 1):
    #         cv2.line(img, pts[i], pts[i + 1], (0, 0, 255), 3)

    #     return img

    def _draw_l_shape(self, img, style="dashed"):
        """
        Draw a closed L-shaped pattern on the image.
        :param style: "solid" line or "dashed" line
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

        # Draw points.
        for (x, y) in pts:
            cv2.circle(img, (x, y), 2, color, -1)

        # Draw different line types based on style.
        if style == "solid":
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 3)
        else:  # dashed
            for i in range(len(pts) - 1):
                self._draw_dashed_line(img, pts[i], pts[i + 1], color, thickness, 10, 6)

        return img

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
        """Draw a dashed line composed of short segments and gaps."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = int(np.hypot(x2 - x1, y2 - y1))
        # Draw one segment at a time along the line direction.
        for i in range(0, dist, dash_length + gap_length):
            start_ratio = i / dist
            end_ratio = min((i + dash_length) / dist, 1.0)
            xs = int(x1 + (x2 - x1) * start_ratio)
            ys = int(y1 + (y2 - y1) * start_ratio)
            xe = int(x1 + (x2 - x1) * end_ratio)
            ye = int(y1 + (y2 - y1) * end_ratio)
            cv2.line(img, (xs, ys), (xe, ye), color, thickness)
