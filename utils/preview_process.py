from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from importlib import metadata
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from queue import Empty, Full
from typing import Callable, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)

def probe_pyqt_preview_support() -> tuple[bool, str]:
    try:
        from PyQt6 import QtCore  # noqa: F401
        from PyQt6 import QtGui  # noqa: F401
        from PyQt6 import QtWidgets  # noqa: F401
    except Exception as exc:
        return (
            False,
            "PyQt6 preview backend is unavailable. "
            f"Install PyQt6 and its Qt runtime to enable the local preview window. error={exc}",
        )

    try:
        version = metadata.version("PyQt6")
    except Exception:
        version = "unknown"

    return True, f"PyQt6 preview backend detected: PyQt6=={version}"


def _normalize_frame(frame: object) -> Optional[np.ndarray]:
    if frame is None:
        return None

    array = np.asarray(frame)
    if array.size == 0:
        return None

    if array.ndim == 2:
        if array.dtype != np.uint8:
            array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

    if array.ndim == 3:
        if array.shape[2] == 1:
            base = array[:, :, 0]
            if base.dtype != np.uint8:
                base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        color = array[:, :, :3]
        if color.dtype != np.uint8:
            color = cv2.normalize(color, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return color

    return None


def _safe_put_status(status_queue: MPQueue, kind: str, message: str) -> None:
    try:
        status_queue.put_nowait({"kind": kind, "message": message})
    except Exception:
        pass


def _preview_worker(frame_queue: MPQueue, status_queue: MPQueue, stop_event: MPEvent, window_name: str) -> None:
    try:
        _run_pyqt_preview(frame_queue, status_queue, stop_event, window_name)
    except Exception as exc:
        _safe_put_status(
            status_queue,
            "pyqt_window_error",
            f"PyQt6 preview window is unavailable in child process: {exc}",
        )


def _run_pyqt_preview(frame_queue: MPQueue, status_queue: MPQueue, stop_event: MPEvent, window_name: str) -> None:
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    fallback_frame = np.zeros((720, 720, 3), dtype=np.uint8)

    class PreviewWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self._current_qimage: Optional[QImage] = None
            self._label = QLabel(self)
            self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._label.setMinimumSize(320, 240)

            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._label)
            self.setLayout(layout)
            self.setWindowTitle(window_name)
            self.resize(820, 820)

            self._frame_timer = QTimer(self)
            self._frame_timer.timeout.connect(self._poll_frame_queue)
            self._frame_timer.start(30)

            self._stop_timer = QTimer(self)
            self._stop_timer.timeout.connect(self._check_stop_requested)
            self._stop_timer.start(50)

            self._set_frame(fallback_frame)

        def _check_stop_requested(self) -> None:
            if stop_event.is_set():
                self.close()

        def _poll_frame_queue(self) -> None:
            latest_frame: Optional[np.ndarray] = None

            while True:
                try:
                    incoming = frame_queue.get_nowait()
                except Empty:
                    break
                except (BrokenPipeError, EOFError, OSError, ValueError):
                    stop_event.set()
                    self.close()
                    return

                if incoming is None:
                    if stop_event.is_set():
                        self.close()
                        return
                    continue

                normalized = _normalize_frame(incoming)
                if normalized is not None:
                    latest_frame = normalized

            if latest_frame is not None:
                self._set_frame(latest_frame)

        def _set_frame(self, frame: np.ndarray) -> None:
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
            height, width = rgb_frame.shape[:2]
            image = QImage(
                rgb_frame.data,
                width,
                height,
                rgb_frame.strides[0],
                QImage.Format.Format_RGB888,
            )
            self._current_qimage = image.copy()
            self._refresh_pixmap()

        def _refresh_pixmap(self) -> None:
            if self._current_qimage is None:
                return

            pixmap = QPixmap.fromImage(self._current_qimage)
            target_size = self._label.size()
            if target_size.width() > 0 and target_size.height() > 0:
                pixmap = pixmap.scaled(
                    target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self._label.setPixmap(pixmap)

        def resizeEvent(self, event) -> None:  # type: ignore[override]
            self._refresh_pixmap()
            super().resizeEvent(event)

        def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
            if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
                self.close()
                return
            super().keyPressEvent(event)

        def closeEvent(self, event) -> None:  # type: ignore[override]
            self._frame_timer.stop()
            self._stop_timer.stop()
            stop_event.set()
            super().closeEvent(event)

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])

    window = PreviewWindow()
    window.show()
    _safe_put_status(status_queue, "preview_info", f"PyQt6 preview window started: {window_name}")

    try:
        app.exec()
    finally:
        stop_event.set()
        if owns_app:
            app.quit()


class PreviewProcessManager:
    def __init__(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        *,
        window_name: str = "micro live",
        relay_interval_sec: float = 0.03,
    ) -> None:
        self._frame_provider = frame_provider
        self._window_name = window_name
        self._relay_interval_sec = relay_interval_sec

        self._ctx: SpawnContext = mp.get_context("spawn")
        self._frame_queue: Optional[MPQueue] = None
        self._status_queue: Optional[MPQueue] = None
        self._stop_event: Optional[MPEvent] = None
        self._process: Optional[BaseProcess] = None

        self._relay_stop = threading.Event()
        self._relay_thread: Optional[threading.Thread] = None
        self._empty_frame_warnings = 0

    def is_running(self) -> bool:
        return bool(self._process and self._process.is_alive())

    def start(self) -> None:
        if self.is_running():
            return

        preview_available, preview_message = probe_pyqt_preview_support()
        if not preview_available:
            raise RuntimeError(preview_message)
        logger.info(preview_message)

        self._frame_queue = self._ctx.Queue(maxsize=1)
        self._status_queue = self._ctx.Queue(maxsize=8)
        self._stop_event = self._ctx.Event()
        self._relay_stop.clear()
        self._empty_frame_warnings = 0

        self._process = self._ctx.Process(
            target=_preview_worker,
            args=(self._frame_queue, self._status_queue, self._stop_event, self._window_name),
            daemon=True,
            name="eims_preview_process",
        )
        self._process.start()

        self._relay_thread = threading.Thread(
            target=self._relay_loop,
            daemon=True,
            name="eims_preview_relay",
        )
        self._relay_thread.start()

    def stop(self) -> None:
        self._relay_stop.set()
        if self._stop_event is not None:
            self._stop_event.set()
        self._try_send_sentinel()

        if self._relay_thread and self._relay_thread.is_alive():
            self._relay_thread.join(timeout=1.5)

        if self._process and self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        if self._frame_queue is not None:
            try:
                self._frame_queue.close()
                self._frame_queue.join_thread()
            except Exception:
                pass
        if self._status_queue is not None:
            try:
                self._status_queue.close()
                self._status_queue.join_thread()
            except Exception:
                pass

        self._relay_thread = None
        self._process = None
        self._frame_queue = None
        self._status_queue = None
        self._stop_event = None

    def _relay_loop(self) -> None:
        while not self._relay_stop.is_set():
            if self._stop_event is not None and self._stop_event.is_set():
                break
            self._drain_status_queue()

            try:
                frame = self._frame_provider()
            except Exception as exc:
                logger.warning("Preview frame provider failed before GUI render. This usually points to start_preview() or microscope acquisition state: %s", exc)
                frame = None

            normalized = _normalize_frame(frame)
            if normalized is not None:
                self._empty_frame_warnings = 0
                self._push_latest_frame(normalized)
            else:
                self._empty_frame_warnings += 1
                if self._empty_frame_warnings == 20:
                    logger.warning(
                        "Preview frame source is not producing frames. This usually means start_preview() did not start acquisition, "
                        "or the microscope preview thread is not delivering images yet."
                    )

            time.sleep(self._relay_interval_sec)

    def _push_latest_frame(self, frame: np.ndarray) -> None:
        if self._frame_queue is None:
            return

        try:
            self._frame_queue.put_nowait(frame)
            return
        except Full:
            pass
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._relay_stop.set()
            return

        try:
            self._frame_queue.get_nowait()
        except Empty:
            pass
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._relay_stop.set()
            return

        try:
            self._frame_queue.put_nowait(frame)
        except Full:
            pass
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._relay_stop.set()

    def _try_send_sentinel(self) -> None:
        if self._frame_queue is None:
            return
        try:
            self._frame_queue.put_nowait(None)
        except Full:
            try:
                self._frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self._frame_queue.put_nowait(None)
            except Exception:
                pass
        except Exception:
            pass

    def _drain_status_queue(self) -> None:
        if self._status_queue is None:
            return

        while True:
            try:
                status = self._status_queue.get_nowait()
            except Empty:
                return
            except Exception:
                return

            kind = str(status.get("kind", "preview_error"))
            message = str(status.get("message", "Preview process reported an unknown error"))
            if kind == "preview_info":
                logger.info("%s: %s", kind, message)
            else:
                logger.warning("%s: %s", kind, message)
