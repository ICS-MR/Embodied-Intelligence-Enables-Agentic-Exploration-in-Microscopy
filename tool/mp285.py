from __future__ import annotations

import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable


from tool.base import BaseTool, tool_func

try:
    import serial
except Exception:  # pragma: no cover
    serial = None


CR = b"\r"
DEFAULT_MICROSTEPS_PER_UM = 25.0
DEFAULT_UM_PER_MICROSTEP = 0.04
PUMP_LINE_ENDING = "\r\n"
PUMP_STEPS_PER_UL = 100.0
PUMP_SPEED_STEPS_PER_UL_PER_S = 100.0


class MP285Error(RuntimeError):
    """Base error for MP-285 tool failures."""


class MP285ConnectionError(MP285Error):
    """Raised when the MP-285 serial connection is unavailable."""


class MP285ProtocolError(MP285Error):
    """Raised when the MP-285 controller returns unexpected data."""


class MP285Tool(BaseTool):
    """Micromanipulator system tool for MP-285A motion and pump control."""

    planning_hint = (
    """Controls the micromanipulator system. To capture/grasp a target, the microscope stage first moves to align the target directly below the needle (at the field center); the needle then moves along the Z axis to the working height and the pump aspirates to complete the grasp; release is the reverse (dispense at the working height). After completing aspiration/dispensing at the working height, lift the robotic arm Z axis back to the safe height before the next stage movement. The safe height and working height are given by the task instruction. When the task does not specify an aspiration/dispensing volume, the tool defaults to 80 µL at 20 µL/s. Unless explicitly stated otherwise, all XY positioning is performed by moving the microscope stage, not the needle. The needle moves mainly along the Z axis (safe height / working height). At system startup, the needle is in its initial state: initial Z is 1400, and X/Y is at the field center."""
    )

    execution_hint = (
    """Establish the connection before any motion. The needle moves mainly in Z (safe height / working height), and the pump performs aspiration/dispensing. When the task does not specify a volume, aspiration/dispensing defaults to 80 µL at 20 µL/s, and say() records that the default parameters were used."""
    )

    def __init__(
        self,
        storage_manager=None,
        output_dir: str = "./output",
        *,
        port: str = "COM3",
        baudrate: int = 9600,
        timeout: float = 0.5,
        write_timeout: float = 0.5,
        intercommand_delay_s: float = 0.002,
        rtscts: bool = True,
        serial_factory: Callable[..., Any] | None = None,
        pump_port: str = "COM6",
        pump_baudrate: int = 115200,
        pump_timeout: float = 0.5,
        pump_write_timeout: float = 0.5,
        pump_motor_num: int = 2,
        pump_default_speed: int = 15000,
    ) -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.write_timeout = float(write_timeout)
        self.intercommand_delay_s = float(intercommand_delay_s)
        self.rtscts = bool(rtscts)
        self._serial_factory = serial_factory
        self._serial = None
        self.pump_port = pump_port
        self.pump_baudrate = int(pump_baudrate)
        self.pump_timeout = float(pump_timeout)
        self.pump_write_timeout = float(pump_write_timeout)
        self.pump_motor_num = int(pump_motor_num)
        self.pump_default_speed = int(pump_default_speed)
        self._pump_speed = self.pump_default_speed
        self._pump_serial = None
        self._lock = threading.RLock()
        self._microsteps_per_um = DEFAULT_MICROSTEPS_PER_UM
        self._um_per_microstep = DEFAULT_UM_PER_MICROSTEP
        self._speed_um_s = 1310

    @tool_func
    def connect(self) -> None:
        """Connect to both the MP-285 micromanipulator and the pump using default serial settings."""
        with self._lock:
            self._ensure_connected()
            self._ensure_pump_connected()

    @tool_func
    def get_micromanipulator_position(self) -> dict:
        """
        Get the current absolute XYZ position of the micromanipulator needle (not the stage).

        x, y, z are absolute needle coordinates (microns); X/Y are normally at 0
        (field center), and Z is the needle operating height (microns), distinct from
        the microscope focus Z.

        Returns:
            Dictionary containing the current needle position in microns:
            - 'x': Absolute X position (microns)
            - 'y': Absolute Y position (microns)
            - 'z': Z operating height (microns)
        """
        with self._lock:
            self._ensure_connected()
            self._refresh_status()
            x_us, y_us, z_us = self._query_position_microsteps()
            return self._position_to_microns(x_us, y_us, z_us)

    @tool_func
    def micromanipulator_move(self, x: int, y: int, z: int) -> None:
        """
        Set the XYZ coordinate position of the robotic arm.

        Args:
            x: Absolute X position of the needle (microns); normally 0
            y: Absolute Y position of the needle (microns); normally 0
            z: Robotic-arm Z axis, needle operating height (microns)
        """
        target_us = self._to_microsteps(int(x), int(y), int(z))
        with self._lock:
            self._ensure_connected()
            self._refresh_status()
            current_us = self._query_position_microsteps()
            timeout = self._estimate_move_timeout(current_us, target_us)
            self._simple_command(b"a\r")
            payload = b"m" + struct.pack("<iii", *target_us) + CR
            self._simple_command(payload, timeout=timeout)

    @tool_func
    def pump_set_velocity(self, velocity: float) -> None:
        """
        Set the fluid handling speed of the pump.

        Args:
            velocity: Flow rate in microliters per second (μL/s)
        """
        with self._lock:
            self._ensure_pump_connected()
            self._pump_speed = self._bounded_pump_speed(self._velocity_to_pump_speed(float(velocity)))
            self._pump_set_speed(self._pump_speed)

    @tool_func
    def pump_in(self, volume: float) -> None:
        """
        Perform fluid aspiration operation.

        Args:
            volume: Volume to aspirate in microliters (μL)
        """
        self._pump_run_relative(self._volume_to_pump_steps(float(volume)))

    @tool_func
    def pump_out(self, volume: float) -> None:
        """
        Perform fluid dispensing operation.

        Args:
            volume: Volume to dispense in microliters (μL)
        """
        self._pump_run_relative(-self._volume_to_pump_steps(float(volume)))

    # MP-285 serial helpers

    def _ensure_connected(self) -> None:
        self._serial = self._open_serial(
            current=self._serial,
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
            rtscts=self.rtscts,
            error_message="pyserial is required for MP-285 USB-VCP control.",
        )

    def _sleep_intercommand(self) -> None:
        if self.intercommand_delay_s > 0:
            time.sleep(self.intercommand_delay_s)

    def _reset_buffers(self, serial_obj: Any) -> None:
        reset_input = getattr(serial_obj, "reset_input_buffer", None)
        if callable(reset_input):
            reset_input()
        reset_output = getattr(serial_obj, "reset_output_buffer", None)
        if callable(reset_output):
            reset_output()

    def _open_serial(
        self,
        *,
        current: Any,
        port: str,
        baudrate: int,
        timeout: float,
        write_timeout: float,
        rtscts: bool,
        error_message: str,
    ) -> Any:
        if current is not None and bool(getattr(current, "is_open", True)):
            return current

        if self._serial_factory is not None:
            factory = self._serial_factory
        else:
            if serial is None:
                raise MP285ConnectionError(error_message)
            factory = serial.Serial

        serial_obj = factory(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=timeout,
            write_timeout=write_timeout,
            rtscts=rtscts,
            xonxoff=False,
            dsrdtr=False,
        )
        self._reset_buffers(serial_obj)
        self._sleep_intercommand()
        return serial_obj

    def _require_serial(self):
        if self._serial is None or not bool(getattr(self._serial, "is_open", True)):
            raise MP285ConnectionError("MP-285 is not connected.")
        return self._serial

    def _clear_stale_input(self) -> None:
        serial_obj = self._require_serial()
        waiting = int(getattr(serial_obj, "in_waiting", 0) or 0)
        if waiting > 0:
            reset_input = getattr(serial_obj, "reset_input_buffer", None)
            if callable(reset_input):
                reset_input()

    def _read_exact(self, length: int, *, timeout: float) -> bytes:
        serial_obj = self._require_serial()
        deadline = time.monotonic() + max(float(timeout), 0.01)
        buffer = bytearray()
        while len(buffer) < length:
            chunk = serial_obj.read(length - len(buffer))
            if chunk:
                buffer.extend(chunk)
                continue
            if time.monotonic() >= deadline:
                raise MP285ProtocolError(
                    f"Timed out waiting for {length} bytes from MP-285; received {len(buffer)} bytes."
                )
        return bytes(buffer)

    def _transact(self, payload: bytes, *, expected_length: int, timeout: float | None = None) -> bytes:
        serial_obj = self._require_serial()
        self._clear_stale_input()
        serial_obj.write(payload)
        flush = getattr(serial_obj, "flush", None)
        if callable(flush):
            flush()
        response = self._read_exact(expected_length, timeout=timeout or self.timeout)
        self._sleep_intercommand()
        return response

    def _simple_command(self, payload: bytes, *, timeout: float | None = None) -> None:
        response = self._transact(payload, expected_length=1, timeout=timeout)
        if response != CR:
            raise MP285ProtocolError(f"Expected task-complete CR from MP-285, received {response!r}")

    def _query_position_microsteps(self) -> tuple[int, int, int]:
        response = self._transact(b"c\r", expected_length=13, timeout=max(self.timeout, 1.0))
        if response[-1:] != CR:
            raise MP285ProtocolError(f"Expected CR terminator for position response, received {response[-1:]!r}")
        return struct.unpack("<iii", response[:12])

    def _refresh_status(self) -> None:
        response = self._transact(b"s\r", expected_length=33, timeout=max(self.timeout, 1.0))
        if response[-1:] != CR:
            raise MP285ProtocolError(f"Expected CR terminator for status response, received {response[-1:]!r}")
        block = response[:32]
        step_div = struct.unpack_from("<H", block, 24)[0]
        step_mul = struct.unpack_from("<H", block, 26)[0]
        xspeed_word = struct.unpack_from("<H", block, 28)[0]
        if step_mul > 0:
            self._um_per_microstep = step_mul / 10000.0
            self._microsteps_per_um = 1.0 / self._um_per_microstep
        elif step_div > 0:
            self._microsteps_per_um = float(step_div)
            self._um_per_microstep = 1.0 / self._microsteps_per_um
        self._speed_um_s = max(1, xspeed_word & 0x7FFF)

    def _estimate_move_timeout(self, current_us: tuple[int, int, int], target_us: tuple[int, int, int]) -> float:
        deltas_um = [
            abs(target - current) * self._um_per_microstep
            for current, target in zip(current_us, target_us)
        ]
        max_distance_um = max(deltas_um, default=0.0)
        return max(5.0, max_distance_um / max(float(self._speed_um_s), 1.0) + 2.0)

    def _to_microsteps(self, x_um: int, y_um: int, z_um: int) -> tuple[int, int, int]:
        return (
            int(round(x_um * self._microsteps_per_um)),
            int(round(y_um * self._microsteps_per_um)),
            int(round(z_um * self._microsteps_per_um)),
        )

    def _position_to_microns(self, x_us: int, y_us: int, z_us: int) -> dict:
        return {
            "x": float(x_us) * self._um_per_microstep,
            "y": float(y_us) * self._um_per_microstep,
            "z": float(z_us) * self._um_per_microstep,
        }

    # Pump serial helpers

    def _ensure_pump_connected(self) -> None:
        self._pump_serial = self._open_serial(
            current=self._pump_serial,
            port=self.pump_port,
            baudrate=self.pump_baudrate,
            timeout=self.pump_timeout,
            write_timeout=self.pump_write_timeout,
            rtscts=True,
            error_message="pyserial is required for pump USB-serial control.",
        )

    def _require_pump_serial(self):
        if self._pump_serial is None or not bool(getattr(self._pump_serial, "is_open", True)):
            raise MP285ConnectionError("Pump is not connected.")
        return self._pump_serial

    def _pump_write(self, command: str) -> None:
        serial_obj = self._require_pump_serial()
        payload = (command + PUMP_LINE_ENDING).encode("ascii")
        serial_obj.write(payload)
        flush = getattr(serial_obj, "flush", None)
        if callable(flush):
            flush()
        self._sleep_intercommand()

    def _pump_set_speed(self, speed: int) -> None:
        self._pump_write(f"@199:Motor:Speed:M{self.pump_motor_num} {self._bounded_pump_speed(speed)}")

    def _bounded_pump_speed(self, speed: int) -> int:
        return max(23, min(abs(int(speed)), 15000))

    def _pump_run_relative(self, steps: int) -> None:
        with self._lock:
            self._ensure_pump_connected()
            self._pump_set_speed(self._pump_speed)
            self._pump_write(f"@199:Motor:Run:M{self.pump_motor_num} {int(steps)}")

    def _volume_to_pump_steps(self, volume_ul: float) -> int:
        return int(round(max(volume_ul, 0.0) * PUMP_STEPS_PER_UL))

    def _velocity_to_pump_speed(self, velocity_ul_s: float) -> int:
        return int(round(max(velocity_ul_s, 0.0) * PUMP_SPEED_STEPS_PER_UL_PER_S))
