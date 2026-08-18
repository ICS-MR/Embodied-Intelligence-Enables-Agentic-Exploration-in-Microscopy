from __future__ import annotations

from .common import *


class MP285Tool(BaseTool):
    """Simulation implementation of the public MP-285 micromanipulator + pump API."""

    _active_instance: "MP285Tool | None" = None

    planning_hint = (
        "Use for direct MP-285A USB-VCP manipulator control on COM3. Positions are in microns "
        "and the tool manages the underlying microstep conversion."
    )
    execution_hint = (
        "Connect before motion, prefer absolute moves, and reconfigure soft limits if the origin is changed."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._connected = False
        self._position = {"x": 0, "y": 0, "z": 0}
        self._velocity_ul_s = 0.0
        self._aspirated_ul = 0.0
        self._dispensed_ul = 0.0
        self._move_history = []
        type(self)._active_instance = self

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MP-285 is not connected; call connect() first")

    @tool_func
    def connect(self) -> None:
        """
        Connect to both the MP-285 micromanipulator and the pump using default serial settings.

        Simulation: marks the tool as connected without any hardware.
        """
        self._connected = True

    @tool_func
    def get_micromanipulator_position(self) -> dict:
        """
        Get current XYZ position of the micromanipulator.

        Returns:
            Dictionary containing the current position in microns:
            - 'x': X-axis position (microns)
            - 'y': Y-axis position (microns)
            - 'z': Z-axis position (microns)
        """
        self._require_connected()
        return dict(self._position)

    @tool_func
    def micromanipulator_move(self, x: int, y: int, z: int) -> None:
        """
        Set the XYZ coordinate position of the microscope stage (absolute move, microns).

        Simulation: updates the in-memory position; no hardware motion.
        """
        self._require_connected()
        self._position = {"x": int(x), "y": int(y), "z": int(z)}
        self._move_history.append((int(x), int(y), int(z)))

    @tool_func
    def pump_set_velocity(self, velocity: float) -> None:
        """
        Set the fluid handling speed of the pump.

        Args:
            velocity: Flow rate in microliters per second (uL/s).
        """
        self._require_connected()
        self._velocity_ul_s = float(velocity)

    @tool_func
    def pump_in(self, volume: float) -> None:
        """
        Perform fluid aspiration operation.

        Args:
            volume: Volume to aspirate in microliters (uL).
        """
        self._require_connected()
        self._aspirated_ul += float(volume)

    @tool_func
    def pump_out(self, volume: float) -> None:
        """
        Perform fluid dispensing operation.

        Args:
            volume: Volume to dispense in microliters (uL).
        """
        self._require_connected()
        self._dispensed_ul += float(volume)

    @tool_func
    def cell_detection(self) -> dict:
        """
        Detect and return the position of the target cell.

        Returns:
            Dictionary with 'x' and 'y' (microns). Simulation: returns the current XY position.
        """
        self._require_connected()
        return {"x": self._position["x"], "y": self._position["y"]}
