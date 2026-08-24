from __future__ import annotations


from .common import *

class MP285Tool(BaseTool):
    """Simulation implementation of the public MP-285 micromanipulator + pump API."""

    _active_instance: "MP285Tool | None" = None

    planning_hint = (
    """Controls the micromanipulator system. To capture/grasp a target, the microscope stage first moves to align the target directly below the needle (at the field center); the needle then moves along the Z axis to the working height and the pump aspirates to complete the grasp; release is the reverse (dispense at the working height). After completing aspiration/dispensing at the working height, lift the robotic arm Z axis back to the safe height before the next stage movement. The safe height and working height are given by the task instruction. When the task does not specify an aspiration/dispensing volume, the tool defaults to 80 µL at 20 µL/s. Unless explicitly stated otherwise, all XY positioning is performed by moving the microscope stage, not the needle. The needle moves mainly along the Z axis (safe height / working height). At system startup, the needle is in its initial state: the initial Z-axis is at the safe height, and the X/Y coordinates are at the center of the field of view."""
    )

    execution_hint = (
    """Establish the connection before any motion. The needle moves mainly in Z (safe height / working height), and the pump performs aspiration/dispensing. When the task does not specify a volume, aspiration/dispensing defaults to 80 µL at 20 µL/s, and say() records that the default parameters were used."""
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._connected = False
        self._position = {"x": 0, "y": 0, "z": 1400}
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

        """
        self._connected = True

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
        self._require_connected()
        return dict(self._position)

    @tool_func
    def micromanipulator_move(self, x: int, y: int, z: int) -> None:
        """
        Set the XYZ coordinate position of the robotic arm.

        Args:
            x: Absolute X position of the needle (microns); normally 0
            y: Absolute Y position of the needle (microns); normally 0
            z: Robotic-arm Z axis, needle operating height (microns)
        """
        self._require_connected()
        self._position = {"x": x, "y": y, "z": z}
        self._move_history.append((x, y, z))

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
