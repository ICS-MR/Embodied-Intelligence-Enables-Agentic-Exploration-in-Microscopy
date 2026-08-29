import time
from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class TaskProfile:
    backend: str
    control_mode: str
    dichroic: Optional[str] = None
    brightness: Optional[float] = None
    exposure: Optional[float] = None
    z_position: Optional[float] = None
    reset_z_position_on_stop: Optional[float] = None
    xy_position: Optional[Tuple[float, float]] = None
    relative_xy: bool = False
    interval: float = 0.033


def _brightness_microscope_profile(brightness: float) -> TaskProfile:
    return TaskProfile(
        backend="microscope",
        control_mode="brightness",
        dichroic="1-NONE",
        brightness=brightness,
        z_position=6550.0,
        reset_z_position_on_stop=None,
        relative_xy=False,
        interval=0.1,
    )


def _xy_microscope_profile(brightness: float) -> TaskProfile:
    return TaskProfile(
        backend="microscope",
        control_mode="xy",
        dichroic="1-NONE",
        brightness=brightness,
        z_position=6550.0,
        reset_z_position_on_stop=None,
        relative_xy=True,
        interval=0.1,
    )


def _z_microscope_profile(brightness: float) -> TaskProfile:
    return TaskProfile(
        backend="microscope",
        control_mode="z",
        dichroic="1-NONE",
        brightness=brightness,
        z_position=None,
        reset_z_position_on_stop=5000,
        relative_xy=False,
        interval=0.1,
    )


def _funa_microscope_profile(control_mode: str) -> TaskProfile:
    return TaskProfile(
        backend="microscope",
        control_mode=control_mode,
        dichroic="2-U-FUNA",
        brightness=0,
        exposure=30,
        z_position=6550.0 if control_mode in ("brightness", "xy") else None,
        reset_z_position_on_stop=None,
        relative_xy=control_mode == "xy",
        interval=0.1,
    )


SUPPORTED_TASK_PROFILES = {
    "2d_move_none": _xy_microscope_profile(235),
    "2d_set_brightness_none": _brightness_microscope_profile(125),
    "2d_set_z_none": _z_microscope_profile(250),
    "cell_move_none": _xy_microscope_profile(235),
    "cell_set_brightness_none": _brightness_microscope_profile(125),
    "cell_set_z_none": _z_microscope_profile(250),
    "slice_move_none": _xy_microscope_profile(235),
    "slice_set_brightness_none": _brightness_microscope_profile(125),
    "slice_set_z_none": _z_microscope_profile(250),
    "push_to_target": TaskProfile(backend="robot", control_mode="xy", interval=0.033),
    "cell_move_funa": _funa_microscope_profile("xy"),
}


def normalize_task_name(task_name: str) -> str:
    """Normalize task, dataset, and compressed artifact names to one registry key."""
    normalized = task_name.strip().split("/")[-1].split("\\")[-1].lower().replace(" ", "_")
    if normalized.endswith(".zip"):
        normalized = normalized[:-4]
    for prefix in ("dataset_", "task_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    if normalized.endswith("_compressed"):
        normalized = normalized[: -len("_compressed")]
    return normalized


def list_supported_tasks():
    return sorted(SUPPORTED_TASK_PROFILES)


def resolve_task_profile(task_name: str, backend: str = "auto", control_mode: str = "auto") -> TaskProfile:
    normalized = normalize_task_name(task_name)
    if normalized not in SUPPORTED_TASK_PROFILES:
        supported = ", ".join(list_supported_tasks())
        raise ValueError(
            f"Unsupported task '{task_name}' normalized as '{normalized}'. "
            f"Supported tasks: {supported}"
        )

    profile = SUPPORTED_TASK_PROFILES[normalized]
    if backend != "auto" and backend != profile.backend:
        raise ValueError(
            f"Task '{task_name}' uses backend '{profile.backend}', "
            f"but backend '{backend}' was requested."
        )
    if control_mode != "auto" and control_mode != profile.control_mode:
        raise ValueError(
            f"Task '{task_name}' uses control_mode '{profile.control_mode}', "
            f"but control_mode '{control_mode}' was requested."
        )
    return replace(profile)


def create_task_agent(
    task_name: str,
    port_id: str = "/dev/ttyUSB0",
    baudrate: int = 115200,
    timeout: float = 0.1,
    backend: str = "auto",
    control_mode: str = "auto",
):
    profile = resolve_task_profile(task_name, backend=backend, control_mode=control_mode)
    if profile.backend == "robot":
        from utils.agent import Agent as RobotAgent

        return RobotTaskAdapter(RobotAgent(port_id, baudrate, timeout), profile)
    return MicroscopeTaskAdapter(profile)


class RobotTaskAdapter:
    def __init__(self, agent, profile: TaskProfile):
        self.agent = agent
        self.profile = profile

    def open(self):
        self.agent.open()

    def close(self):
        self.agent.close()

    @property
    def recording_Flag(self):
        return self.agent.recording_Flag

    @recording_Flag.setter
    def recording_Flag(self, value):
        self.agent.recording_Flag = value

    @property
    def delete_Flag(self):
        return self.agent.delete_Flag

    @delete_Flag.setter
    def delete_Flag(self, value):
        self.agent.delete_Flag = value

    @property
    def should_exit(self):
        return self.agent.should_exit

    def get_current_stage(self):
        return self.agent.get_current_stage()

    def get_img(self):
        return self.agent.get_img()

    def get_ee_pos(self):
        return self.agent.get_ee_pos()

    def get_qpos_vec(self):
        import numpy as np

        qpos = self.get_ee_pos()
        vec = np.zeros(14, dtype=np.float32)
        if qpos is not None:
            vec[: min(len(qpos), 14)] = np.asarray(qpos[:14], dtype=np.float32)
        return vec

    def execute_action(self, action, current_qpos=None, offsets=None):
        import numpy as np

        action = np.asarray(action, dtype=np.float32)
        current_qpos = current_qpos if current_qpos is not None else self.get_ee_pos()
        if current_qpos is None:
            return None, offsets

        dx, dy = offsets if offsets is not None else (action[0] - current_qpos[0], action[1] - current_qpos[1])
        x_position = action[0] - dx
        y_position = action[1] - dy
        target_qpos = [round(float(x_position), 0), round(float(y_position), 0), 0]
        self.agent.move_to_target(target_qpos)
        return [x_position, y_position, 0], (dx, dy)


class MicroscopeTaskAdapter:
    def __init__(self, profile: TaskProfile):
        self.profile = profile
        self.microscope = None
        self.xy_origin = None

    def open(self):
        from utils.olympus import Olympus_api

        self.microscope = Olympus_api()
        self.microscope.initialize()
        self.microscope.listen_thread.start()
        time.sleep(1.0)

        if self.profile.exposure is not None:
            self.microscope.set_exposure(self.profile.exposure)
        self.microscope.live_start()
        if self.profile.brightness is not None:
            self.microscope.set_brightness(self.profile.brightness)
        if self.profile.dichroic is not None:
            self.microscope.set_dichroic(self.profile.dichroic)
        if self.profile.z_position is not None:
            self.microscope.set_z_position(self.profile.z_position)
        if self.profile.xy_position is not None:
            self.microscope.set_xy_position(*self.profile.xy_position)
        self.microscope.stop_reset_z_position = self.profile.reset_z_position_on_stop
        if self.profile.relative_xy:
            self.xy_origin = self.microscope.get_xy_position()

    def close(self):
        if self.microscope is None:
            return
        try:
            self.microscope.live_stop()
        finally:
            self.microscope.shutdown()

    @property
    def recording_Flag(self):
        return self.microscope.recording_Flag

    @recording_Flag.setter
    def recording_Flag(self, value):
        self.microscope.recording_Flag = value

    @property
    def delete_Flag(self):
        return self.microscope.delete_Flag

    @delete_Flag.setter
    def delete_Flag(self, value):
        self.microscope.delete_Flag = value

    @property
    def should_exit(self):
        return getattr(self.microscope, "should_exit", False)

    def get_current_stage(self):
        return 0

    def get_img(self):
        import cv2

        img = self.microscope.get_current_img()
        if img is None:
            return None
        img = cv2.resize(img, (640, 480))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_ee_pos(self):
        mode = self.profile.control_mode
        if mode == "z":
            return [float(self.microscope.get_z_position())]
        if mode == "brightness":
            return [float(self.microscope.get_brightness())]
        if mode == "exposure":
            return [float(self.microscope.get_exposure())]
        if mode == "xy":
            x, y = self.microscope.get_xy_position()
            if self.profile.relative_xy:
                if self.xy_origin is None:
                    self.xy_origin = (x, y)
                x -= self.xy_origin[0]
                y -= self.xy_origin[1]
            return [float(x), float(y)]
        raise ValueError(f"Unsupported microscope control mode: {mode}")

    def get_qpos_vec(self):
        import numpy as np

        qpos = self.get_ee_pos()
        vec = np.zeros(14, dtype=np.float32)
        vec[: min(len(qpos), 14)] = np.asarray(qpos[:14], dtype=np.float32)
        return vec

    def execute_action(self, action, current_qpos=None, offsets=None):
        import numpy as np

        action = np.asarray(action, dtype=np.float32)
        mode = self.profile.control_mode

        if mode == "z":
            value = float(action[0])
            self.microscope.set_z_position(value)
            return [value], offsets
        if mode == "brightness":
            value = float(action[0])
            self.microscope.set_brightness(value)
            return [value], offsets
        if mode == "exposure":
            value = float(action[0])
            self.microscope.set_exposure(value)
            return [value], offsets
        if mode == "xy":
            current_qpos = current_qpos if current_qpos is not None else self.get_ee_pos()
            if self.profile.relative_xy:
                origin = self.xy_origin or (0.0, 0.0)
                target_x = float(action[0] + origin[0])
                target_y = float(action[1] + origin[1])
            else:
                dx, dy = offsets if offsets is not None else (action[0] - current_qpos[0], action[1] - current_qpos[1])
                target_x = float(action[0] - dx)
                target_y = float(action[1] - dy)
                offsets = (dx, dy)
            self.microscope.set_xy_position(target_x, target_y)
            return [target_x, target_y], offsets
        raise ValueError(f"Unsupported microscope control mode: {mode}")
