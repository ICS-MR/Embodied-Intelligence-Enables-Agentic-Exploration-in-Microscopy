from pymmcore_plus import CMMCorePlus
import cv2 as cv
import queue
import threading
import os
import time
import numpy as np
from pynput import keyboard
from docs.VLA.Mircomanipulation_tool.utils.image_processing import image_process

mm_dir = os.environ.get("MICRO_MANAGER_DIR", r"C:\Program Files\Micro-Manager-2.0")
config_path = os.environ.get("MICRO_MANAGER_CONFIG", r"E:\ZZY\MMConfig_demo2.cfg")

Max_Z_position = 10000
Min_Z_position = 0

Max_brightness = 250
Min_brightness = 0

live_fps = 30

objective_labels = {
    '1-UPLFLN4XPH':4,
    '2-SOB':10,
    '3-LUCPLFLN20XRC':20,
    '4-LUCPLFLN40X':40,
    '5-LUCPLFLN60X':60,
    '6-UPLSAPO30XS':30
}


class Olympus_api:
    def __init__(self, app_dir: str = mm_dir, config_file: str = config_path, max_queue_size = 10):
        self.app_dir = app_dir
        self.config_file = config_file
        self.core = CMMCorePlus()

        self.camera_device = 'Camera-1'
        self.xy_stage_device = 'XYStage'
        self.objective_device = 'Objective'
        self.transmittedIllumination_2 = 'TransmittedIllumination 2'
        self.focus_drive = 'FocusDrive'
        self.shutter_device = 'EpiShutter 2'
        self.Dichroic = 'Dichroic 2'

        self.colormaps = {
            '1-NONE': 'gray',
            '2-U-FUNA': 'blue',
            '3-U-FBNA': 'green',
            '4-U-FGNA': 'red'
        }

        self.Max_Z_position = Max_Z_position
        self.Min_Z_position = Min_Z_position
        self.Max_brightness = Max_brightness
        self.Min_brightness = Min_brightness
        self.target_fps = live_fps

        self.current_pixel_size = None

        self.exposure_time = None
        self.gain = None
        self.objective_label = None
        self.Dichroic_label = None
        self.X_position = None
        self.Y_position = None
        self.Z_position = None
        self.shutter_state = None

        self.color = None

        self.objective_states = []
        self.Dichroic_states = []

        self.AutoContrast_flag = False
        self.recording_Flag = False
        self.delete_Flag = False
        self.stage_Flag = False
        self.get_pos_Flag = True
        self.should_exit = False
        self.activate = False
        self.live_running = threading.Event()

        self.Lock = threading.RLock()
        self.frame_queue = queue.Queue(maxsize = max_queue_size)
        self.listen_thread = threading.Thread(target=self.listen, name='listen')    
    
    def listen(self):
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()
    
    def initialize(self):
        """Load Micro-Manager configuration and initialize device state."""
        try:
            current_path = os.environ.get("PATH", "")
            if self.app_dir and self.app_dir not in current_path.split(os.pathsep):
                os.environ["PATH"] += os.pathsep + self.app_dir
                print(f"Added {self.app_dir} to PATH.")

            self.core.loadSystemConfiguration(self.config_file)
            print(f"Loaded Micro-Manager config: {self.config_file}")
            if not self.core.getLoadedDevices():
                print("No devices loaded. Check the config file and adapter paths.")
            else:
                print("Loaded devices:", self.core.getLoadedDevices())

            self.core.waitForSystem()
            self.core.enableDebugLog(False)
            self.core.enableStderrLog(False)

            self.exposure_time = self.get_exposure()
            self.gain = self.get_gain()
            self.X_position, self.Y_position = self.get_xy_position()
            self.Z_position = self.get_z_position()
            self.shutter_state = self.get_shutter_state()

            self.objective_states = self.core.getStateLabels(self.objective_device)
            print(f"Available objectives: {self.objective_states}")
            self.objective_label = self.get_objective()
            self.current_pixel_size = float(1.6234*4/objective_labels[self.objective_label])

            self.Dichroic_states = self.core.getStateLabels(self.Dichroic)
            print(f"Available dichroic states: {self.Dichroic_states}")
            self.Dichroic_label = self.get_dichroic()

            self.set_brightness(self.Max_brightness//2)
            self.set_z_position((self.Max_Z_position - self.Min_Z_position)//2)

            self.ip = image_process(0.05)
            self.activate = True

            return True

        except Exception as e:
            print(f"Initialization failed: {e}")
            self.shutdown()
            raise
    
    def get_xy_position(self):
        try:
            X_position, Y_position = self.core.getXYPosition()
            self.X_position, self.Y_position = X_position, Y_position
            return self.X_position, self.Y_position
        except Exception as e:
            print(f'Failed to read XY stage position: {e}')
            self.shutdown()
    
    def set_xy_position(self, x, y):
        try:
            if x == self.X_position and y == self.Y_position:
                return
            self.core.setXYStageDevice(self.xy_stage_device)
            self.core.setXYPosition(x, y)
            self.X_position = x
            self.Y_position = y
            self.core.waitForDevice(self.xy_stage_device)
        except Exception as e:
            print(f'Failed to move XY stage: {e}')
            self.shutdown()
    
    def get_z_position(self):
        try:
            self.Z_position = self.core.getPosition(self.focus_drive)
            return self.Z_position
        except Exception as e:
            print(f'Failed to read Z position: {e}')
            self.shutdown()

    def set_z_position(self, z):
        try:
            z = max(Min_Z_position, min(z, Max_Z_position))
            self.core.setPosition(self.focus_drive, z)
            self.Z_position = self.get_z_position()
            self.core.waitForDevice(self.focus_drive)
        except Exception as e:
            print(f'Failed to set Z position: {e}')
            self.shutdown()
    
    def get_exposure(self):
        try:
            self.exposure_time = self.core.getProperty(self.camera_device, "Exposure")
            print(f"Current Exposure Time: {self.exposure_time} ms")
            return self.exposure_time
        except Exception as e:
            print(f"Failed to read exposure: {e}")
            self.shutdown()
            raise
    
    def set_exposure(self, exposure_time):
        try:
            with self.Lock:
                live_state = self.live_running.is_set()
                if live_state:
                    self.live_stop()
                self.core.setProperty(self.camera_device, 'Exposure', exposure_time )
                self.core.waitForDevice(self.camera_device)
                self.exposure_time = self.get_exposure()
                print(f"Set Exposure Time: {self.exposure_time} ms")
                if live_state:
                    threading.Thread(target=self.live_start, name='1').start()
        except Exception as e:
            print(f"Failed to set exposure: {e}")
            self.shutdown()
            raise

    def get_gain(self):
        try:
            self.gain = self.core.getProperty(self.camera_device, "Gain")
            print(f"Current Gain: {self.gain}")
            return self.gain
        except Exception as e:
            print(f"Failed to read gain: {e}")
            self.shutdown()
            raise

    def set_gain(self, gain):
        try:
            self.core.setProperty(self.camera_device, "Gain", gain)
            self.core.waitForDevice(self.camera_device)
            self.gain = self.get_gain()
            print(f"Set Gain: {self.gain}")
        except Exception as e:
            print(f"Failed to set gain: {e}")
            self.shutdown()
            raise
    
    def get_objective(self):
        self.objective_label = self.core.getStateLabel(self.objective_device)
        self.current_pixel_size = float(1.6234*4/objective_labels[self.objective_label])
        return self.objective_label

    def set_objective(self, objective_label):
        try:
            if self.objective_device and objective_label in self.objective_states:
                self.core.setStateLabel(self.objective_device, objective_label)
                self.core.waitForDevice(self.objective_device)
                self.objective_label = self.get_objective()
                print(f"Switched to {self.objective_label} objective.")

            else:
                raise ValueError(f"Objective {objective_label} not found.")
        except Exception as e:
            print(f"Failed to set objective: {e}")
            self.shutdown()
            raise
    
    def get_dichroic(self):
        self.Dichroic_label = self.core.getStateLabel(self.Dichroic)
        self.color = self.colormaps[self.Dichroic_label]
        return self.Dichroic_label

    def set_dichroic(self, Dichroic_label):

        try:
            if self.Dichroic and Dichroic_label in self.Dichroic_states:
                self.core.setStateLabel(self.Dichroic, Dichroic_label)
                self.core.waitForDevice(self.Dichroic)
                self.Dichroic_label = self.get_dichroic()
                print(f"Switched to {self.Dichroic_label} objective.")

            else:
                raise ValueError(f"Objective {Dichroic_label} not found.")
        except Exception as e:
            print(f"Failed to set dichroic state: {e}")
            self.shutdown()
            raise

    def get_shutter_state(self):
        self.shutter_state = self.core.getProperty(self.shutter_device, "State")
        print("Current shutter state:", self.shutter_state)

    def set_shutter_state(self, state):
        self.core.setProperty(self.shutter_device, "State", state)
        self.core.waitForDevice(self.shutter_device)
        self.shutter_state = self.get_shutter_state()
        print("Current shutter state:", state)
    
    def get_brightness(self):
        self.brightness = self.core.getProperty(self.transmittedIllumination_2, 'Brightness')
        print("Current Brightness:", self.brightness)
        return self.brightness
    
    def set_brightness(self, brightness):
        brightness = max(Min_brightness, min(brightness, Max_brightness))
        self.core.setProperty(self.transmittedIllumination_2, 'Brightness', brightness)
        self.core.waitForDevice(self.transmittedIllumination_2)
        time.sleep(1.5)
        print("Switch to Current Brightness:", brightness)

    def get_current_img(self):
        if self.activate:
            while True:
                with self.Lock:
                    if self.live_running.is_set():
                        max_retries = 3
                        retries = 0
                        frame = None
                        while retries < max_retries and self.live_running.is_set():
                            
                            try:
                                frame = self.core.getLastImage()
                                break
                            except Exception as e:
                                retries += 1
                                time.sleep(0.001)
                                if retries >= max_retries:
                                    print(f"Failed to get image after {max_retries} retries: {e}")
                    else:
                        frame = self.core.snap()
                
                if frame is  not None:
                    if frame.dtype == np.uint16:
                        img = cv.convertScaleAbs(frame, alpha=(1.0/256.0))
                    else:
                        img = frame.copy()
                    with self.Lock:
                        if self.AutoContrast_flag:
                            img = self.ip.auto_adjust_brightness_contrast(img)
                    img = self.ip.set_color(img, self.color)
                    return img
                else:
                    continue
        else:
            print("Device is not initialized")

    def capture_frame(self):
        try:
            target_period = 1.0 / self.target_fps
            next_time = time.perf_counter()
            while self.live_running.is_set():
                img = self.get_current_img()
                try:
                    self.frame_queue.put_nowait(img)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(img)

                current_time = time.perf_counter()
                sleep_time = next_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_time += target_period
        except Exception as e:
            print(f"Capture error: {e}")
            self.live_stop()
    
    def process_frame(self):
        cv.namedWindow("Live Image", cv.WINDOW_NORMAL)
        cv.resizeWindow("Live Image", 1024, 1024)
        try:
            while self.live_running.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.05)
                    cv.imshow("Live Image", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        self.live_running.clear()
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Processing error: {e}")
            self.live_stop()

    def live_start(self):
        try:
            self.core.startContinuousSequenceAcquisition(intervalMs=1000//self.target_fps)
            self.live_running.set()
            time.sleep(2)

            cature_frame_thread = threading.Thread(target=self.capture_frame, name='cature_frame', daemon=True)
            cature_frame_thread.start()

            process_frame_thread = threading.Thread(target=self.process_frame, name='process_frame',daemon=True)
            process_frame_thread.start()

        except Exception:
            self.shutdown()
        
    def live_stop(self):
        try:
            if self.live_running.is_set():
                self.live_running.clear()
                self.core.stopSequenceAcquisition()
                time.sleep(1)
        except Exception as e:
            print(f"Live stop error: {e}")
            self.shutdown()

    def set_AutoContrast_flag(self, flag):
        with self.Lock:
            self.AutoContrast_flag = flag
    

    def get_field_of_view(self):
        """Return the current field of view in micrometers."""
        try:
            pixel_size = self.current_pixel_size
            image_width = self.core.getImageWidth()
            image_height = self.core.getImageHeight()
            fov_width = image_width * pixel_size
            fov_height = image_height * pixel_size

            print(f"Current field of view: {fov_width} x {fov_height} um")
            return fov_width, fov_height

        except Exception as e:
            print(f"Failed to calculate field of view: {e}")
            raise

    def shutdown(self):
        try:
            self.core.unloadAllDevices()
            self.core.reset()
            print("System shutdown completed.")
        except Exception as e:
            print(f"Failed to shut down system: {e}")
    
    def on_key_press(self, key):
        try:    
            key_name = key.char if isinstance(key, keyboard.KeyCode) else key.name
            
            if key_name == 'y':
                self.recording_Flag = True
                print('ok')
            elif key_name == "n":
                self.recording_Flag = False
                time.sleep(1)
                self.set_z_position(5000)
                print('no')
            elif key_name == 'q':
                self.live_stop()
                self.shutdown()
                print('stop')
                return False
            elif key_name == 'esc':
                self.should_exit = True
                self.recording_Flag = False
                print('stop')
                return False
            elif key_name == 'delete':
                self.delete_Flag = True 
                print('delete last result of your collection')
            elif key_name == 'space':
                self.stage_Flag = True
                print('enter a new stage')
            else:
                pass
        except Exception as e:
            print(f"Failed to handle key press: {e}")
            raise

def adjust_contrast(image: np.ndarray) -> np.ndarray:
    if len(image.shape) not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            print("Color input detected; converting to grayscale")
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    try:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    except Exception as e:
        raise RuntimeError(f"CLAHE failed: {str(e)}")
