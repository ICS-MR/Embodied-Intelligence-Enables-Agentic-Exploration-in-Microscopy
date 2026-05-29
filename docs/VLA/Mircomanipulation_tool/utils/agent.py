import copy
import time
from docs.VLA.Mircomanipulation_tool.utils.robot import Robot
import threading
from pynput import keyboard
import datetime
from docs.VLA.Mircomanipulation_tool.utils.camera import DahengCamera
# 用于存储按键按下时间的字典
key_press_times = {}
lock = threading.Lock()

step = 25
class Agent:
    def __init__(self, port_id, baudrate, timeout):

        self.ee_pos = None
        self.robot = Robot(port_id, baudrate, timeout)
        self.camera = DahengCamera()
        self.img = None
        self.latest_snapshot = {'image': None, 'qpos': None}
        self.snapshot_lock = threading.Lock()
        self.recording_Flag = False
        self.stage_Flag = False
        self.delete_Flag = False
        self.should_exit = False

        self.stage_definitions = {
            0: "approach",  # 接近阶段
            1: "manipulate", # 操作阶段
            2: "retract"    # 回撤阶段
        }
        self.current_stage = 0
        self.stage_lock = threading.Lock()

        self.camera_thread = threading.Thread(target=self.camera.cv_show_image, name='camera_show')
        self.listen_thread = threading.Thread(target=self.listen, name='listen')
        self.sync_thread_running = False
        self.sync_thread = threading.Thread(target=self._synchronization_loop, name='sync_loop', daemon=True)
        self.comm_lock = threading.Lock()
        self.buffer = bytearray()

    def _synchronization_loop(self):
        """后台同步读取相机缓存和机械臂位置，前台采集只读快照。"""
        self.sync_thread_running = True
        print("[信息] 同步线程已启动...")
        while self.sync_thread_running:
            with self.comm_lock:
                current_pose = self.robot.get_pose()

            if current_pose is None:
                time.sleep(0.01)
                continue

            current_image = self.camera.cv_get_image()
            if current_image is None:
                time.sleep(0.01)
                continue

            with self.snapshot_lock:
                self.latest_snapshot['image'] = current_image
                self.latest_snapshot['qpos'] = current_pose

            time.sleep(0.01)
        print("[信息] 同步线程已停止。")

    def open(self):
        self.robot.open()
        # 打开相机
        self.camera_thread.start()
        # self.camera_thread.daemon = True
        self.listen_thread.start()
        self.sync_thread.start()
        print("等待初始同步快照...")
        temp_pos = None
        while temp_pos is None and not self.should_exit:
            with self.snapshot_lock:
                temp_pos = self.latest_snapshot['qpos']
            if temp_pos is None:
                time.sleep(0.1)
        self.ee_pos = temp_pos
        print(f'初始化结束，初始位置: {temp_pos}')

    def close(self):
        print("[INFO] 开始关闭 Agent...")
        self.sync_thread_running = False
        self.camera.close()
        print("[INFO] 相机已安全关闭")
        self.robot.close()
        print("[INFO] 机器人已安全关闭")
        if hasattr(self, 'listener'):
            self.listener.stop()
            print("[INFO] 监听器已安全关闭")
        if self.sync_thread.is_alive():
            self.sync_thread.join(timeout=0.5)
        print("[INFO] 所有线程已安全关闭")
    
    def listen(self):
        # 创建监听器对象
        self.listener = keyboard.Listener(on_press=self.on_key_press,
                                        on_release=self.on_key_release)
        self.listener.start()


    def move_to_target(self, target_pos):
        with self.comm_lock:
            self.robot.move_pose(target_pos)

    def get_ee_pos(self, retry=3):
        with self.snapshot_lock:
            qpos = self.latest_snapshot['qpos']
        self.ee_pos = copy.deepcopy(qpos) if qpos is not None else None
        return self.ee_pos

    def get_img(self):
        with self.snapshot_lock:
            img = self.latest_snapshot['image']
        self.img = img.copy() if img is not None else None
        return self.img

    def get_display_image(self):
        return self.camera.cv_get_image()
    
    def get_current_stage(self):
        with self.stage_lock:
            # 直接返回当前阶段编号，如果不在定义范围内则返回None或-1
            return self.current_stage if self.current_stage in self.stage_definitions else -1

    def get_current_stage_name(self):
        with self.stage_lock:
            return self.stage_definitions.get(self.current_stage, "unknown")

    def on_key_press(self, key):
        # 将按键对象转换为字符串名称
        key_name = key.char if isinstance(key, keyboard.KeyCode) else key.name
            
        # 记录按下时间
        if key_name not in key_press_times:
            key_press_times[key_name] = datetime.datetime.now()

        # 方向键处理
        if key_name in ['up', 'down', 'left', 'right']:
            self.robot.set_interrupt()
            current_pos = self.get_ee_pos()
            print(f"当前位置: {current_pos if current_pos else '获取失败'}")
            
            # 计算新位置
            new_pos = copy.deepcopy(current_pos)
            if key_name == 'up':
                new_pos[1] = max(-10000, new_pos[1] - step)
            elif key_name == 'down':
                new_pos[1] = min(10000, new_pos[1] + step)
            elif key_name == 'left':
                new_pos[0] = min(10000, new_pos[0] + step)
            elif key_name == 'right':
                new_pos[0] = max(-10000, new_pos[0] - step)
        
            move_theard = threading.Thread(target=self.move_to_target, name='move', args=(new_pos,))
            move_theard.start()
        
        if key_name == 'y':
            self.current_stage = 0
            self.recording_Flag = True
            print('ok')
        elif key_name == "n":
            self.recording_Flag = False
            print('no')
        elif key_name == 'space':
            if self.current_stage < len(self.stage_definitions) - 1:
                self.current_stage = self.current_stage + 1
                print(f'\n进入下一阶段: {self.get_current_stage_name()}')
            elif self.current_stage >= len(self.stage_definitions) - 1 :
                self.current_stage = len(self.stage_definitions) - 1
                print(f'\n已经是最后一阶段: {self.get_current_stage_name()}')
        elif key_name == "delete":
            self.delete_Flag = True
            print('delete last result of your collection')
        elif key_name == "esc":
            print("[ESC] 收到退出指令")
            self.should_exit = True
        else:
            pass

    def on_key_release(self, key):
        # 将按键对象转换为字符串名称
        if isinstance(key, keyboard.KeyCode):
            key_name = key.char
        else:
            key_name = key.name

        if key_name in ['up', 'down', 'left', 'right']:
            # 如果按键在字典中，计算持续时间
            if key_name in key_press_times:
                self.robot.set_interrupt()
                pressed_time = key_press_times.pop(key_name)  # 取出按下时间并移除
                released_time = datetime.datetime.now()
                duration = (released_time - pressed_time).total_seconds()
                # print(f"方向键 {key_name} 被松开，时间: {released_time}, 持续时间: {duration:.3f} 秒")
                # time.sleep(0.1)
                self.get_ee_pos()
                print(f"当前位置为{self.ee_pos}")

if __name__ == "__main__":
    agent = Agent('/dev/ttyUSB0', 115200, 0.1)
    agent.open()
    # try:
    #     while not agent.should_exit:
    #         time.sleep(0.01)
    # finally:
    #     agent.close()
    #     print("[MAIN] 程序已退出")
