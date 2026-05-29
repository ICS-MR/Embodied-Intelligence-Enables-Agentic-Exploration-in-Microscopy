import threading
import serial
import time
import struct
from typing import Optional
import numpy as np


class SerialCommunication:
    def __init__(self, port, baudrate, timeout=1):
        """
        初始化串口通信类
        :param port: 串口号，例如 'COM3' 或 '/dev/ttyUSB0'
        :param baudrate: 波特率，例如 9600
        :param timeout: 超时时间（秒），默认为 1 秒
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.rx_buffer = bytearray()
        self.frame_head = b"\x5D\x5B"
        self.frame_tail = b"\x5D\x5D"

    def open_port(self):
        """
        打开串口
        """
        try:
            if not self.ser or not self.ser.is_open:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                # 添加串口打开状态验证
                for _ in range(3):  # 重试3次
                    if self.ser.is_open:
                        print("端口打开成功")
                        return
                    time.sleep(0.1)
                raise serial.SerialException("端口打开超时")
        except Exception as e:
            print(f"打开串口失败: {e}")
            raise  # 向上抛出异常
    
    

    def write_data(self, data):
        """
        向串口写入数据
        :param data: 要写入的数据，必须是 bytes 类型
        """
        if not isinstance(data, bytes):
            raise TypeError("数据必须是 bytes 类型")
        try:

            if self.ser and self.ser.is_open:
                self.ser.reset_input_buffer()
                self.rx_buffer.clear()
                self.ser.write(data)
            else:
                print("串口未打开，无法写入数据")
        except serial.SerialException as e:
            print(f"写入数据失败: {e}")

    def _trim_buffer(self):
        head_idx = self.rx_buffer.find(self.frame_head)
        if head_idx == -1:
            if self.rx_buffer and self.rx_buffer[-1] == self.frame_head[0]:
                self.rx_buffer[:] = self.rx_buffer[-1:]
            else:
                self.rx_buffer.clear()
            return
        if head_idx > 0:
            del self.rx_buffer[:head_idx]

    def _extract_frame(self) -> Optional[bytes]:
        self._trim_buffer()
        if not self.rx_buffer.startswith(self.frame_head):
            return None

        tail_idx = self.rx_buffer.find(self.frame_tail, len(self.frame_head))
        if tail_idx == -1:
            return None

        frame_end = tail_idx + len(self.frame_tail)
        frame = bytes(self.rx_buffer[:frame_end])
        del self.rx_buffer[:frame_end]
        return frame

    def read_data(self) -> Optional[bytes]:
        """
        从串口读取数据
        :return: 读取到的数据（bytes 类型），如果没有数据则返回 None
        """
        try:
            if not self.ser or not self.ser.is_open:
                return None

            deadline = time.perf_counter() + max(float(self.timeout), 0.1)
            while time.perf_counter() < deadline:
                frame = self._extract_frame()
                if frame is not None:
                    return frame

                waiting = self.ser.in_waiting
                if waiting > 0:
                    chunk = self.ser.read(waiting)
                    if chunk:
                        self.rx_buffer.extend(chunk)
                        frame = self._extract_frame()
                        if frame is not None:
                            return frame
                    continue

                time.sleep(0.001)

            return self._extract_frame()
        except serial.SerialException as e:
            print(f"读取数据失败: {e}")
            return None

    def close_port(self):
        """
        关闭串口
        """
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                # print("串口已关闭")
            else:
                print("串口未打开或已关闭")
        except serial.SerialException as e:
            print(f"关闭串口失败: {e}")

class Robot(object):
    def __init__(self, port_id, baudrate, timeout):
        self.port = SerialCommunication(port_id, baudrate, timeout)
        self.ee_pose = [0, 0, 0]
        self.actions = [0, 0, 0]

        self.channel = False 
        self.is_moving = False
        self.is_interrupt = False
        self.lock = threading.Lock()
        self.buffer = bytearray()

        self.thread_motoring = threading.Thread(target=self.monitoring_interrupt, name='robot_motoring')
        self.thread_motoring.daemon = True

        self.current_pose = None
        self.smoothed_pose = None
        self.alpha = 0.2

    def open(self):
        self.thread_motoring.start()
        self.port.open_port()
    
    def close(self):
        self.is_interrupt = True
        if self.thread_motoring.is_alive():
            self.thread_motoring.join(timeout=2)
        self.port.close_port()

    def _read_to_buffer(self):
        with self.lock:
            data = self.port.read_data()
            if data:
                self.buffer.extend(data)

    def _extract_frame(self) -> Optional[bytes]:
        """从缓冲区提取完整帧"""
        while len(self.buffer) >= 4:
            # 查找帧头
            start = self.buffer.find(b'\x5D\x5B')
            if start == -1:
                self.buffer.clear()
                return None
            
            # 查找帧尾
            end = self.buffer.find(b'\x5D\x5D', start + 2)
            if end == -1:
                return None
            
            frame = self.buffer[start:end+2]
            del self.buffer[:end+2]
            return frame
        return None

    def get_pose(self):
        # 控制帧：请求获取机械臂坐标
        my_set = bytes([0x5D, 0x5B, 0x01, 0x01, 0x01, 0xFE, 0x40, 0x55,
                        0x00, 0x01, 0x00, 0xD2, 0x21, 0x5D, 0x5D])

        MAX_RETRY = 5            # 最多允许失败次数
        COORD_START_IDX = 11     # 坐标数据起始索引
        COORD_RANGE = (-10000.0, 10000.0)
        for attempt in range(1, MAX_RETRY + 1):
            try:
                with self.lock:
                    self.port.write_data(my_set)
                    axis_pose_bytes = self.port.read_data()

                if not axis_pose_bytes:
                    print(f"[尝试 {attempt}] ❌ 无数据返回")
                    continue

                # 1. ✅ 校验帧头帧尾
                if axis_pose_bytes[:2] != b'\x5D\x5B' or axis_pose_bytes[-2:] != b'\x5D\x5D':
                    print(f"[尝试 {attempt}] ❌ 帧头帧尾错误")
                    continue

                # 2. ✅长度检测（跳过异常帧）
                # if not (len(axis_pose_bytes) == 194):
                #     print(f"[尝试 {attempt}] ⚠️ 异常帧长度: {len(axis_pose_bytes)}，跳过")
                #     continue

                # 3. ✅ 解析坐标数据
                try:
                    x, y, z = struct.unpack('>fff', axis_pose_bytes[COORD_START_IDX:COORD_START_IDX+12])
                except struct.error as e:
                    print(f"[尝试 {attempt}] ❌ 坐标解码失败: {e}")
                    continue

                # 4. ✅ 校验坐标范围
                if not all(COORD_RANGE[0] <= val <= COORD_RANGE[1] for val in (x, y, z)):
                    print(f"[尝试 {attempt}] ❌ 坐标越界: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                    continue
                
                # 5. ✅ 特例验证
                if self.current_pose is not None:
                    delta_y = abs(self.current_pose[1] - y)
                    if delta_y > 600:
                        print(f"[尝试 {attempt}] ⚠️ y轴跳变异常: {delta_y:.2f}")
                        continue

                self.current_pose = [x, y, z]
                current_raw_pose = np.array([x, y, z])
                if self.smoothed_pose is None:
                    self.smoothed_pose = current_raw_pose
                else:
                    self.smoothed_pose = self.alpha * current_raw_pose + (1 - self.alpha) * self.smoothed_pose

                # ✅ 全部校验通过，返回值
                return [round(val, 2) for val in self.smoothed_pose]

            except Exception as e:
                print(f"[尝试 {attempt}] ⚠️ 系统异常: {e}")
                self._reconnect_port()

        print(f"⚠️ 连续 {MAX_RETRY} 次采集机械臂坐标失败。")
        if self.smoothed_pose is not None:
            last_pose = [round(val, 2) for val in self.smoothed_pose]
            print(f"    返回上一次的平滑坐标: {last_pose}")
            return last_pose
        print("    严重警告：从未获取到有效坐标。返回 [0, 0, 0]")
        return [0.0, 0.0, 0.0]

    def _reconnect_port(self):
        """端口重连"""
        try:
            self.port.close_port()
            time.sleep(0.5)
            self.port.open_port()
        except Exception as e:
            print(f"端口重连失败: {e}")

    def move_pose(self, actions):
        # 固定速度值
        speed = 4000.0
        MAX_RETRY = 3
        self.is_moving = True
        # 发送每个坐标的移动指令
        for axis, coord in zip(['x', 'y', 'z'], actions):
            if axis == 'z':
                continue
            # 确定轴选择字节 (57:x, 58:y, 59:z)
            axis_byte = {
                'x': 0x57,
                'y': 0x58,
                'z': 0x59
            }[axis]
            command_bytes = self.command_create(axis_byte, speed, coord)
            # print(f'移动命令：{command_bytes.hex()}')
            for attempt in range(MAX_RETRY):
                try:
                    with self.lock:
                        self.port.write_data(command_bytes)
                        response = self.port.read_data()
                    if response and response[:2] == b'\x5D\x5B':
                        break
                    if attempt == MAX_RETRY - 1:
                        print(f'{axis} 轴移动应答超时，跳过本次应答等待')
                except Exception as e:
                    print(f'命令发送错误：{e}')
                    self._reconnect_port()
            self.is_moving = False

    @staticmethod
    def crc16_modbus(data: bytes) -> int:
        """
        计算CRC16校验码
        :param data: 要计算校验码的数据
        :return: CRC16校验码
        """
        crc = 0xFFFF  
        for byte in data:
            crc ^= byte  
            for _ in range(8):
                if crc & 0x0001: 
                    crc >>= 1
                    crc ^= 0xA001  
                else:
                    crc >>= 1
        return crc
    def command_create(self, axis_byte, speed, coord):
        # 构建指令帧
        command = [
            0x5D, 0x5B,                                         # 帧头
            0x01, 0x01, 0x01, 0xFE, 0x60, 0x66, 0x00, 0x09,     # 固定参数
            axis_byte,                                          # 轴选择
        ]
        # 添加位置和速度 (IEEE 754大端单精度浮点)
        command.extend(struct.pack('>f', coord))                    # 位置
        command.extend(struct.pack('>f', speed))                    # 速度
        crc = self.crc16_modbus(bytes(command[2:]))
        command.extend([crc & 0xFF, (crc >> 8) & 0xFF])             # CRC校验 
        command.extend([0x5D, 0x5D])                                # 帧尾
        command_bytes = bytes(command)
        return command_bytes
    
    def interrupt_move(self):
        if self.is_interrupt:
            interrupt_cmd = bytes([0x5D, 0x5B, 0x01, 0x01, 0x01, 0xFE, 0x30, 0x55, 0x00, 0x01, 0x04, 0x92, 0x29, 0x5D, 0x5D])
            with self.lock:
                self.port.write_data(interrupt_cmd)
                response = self.port.read_data()
                # print(f"中断响应: {response.hex() if response else None}")

            # 短帧可能是中断确认，无需解析坐标
            self.is_moving = False
            self.is_interrupt = False
            print('中断完成')

    def set_interrupt(self):
        self.is_interrupt = True

    def monitoring_interrupt(self):
        while not self.is_interrupt:
            time.sleep(0.05)
        self.interrupt_move()
        print("[robot_motoring] 退出监控线程")

if __name__ == "__main__":
    target_pos = [0, 0, 0]
    robot = Robot('/dev/ttyUSB0', 115200, 0.1)
    robot.open()
