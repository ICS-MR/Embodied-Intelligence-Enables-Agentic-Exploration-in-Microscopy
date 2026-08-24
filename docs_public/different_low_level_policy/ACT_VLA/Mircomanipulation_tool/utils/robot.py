import threading
import serial
import time
import struct
from typing import Optional
import numpy as np


class SerialCommunication:
    def __init__(self, port, baudrate, timeout=1):
        """
        Initialize the serial communication class.
        :param port: Serial port, such as 'COM3' or '/dev/ttyUSB0'
        :param baudrate: Baud rate, such as 9600
        :param timeout: Timeout in seconds; defaults to 1 second
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
        Open the serial port.
        """
        try:
            if not self.ser or not self.ser.is_open:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                # Verify that the serial port opened successfully.
                for _ in range(3):  # Retry three times.
                    if self.ser.is_open:
                        print("Port opened successfully")
                        return
                    time.sleep(0.1)
                raise serial.SerialException("Timed out while opening the port")
        except Exception as e:
            print(f"Failed to open the serial port: {e}")
            raise  # Propagate the exception.
    
    

    def write_data(self, data):
        """
        Write data to the serial port.
        :param data: Data to write; must be bytes
        """
        if not isinstance(data, bytes):
            raise TypeError("Data must be of type bytes")
        try:

            if self.ser and self.ser.is_open:
                self.ser.reset_input_buffer()
                self.rx_buffer.clear()
                self.ser.write(data)
            else:
                print("The serial port is not open; data cannot be written")
        except serial.SerialException as e:
            print(f"Failed to write data: {e}")

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
        Read data from the serial port.
        :return: Received bytes, or None if no data is available
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
            print(f"Failed to read data: {e}")
            return None

    def close_port(self):
        """
        Close the serial port.
        """
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                # print("Serial port closed")
            else:
                print("The serial port is not open or is already closed")
        except serial.SerialException as e:
            print(f"Failed to close the serial port: {e}")

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
        """Extract a complete frame from the buffer."""
        while len(self.buffer) >= 4:
            # Find the frame header.
            start = self.buffer.find(b'\x5D\x5B')
            if start == -1:
                self.buffer.clear()
                return None
            
            # Find the frame trailer.
            end = self.buffer.find(b'\x5D\x5D', start + 2)
            if end == -1:
                return None
            
            frame = self.buffer[start:end+2]
            del self.buffer[:end+2]
            return frame
        return None

    def get_pose(self):
        # Control frame requesting robotic manipulator coordinates.
        my_set = bytes([0x5D, 0x5B, 0x01, 0x01, 0x01, 0xFE, 0x40, 0x55,
                        0x00, 0x01, 0x00, 0xD2, 0x21, 0x5D, 0x5D])

        MAX_RETRY = 5            # Maximum allowed failures.
        COORD_START_IDX = 11     # Starting index of coordinate data.
        COORD_RANGE = (-10000.0, 10000.0)
        for attempt in range(1, MAX_RETRY + 1):
            try:
                with self.lock:
                    self.port.write_data(my_set)
                    axis_pose_bytes = self.port.read_data()

                if not axis_pose_bytes:
                    print(f"[Attempt {attempt}] ❌ No data returned")
                    continue

                # 1. ✅ Validate the frame header and trailer.
                if axis_pose_bytes[:2] != b'\x5D\x5B' or axis_pose_bytes[-2:] != b'\x5D\x5D':
                    print(f"[Attempt {attempt}] ❌ Invalid frame header or trailer")
                    continue

                # 2. ✅ Check the length and skip malformed frames.
                # if not (len(axis_pose_bytes) == 194):
                #     print(f"[Attempt {attempt}] ⚠️ Invalid frame length: {len(axis_pose_bytes)}; skipping")
                #     continue

                # 3. ✅ Parse coordinate data.
                try:
                    x, y, z = struct.unpack('>fff', axis_pose_bytes[COORD_START_IDX:COORD_START_IDX+12])
                except struct.error as e:
                    print(f"[Attempt {attempt}] ❌ Failed to decode coordinates: {e}")
                    continue

                # 4. ✅ Validate the coordinate range.
                if not all(COORD_RANGE[0] <= val <= COORD_RANGE[1] for val in (x, y, z)):
                    print(f"[Attempt {attempt}] ❌ Coordinates out of range: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                    continue
                
                # 5. ✅ Validate special cases.
                if self.current_pose is not None:
                    delta_y = abs(self.current_pose[1] - y)
                    if delta_y > 600:
                        print(f"[Attempt {attempt}] ⚠️ Abnormal Y-axis jump: {delta_y:.2f}")
                        continue

                self.current_pose = [x, y, z]
                current_raw_pose = np.array([x, y, z])
                if self.smoothed_pose is None:
                    self.smoothed_pose = current_raw_pose
                else:
                    self.smoothed_pose = self.alpha * current_raw_pose + (1 - self.alpha) * self.smoothed_pose

                # ✅ Return the value after all checks pass.
                return [round(val, 2) for val in self.smoothed_pose]

            except Exception as e:
                print(f"[Attempt {attempt}] ⚠️ System exception: {e}")
                self._reconnect_port()

        print(f"⚠️ Failed to acquire robotic manipulator coordinates {MAX_RETRY} consecutive times.")
        if self.smoothed_pose is not None:
            last_pose = [round(val, 2) for val in self.smoothed_pose]
            print(f"    Returning the previous smoothed coordinates: {last_pose}")
            return last_pose
        print("    Critical warning: no valid coordinates have ever been received. Returning [0, 0, 0]")
        return [0.0, 0.0, 0.0]

    def _reconnect_port(self):
        """Reconnect the serial port."""
        try:
            self.port.close_port()
            time.sleep(0.5)
            self.port.open_port()
        except Exception as e:
            print(f"Failed to reconnect the port: {e}")

    def move_pose(self, actions):
        # Fixed speed value.
        speed = 4000.0
        MAX_RETRY = 3
        self.is_moving = True
        # Send a movement command for each coordinate.
        for axis, coord in zip(['x', 'y', 'z'], actions):
            if axis == 'z':
                continue
            # Select the axis byte (57:x, 58:y, 59:z).
            axis_byte = {
                'x': 0x57,
                'y': 0x58,
                'z': 0x59
            }[axis]
            command_bytes = self.command_create(axis_byte, speed, coord)
            # print(f'Movement command: {command_bytes.hex()}')
            for attempt in range(MAX_RETRY):
                try:
                    with self.lock:
                        self.port.write_data(command_bytes)
                        response = self.port.read_data()
                    if response and response[:2] == b'\x5D\x5B':
                        break
                    if attempt == MAX_RETRY - 1:
                        print(f'Timed out waiting for the {axis}-axis movement response; skipping this wait')
                except Exception as e:
                    print(f'Command transmission error: {e}')
                    self._reconnect_port()
            self.is_moving = False

    @staticmethod
    def crc16_modbus(data: bytes) -> int:
        """
        Calculate a CRC16 checksum.
        :param data: Data used to calculate the checksum
        :return: CRC16 checksum
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
        # Build the command frame.
        command = [
            0x5D, 0x5B,                                         # Frame header
            0x01, 0x01, 0x01, 0xFE, 0x60, 0x66, 0x00, 0x09,     # Fixed parameters
            axis_byte,                                          # Axis selection
        ]
        # Add position and speed as IEEE 754 big-endian single-precision floats.
        command.extend(struct.pack('>f', coord))                    # Position
        command.extend(struct.pack('>f', speed))                    # Speed
        crc = self.crc16_modbus(bytes(command[2:]))
        command.extend([crc & 0xFF, (crc >> 8) & 0xFF])             # CRC checksum
        command.extend([0x5D, 0x5D])                                # Frame trailer
        command_bytes = bytes(command)
        return command_bytes
    
    def interrupt_move(self):
        if self.is_interrupt:
            interrupt_cmd = bytes([0x5D, 0x5B, 0x01, 0x01, 0x01, 0xFE, 0x30, 0x55, 0x00, 0x01, 0x04, 0x92, 0x29, 0x5D, 0x5D])
            with self.lock:
                self.port.write_data(interrupt_cmd)
                response = self.port.read_data()
                # print(f"Interrupt response: {response.hex() if response else None}")

            # A short frame may acknowledge the interrupt and need no coordinate parsing.
            self.is_moving = False
            self.is_interrupt = False
            print('Interrupt completed')

    def set_interrupt(self):
        self.is_interrupt = True

    def monitoring_interrupt(self):
        while not self.is_interrupt:
            time.sleep(0.05)
        self.interrupt_move()
        print("[robot_motoring] Monitoring thread exited")

if __name__ == "__main__":
    target_pos = [0, 0, 0]
    robot = Robot('/dev/ttyUSB0', 115200, 0.1)
    robot.open()
