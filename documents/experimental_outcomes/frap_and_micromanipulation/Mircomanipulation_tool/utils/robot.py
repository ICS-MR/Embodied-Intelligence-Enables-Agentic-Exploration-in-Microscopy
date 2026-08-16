import threading
import serial
import time
import struct
from typing import Optional
import numpy as np


class SerialCommunication:
    def __init__(self, port, baudrate, timeout=1):
        """
        Initialize serial communication.
        :param port: Serial port, for example 'COM3' or '/dev/ttyUSB0'.
        :param baudrate: Serial baud rate, for example 9600.
        :param timeout: Read timeout in seconds.
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
                # Verify the port state with a few short retries.
                for _ in range(3):
                    if self.ser.is_open:
                        print("Serial port opened successfully")
                        return
                    time.sleep(0.1)
                raise serial.SerialException("Timed out while opening serial port")
        except Exception as e:
            print(f"Unable to open serial port: {e}")
            raise
    
    

    def write_data(self, data):
        """
        Write bytes to the serial port.
        :param data: Payload to write; must be bytes.
        """
        if not isinstance(data, bytes):
            raise TypeError("Serial payload must be bytes")
        try:

            if self.ser and self.ser.is_open:
                self.ser.reset_input_buffer()
                self.rx_buffer.clear()
                self.ser.write(data)
            else:
                print("Serial port is not open; cannot write data")
        except serial.SerialException as e:
            print(f"Serial write failed: {e}")

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
        Read one framed payload from the serial port.
        :return: A bytes frame, or None when no frame is available.
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
            print(f"Serial read failed: {e}")
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
                print("Serial port is not open or is already closed")
        except serial.SerialException as e:
            print(f"Unable to close serial port: {e}")

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
        """Extract one complete frame from the receive buffer."""
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
        # Control frame requesting the robot coordinates.
        my_set = bytes([0x5D, 0x5B, 0x01, 0x01, 0x01, 0xFE, 0x40, 0x55,
                        0x00, 0x01, 0x00, 0xD2, 0x21, 0x5D, 0x5D])

        MAX_RETRY = 5
        COORD_START_IDX = 11
        COORD_RANGE = (-10000.0, 10000.0)
        for attempt in range(1, MAX_RETRY + 1):
            try:
                with self.lock:
                    self.port.write_data(my_set)
                    axis_pose_bytes = self.port.read_data()

                if not axis_pose_bytes:
                    print(f"[Attempt {attempt}] No data returned")
                    continue

                # Validate the frame header and trailer.
                if axis_pose_bytes[:2] != b'\x5D\x5B' or axis_pose_bytes[-2:] != b'\x5D\x5D':
                    print(f"[Attempt {attempt}] Invalid frame header or trailer")
                    continue

                # Optional frame-length validation.
                # if not (len(axis_pose_bytes) == 194):
                #     print(f"[Attempt {attempt}] Unexpected frame length: {len(axis_pose_bytes)}")
                #     continue

                # Decode coordinate values.
                try:
                    x, y, z = struct.unpack('>fff', axis_pose_bytes[COORD_START_IDX:COORD_START_IDX+12])
                except struct.error as e:
                    print(f"[Attempt {attempt}] Coordinate decoding failed: {e}")
                    continue

                # Validate coordinate ranges.
                if not all(COORD_RANGE[0] <= val <= COORD_RANGE[1] for val in (x, y, z)):
                    print(f"[Attempt {attempt}] Coordinates out of range: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                    continue
                
                # Reject implausibly large jumps on the Y axis.
                if self.current_pose is not None:
                    delta_y = abs(self.current_pose[1] - y)
                    if delta_y > 600:
                        print(f"[Attempt {attempt}] Abnormal Y-axis jump: {delta_y:.2f}")
                        continue

                self.current_pose = [x, y, z]
                current_raw_pose = np.array([x, y, z])
                if self.smoothed_pose is None:
                    self.smoothed_pose = current_raw_pose
                else:
                    self.smoothed_pose = self.alpha * current_raw_pose + (1 - self.alpha) * self.smoothed_pose

                # Return the smoothed pose after all checks pass.
                return [round(val, 2) for val in self.smoothed_pose]

            except Exception as e:
                print(f"[Attempt {attempt}] Robot communication error: {e}")
                self._reconnect_port()

        print(f"Failed to acquire robot coordinates after {MAX_RETRY} attempts.")
        if self.smoothed_pose is not None:
            last_pose = [round(val, 2) for val in self.smoothed_pose]
            print(f"    Returning the last smoothed pose: {last_pose}")
            return last_pose
        print("    No valid coordinates have been received; returning [0, 0, 0]")
        return [0.0, 0.0, 0.0]

    def _reconnect_port(self):
        """Reconnect the serial port."""
        try:
            self.port.close_port()
            time.sleep(0.5)
            self.port.open_port()
        except Exception as e:
            print(f"Serial port reconnection failed: {e}")

    def move_pose(self, actions):
        # Use a fixed movement speed.
        speed = 4000.0
        MAX_RETRY = 3
        self.is_moving = True
        # Send one movement command per axis.
        for axis, coord in zip(['x', 'y', 'z'], actions):
            if axis == 'z':
                continue
            # Select the protocol byte for this axis.
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
                        print(f'Timed out waiting for the {axis}-axis movement response')
                except Exception as e:
                    print(f'Command transmission failed: {e}')
                    self._reconnect_port()
            self.is_moving = False

    @staticmethod
    def crc16_modbus(data: bytes) -> int:
        """
        Compute a Modbus CRC16 checksum.
        :param data: Data covered by the checksum.
        :return: CRC16 checksum.
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
        # Build the protocol command frame.
        command = [
            0x5D, 0x5B,                                         # Frame header
            0x01, 0x01, 0x01, 0xFE, 0x60, 0x66, 0x00, 0x09,     # Fixed parameters
            axis_byte,                                          # Axis selector
        ]
        # Append position and speed as big-endian IEEE 754 floats.
        command.extend(struct.pack('>f', coord))
        command.extend(struct.pack('>f', speed))
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

            # A short frame may acknowledge the interrupt without coordinates.
            self.is_moving = False
            self.is_interrupt = False
            print('Movement interrupted')

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
