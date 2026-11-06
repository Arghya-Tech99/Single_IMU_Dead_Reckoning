import socket
import struct
import threading
import collections
import time

class UDPReceiver:
    def __init__(self, ip="0.0.0.0", port=5005, packet_format="<II6fI3f", max_dt=0.2):
        self.ip = ip
        self.port = port
        self.packet_format = packet_format
        self.packet_size = struct.calcsize(packet_format)
        self.max_dt = max_dt
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        
        self.latest_data = None
        self.packet_queue = collections.deque()
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listener_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _listener_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                if len(data) == self.packet_size:
                    unpacked = struct.unpack(self.packet_format, data)
                    with self.lock:
                        self.latest_data = unpacked
                        self.packet_queue.append(unpacked)
            except Exception as e:
                print(f"UDP Receiver Error: {e}")
                time.sleep(0.1)

    def get_latest_data(self):
        with self.lock:
            return self.latest_data

    def get_all_packets(self):
        with self.lock:
            packets = list(self.packet_queue)
            self.packet_queue.clear()
            return packets
