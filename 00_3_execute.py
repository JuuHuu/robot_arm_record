import socket
from pathlib import Path

HOST = "192.168.100.162"   # your PC IP or 0.0.0.0 to bind all
PORT = 30002       # must match External Control node port

urscript_path = Path("/home/juu/Documents/robot_arm_record/test.urscript")
script = urscript_path.read_text()

# Ensure script ends with newline
if not script.endswith("\n"):
    script += "\n"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(script.encode("utf-8"))
    print("Script sent.")
    s.close