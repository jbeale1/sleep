# display an updated list of what serial ports are available
# handy when you need to know what the new device is you just connected
# 2026-02-19 J.Beale

import serial.tools.list_ports
import time

def get_ports():
    return {p.device: p.description for p in serial.tools.list_ports.comports()}

known_ports = get_ports()
print("Initial ports:", known_ports)

while True:
    time.sleep(2)
    current_ports = get_ports()

    added = {k: v for k, v in current_ports.items() if k not in known_ports}
    removed = {k: v for k, v in known_ports.items() if k not in current_ports}

    for port, desc in added.items():
        print(f"[+] Added:   {port} — {desc}")
    for port, desc in removed.items():
        print(f"[-] Removed: {port} — {desc}")

    known_ports = current_ports