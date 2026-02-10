import serial.tools.list_ports

for port in serial.tools.list_ports.comports():
    print(f"{port.device}  {port.description}  VID:PID={port.vid}:{port.pid}")
    if port.vid == 0x2E8A:  # Raspberry Pi VID
        print(f"  ^^^ This is your Pico on {port.device}")