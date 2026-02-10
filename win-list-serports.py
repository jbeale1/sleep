import serial.tools.list_ports

for port in serial.tools.list_ports.comports():
    print(f"{port.device}")
    print(f"  Description: {port.description}")
    print(f"  HWID:        {port.hwid}")
    print(f"  VID:PID:     {port.vid:04X}:{port.pid:04X}" if port.vid else "  VID:PID:     N/A")
    print(f"  Serial:      {port.serial_number}")
    print(f"  Manufacturer:{port.manufacturer}")
    print(f"  Product:     {port.product}")
    print()