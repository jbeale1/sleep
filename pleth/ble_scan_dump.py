#!/usr/bin/env python3
"""Dump all BLE devices seen in 10 seconds, showing names, addresses, and service UUIDs."""
import asyncio
from bleak import BleakScanner

seen = {}

def callback(device, adv):
    key = device.address
    if key not in seen:
        seen[key] = {
            'device_name': device.name,
            'local_name': adv.local_name,
            'address': device.address,
            'service_uuids': adv.service_uuids,
            'manufacturer_data': dict(adv.manufacturer_data),
            'rssi': adv.rssi,
        }
        print(f"  {device.address}  name={device.name!r}  local={adv.local_name!r}  "
              f"uuids={adv.service_uuids}  mfr={dict(adv.manufacturer_data)}  rssi={adv.rssi}")

async def main():
    print("Scanning 10 seconds... (have oximeter on with finger inserted)")
    scanner = BleakScanner(detection_callback=callback)
    await scanner.start()
    await asyncio.sleep(10)
    await scanner.stop()
    print(f"\n{len(seen)} devices found total.")

asyncio.run(main())
