import asyncio
from bleak import BleakClient

ALL_CHARS = [
    ("0000fff0-0000-1000-8000-00805f9b34fb", "fff0-indicate"),
    ("0000fff1-0000-1000-8000-00805f9b34fb", "fff1-notify"),
    ("00000003-0000-6465-6d6d-65636c6f6843", "0003-notify"),
    ("0000ff01-0000-1000-8000-00805f9b34fb", "ff01-notify"),
]

def handler(label):
    def callback(sender, data):
        hex_str = data.hex(' ')
        print(f"[{label}] ({len(data):2d} bytes) {hex_str}")
    return callback

async def main():
    async with BleakClient("ED:9B:47:2E:0F:68") as client:
        # Subscribe to all notify/indicate characteristics
        for uuid, label in ALL_CHARS:
            try:
                await client.start_notify(uuid, handler(label))
                print(f"Subscribed: {label} ({uuid})")
            except Exception as e:
                print(f"Failed: {label} - {e}")

        # Try reading readable chars for clues
        for uuid in ["0000ff01-0000-1000-8000-00805f9b34fb",
                      "0000ff02-0000-1000-8000-00805f9b34fb",
                      "0000fff2-0000-1000-8000-00805f9b34fb"]:
            try:
                val = await client.read_gatt_char(uuid)
                print(f"Read {uuid[-8:-4]}: {val.hex(' ')}")
            except Exception as e:
                print(f"Read {uuid[-8:-4]} failed: {e}")

        # Some devices need a write to start streaming
        # Try writing 0x01 to the writable chars
        for uuid in ["0000ff02-0000-1000-8000-00805f9b34fb",
                      "0000ff03-0000-1000-8000-00805f9b34fb"]:
            try:
                await client.write_gatt_char(uuid, bytearray([0x01]))
                print(f"Wrote 0x01 to {uuid[-8:-4]}")
            except Exception as e:
                print(f"Write {uuid[-8:-4]} failed: {e}")

        print("\nListening 15 seconds...\n")
        await asyncio.sleep(15)

asyncio.run(main())