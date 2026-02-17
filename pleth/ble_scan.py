import asyncio
from bleak import BleakClient

async def main():
    async with BleakClient("ED:9B:47:2E:0F:68") as client:
        for service in client.services:
            print(f"\nService: {service.uuid}")
            for char in service.characteristics:
                props = ','.join(char.properties)
                print(f"  Char: {char.uuid}  [{props}]")

asyncio.run(main())