#!/usr/bin/env python3
"""Test WebSocket connection and messaging."""

import asyncio
import websockets
import json


async def test_websocket():
    uri = "ws://localhost:9000/ws"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")

            # Wait for initial connected message
            try:
                initial = await asyncio.wait_for(websocket.recv(), timeout=2)
                data = json.loads(initial)
                print(f"📥 Initial: {data}")
            except asyncio.TimeoutError:
                print("⏱️ No initial message")

            # Send test message
            test_msg = {"type": "message", "content": "Hello!"}
            await websocket.send(json.dumps(test_msg))
            print(f"📤 Sent: {test_msg}")

            # Receive response
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(response)
            print(f"📥 Received: {data}")
            print("✅ Success!")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_websocket())
    exit(0 if result else 1)
