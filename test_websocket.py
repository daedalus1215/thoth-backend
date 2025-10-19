import asyncio
import websockets
import numpy as np
import json


async def test_websocket_streaming():
    """Test the WebSocket streaming endpoint"""
    uri = "ws://localhost:8000/stream-audio"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Generate some test audio data (1 second of sine wave)
            sample_rate = 16000
            duration = 1.0
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Send audio data in chunks
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await websocket.send(chunk.tobytes())
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    if data.get("transcription"):
                        print(f"Transcription: {data['transcription']}")
                except asyncio.TimeoutError:
                    print("No transcription received for this chunk")
                
                # Small delay between chunks
                await asyncio.sleep(0.1)
            
            print("Test completed successfully!")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket_streaming())
