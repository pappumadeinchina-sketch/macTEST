import os
import asyncio
import websockets

async def main():
    # Render provides PORT automatically (defaults to 10000 on free tier)
    port = int(os.environ.get("PORT", 10000))
    
    async with websockets.serve(
        handle_client, 
        "0.0.0.0", 
        port,
        ping_interval=20, # Prevents 100-second timeouts
        ping_timeout=20
    ):
        print(f"🚀 Server running on port {port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
