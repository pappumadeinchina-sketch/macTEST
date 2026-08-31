import os
import re
import sys
import time
import json
import asyncio
import socket
import urllib.request
import numpy as np
import soundfile as sf
import websockets
import onnxruntime as ort
from groq import Groq
import edge_tts

# 1. API Key Check
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    print("❌ ERROR: GROQ_API_KEY environment variable is missing.")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_KEY)

# 2. Download Silero VAD ONNX Model if not present
ONNX_MODEL_PATH = "silero_vad.onnx"
if not os.path.exists(ONNX_MODEL_PATH):
    print("📥 Downloading lightweight Silero VAD ONNX model (~1.8 MB)...")
    url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    urllib.request.urlretrieve(url, ONNX_MODEL_PATH)
    print("✅ Model downloaded.")

print("🧠 Loading ONNX VAD Session (No PyTorch)...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
print("✅ ONNX VAD Ready.")

# State variables for ONNX Silero model VAD
h = np.zeros((2, 1, 64), dtype=np.float32)
c = np.zeros((2, 1, 64), dtype=np.float32)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def check_speech_prob(audio_chunk):
    global h, c
    if len(audio_chunk) < 512:
        return 0.0
    
    # Format inputs for ONNX Silero VAD v4
    input_data = np.expand_dims(audio_chunk[:512], axis=0).astype(np.float32)
    sr_data = np.array(16000, dtype=np.int64)
    
    ort_inputs = {
        'input': input_data,
        'sr': sr_data,
        'h': h,
        'c': c
    }
    
    out, h, c = ort_session.run(None, ort_inputs)
    return out[0][0]

async def process_audio_and_respond(audio_data, websocket):
    sample_rate = 16000
    temp_in = f"temp_onnx_{time.time()}.wav"
    sf.write(temp_in, audio_data, sample_rate, subtype="PCM_16")

    print("\n🔄 Transcribing via Whisper...")
    try:
        with open(temp_in, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_in, f.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
                prompt="English, Hindi, and Hinglish conversation.",
                temperature=0.0
            )

        if os.path.exists(temp_in):
            os.remove(temp_in)

        user_text = transcription.text.strip()
        if not user_text or len(user_text) < 2:
            return

        print(f"👂 USER: {user_text}")

        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Drishti, a hands-free smart-glasses AI assistant. Keep responses brief (1-2 sentences)."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7
        )

        reply_text = resp.choices[0].message.content
        print(f"🤖 DRISHTI: {reply_text}")

        voice = "hi-IN-SwaraNeural" if re.search(r'[\u0900-\u097F]', reply_text) else "en-IN-NeerjaNeural"
        clean_text = re.sub(r'[^\w\s,.?!|॥\u0900-\u097F]', '', reply_text, flags=re.UNICODE)
        
        temp_out = f"temp_out_{time.time()}.mp3"
        comm = edge_tts.Communicate(clean_text, voice)
        await comm.save(temp_out)

        with open(temp_out, "rb") as f:
            audio_bytes = f.read()

        if os.path.exists(temp_out):
            os.remove(temp_out)

        await websocket.send(json.dumps({"type": "text", "text": reply_text}))
        await websocket.send(audio_bytes)

    except Exception as e:
        print(f"❌ Processing Error: {e}")

async def audio_handler(websocket):
    global h, c
    print(f"\n📱 Phone Connected: {websocket.remote_address}")
    audio_buffer = []

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                speech_prob = check_speech_prob(audio_chunk)

                if speech_prob > 0.5:
                    audio_buffer.append(audio_chunk)
                elif len(audio_buffer) > 0:
                    complete_audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    h = np.zeros((2, 1, 64), dtype=np.float32)
                    c = np.zeros((2, 1, 64), dtype=np.float32)
                    await process_audio_and_respond(complete_audio, websocket)

    except websockets.exceptions.ConnectionClosed:
        print("📱 Phone Disconnected.")

async def main():
    local_ip = get_local_ip()
    port = 8765
    server = await websockets.serve(audio_handler, "0.0.0.0", port)
    print("\n" + "=" * 60)
    print(f" 👁️ DRISHTI ONNX BACKEND ONLINE (Zero PyTorch)")
    print(f" 🌐 WebSocket URL for your phone: ws://{local_ip}:{port}")
    print("=" * 60)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
