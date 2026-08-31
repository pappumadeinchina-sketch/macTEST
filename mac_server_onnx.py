import os
import io
import json
import asyncio
import numpy as np
import soundfile as sf
import onnxruntime as ort
import websockets
from groq import Groq
import edge_tts

# ---------------------------------------------------------------------------
# 1. INITIALIZE CLIENTS & VAD MODEL
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

ONNX_MODEL_PATH = "silero_vad.onnx"

# Load ONNX Session
opts = ort.SessionOptions()
opts.inter_op_num_threads = 1
opts.intra_op_num_threads = 1
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, opts)

# Silero VAD v5 Recurrent State Tensor [2, 1, 128]
vad_state = np.zeros((2, 1, 128), dtype=np.float32)
sample_rate = np.array(16000, dtype=np.int64)

def reset_vad_state():
    global vad_state
    vad_state = np.zeros((2, 1, 128), dtype=np.float32)

def check_speech_prob(audio_chunk_float32: np.ndarray) -> float:
    """
    Evaluates a 512-sample float32 audio chunk using Silero VAD v5 ONNX model.
    """
    global vad_state
    
    if audio_chunk_float32.ndim == 1:
        input_tensor = np.expand_dims(audio_chunk_float32, axis=0)
    else:
        input_tensor = audio_chunk_float32

    # Flexible fallback: handles both v5 ('state') and legacy v4 ('h', 'c')
    input_names = [inp.name for inp in ort_session.get_inputs()]
    
    if "state" in input_names:
        ort_inputs = {
            "input": input_tensor.astype(np.float32),
            "sr": sample_rate,
            "state": vad_state
        }
        out, vad_state = ort_session.run(None, ort_inputs)
    else:
        # Legacy v4 fallback if old model is supplied
        h = vad_state[:, :, :64]
        c = vad_state[:, :, 64:]
        ort_inputs = {
            "input": input_tensor.astype(np.float32),
            "sr": sample_rate,
            "h": h,
            "c": c
        }
        out, h, c = ort_session.run(None, ort_inputs)
        vad_state = np.concatenate([h, c], axis=-1)

    return float(out[0][0])


# ---------------------------------------------------------------------------
# 2. AUDIO PROCESSING & AI PIPELINE
# ---------------------------------------------------------------------------
async def process_user_speech(pcm_bytes: bytes, websocket):
    """
    Transcribes audio with Groq Whisper -> Generates LLM response -> Synthesizes Speech with Edge-TTS
    """
    if not groq_client:
        print("❌ Error: GROQ_API_KEY environment variable is missing on Render!")
        return

    try:
        # 1. Convert PCM16 bytes to WAV format for Whisper API
        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        wav_io = io.BytesIO()
        sf.write(wav_io, pcm_data, 16000, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        wav_io.name = "input.wav"

        # 2. Speech-to-Text via Groq Whisper
        transcription = groq_client.audio.transcriptions.create(
            file=(wav_io.name, wav_io.read()),
            model="whisper-large-v3",
            response_format="text"
        )
        user_text = transcription.strip()
        if not user_text:
            return

        print(f"🗣️ User: {user_text}")
        await websocket.send(json.dumps({"type": "text", "text": user_text}))

        # 3. LLM Generation via Groq Llama3
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Drishti, a fast and helpful AI voice assistant for smart glasses. Keep responses brief, conversational, and direct (1-2 sentences maximum)."},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=150
        )
        ai_response = chat_completion.choices[0].message.content.strip()
        print(f"🤖 Drishti: {ai_response}")
        
        await websocket.send(json.dumps({"type": "text", "text": ai_response}))

        # 4. Text-to-Speech via Edge-TTS
        communicate = edge_tts.Communicate(ai_response, "en-US-AvaNeural")
        mp3_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        # 5. Send raw MP3 binary payload back to client
        await websocket.send(mp3_bytes)

    except Exception as e:
        print(f"❌ Error in speech pipeline: {e}")


# ---------------------------------------------------------------------------
# 3. WEBSOCKET CLIENT HANDLER
# ---------------------------------------------------------------------------
async def handle_client(websocket):
    print("🟢 Client connected to Drishti Voice Server")
    reset_vad_state()

    audio_buffer = bytearray()
    speech_frames = bytearray()
    is_speaking = False
    silence_counter = 0

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_buffer.extend(message)

                # Process buffer in 512-sample frames (1024 bytes for Int16)
                while len(audio_buffer) >= 1024:
                    frame_bytes = audio_buffer[:1024]
                    del audio_buffer[:1024]

                    # Convert Int16 PCM to Float32 [-1.0, 1.0]
                    pcm_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
                    pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

                    # Evaluate speech probability
                    speech_prob = check_speech_prob(pcm_float32)

                    if speech_prob > 0.5:
                        if not is_speaking:
                            is_speaking = True
                            print("🎤 Speech detected...")
                        speech_frames.extend(frame_bytes)
                        silence_counter = 0
                    else:
                        if is_speaking:
                            speech_frames.extend(frame_bytes)
                            silence_counter += 1
                            
                            # ~800ms silence threshold (25 frames of 32ms)
                            if silence_counter > 25:
                                print("🤫 Silence detected, processing query...")
                                is_speaking = False
                                silence_counter = 0
                                
                                # Send collected speech to AI pipeline
                                recorded_bytes = bytes(speech_frames)
                                speech_frames.clear()
                                reset_vad_state()
                                
                                await process_user_speech(recorded_bytes, websocket)

    except websockets.exceptions.ConnectionClosed:
        print("🔴 Client disconnected")
    except Exception as e:
        print(f"❌ Connection error: {e}")


# ---------------------------------------------------------------------------
# 4. SERVER ENTRY POINT
# ---------------------------------------------------------------------------
async def main():
    # Dynamic port routing for Render environment
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Drishti WebSocket Server on port {port}...")

    async with websockets.serve(
        handle_client,
        "0.0.0.0",
        port,
        ping_interval=20,
        ping_timeout=20
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
