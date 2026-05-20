import os
import io
import time
import queue
import threading
import subprocess
import ctypes
import wave
from llama_cpp import Llama
from faster_whisper import WhisperModel
import speech_recognition as sr
from piper import PiperVoice

# --- 1. SILENCE ALSA ERRORS ---
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

# --- 2. INITIALIZE MODELS ---
print("--- Initializing Final Optimized Pipeline ---")

llm = Llama.from_pretrained(
    repo_id="unsloth/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=4,
    n_gpu_layers=0, 
    verbose=False
)

stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4)

PIPER_MODEL_PATH = "en_US-lessac-medium.onnx"
voice = PiperVoice.load(PIPER_MODEL_PATH)

# --- 3. SHUTDOWN & AUDIO TOOLS ---
speech_queue = queue.Queue()
ai_is_speaking = threading.Event()
running = True # Global flag to stop threads

def speech_worker():
    """Plays AI audio and checks for the running flag."""
    while running:
        try:
            # Short timeout allows the thread to check the 'running' flag periodically
            text = speech_queue.get(timeout=1) 
        except queue.Empty:
            continue

        if text is None: break
        
        ai_is_speaking.set()
        buffer = io.BytesIO()
        
        try:
            with wave.open(buffer, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            
            buffer.seek(0)
            audio_payload = buffer.read()
            
            subprocess.run(
                ['pw-play', '--channels=1', '--rate=22050', '--format=s16le', '-'],
                input=audio_payload,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
        except Exception as e:
            if running: print(f"\n[Speech Error]: {e}")
        finally:
            buffer.close()
            ai_is_speaking.clear()
            speech_queue.task_done()

# Start worker
worker_thread = threading.Thread(target=speech_worker, daemon=True)
worker_thread.start()

# --- 4. LISTENING ENGINE ---
r = sr.Recognizer()

def listen():
    r.pause_threshold = 0.6
    r.non_speaking_duration = 0.3
    try:
        with sr.Microphone() as source:
            print("\n[Listening...]", end="", flush=True)
            # phrase_time_limit prevents it from hanging forever if it hears noise
            audio = r.listen(source, timeout=None, phrase_time_limit=10)
            print(" Processing...", end="", flush=True)
        
        audio_data = io.BytesIO(audio.get_wav_data())
        segments, _ = stt_model.transcribe(audio_data, beam_size=1)
        return "".join([s.text for s in segments]).strip()
    except Exception:
        return ""

# --- 5. MAIN INTERACTION LOOP ---
history = [{"role": "system", "content": "You are a creative museum brainstormer for kids 8-12. Use 2 punchy sentences and always end with a question."}]

print("\n--- Assistant Online (Press Ctrl+C to Exit) ---")

try:
    while running:
        # 1. Wait if AI is talking
        while ai_is_speaking.is_set():
            time.sleep(0.1)
        
        # 2. Capture User Voice
        user_text = listen()
        if not user_text or len(user_text) < 3:
            continue
        
        print(f"\nYou: {user_text}")
        history.append({"role": "user", "content": user_text})
        
        # Keep history concise
        if len(history) > 5:
            history = [history[0]] + history[-4:]

        # 3. Generate LLM Response
        print("AI: ", end="", flush=True)
        response_stream = llm.create_chat_completion(messages=history, stream=True)

        full_response = ""
        sentence_buffer = ""

        for chunk in response_stream:
            if not running: break # Break stream if exiting
            if 'content' in chunk['choices'][0]['delta']:
                token = chunk['choices'][0]['delta']['content']
                print(token, end='', flush=True)
                
                full_response += token
                sentence_buffer += token

                if any(punc in token for punc in [".", "!", "?", "\n", ","]):
                    clean_sent = sentence_buffer.strip()
                    if len(clean_sent) > 1:
                        speech_queue.put(clean_sent)
                        sentence_buffer = ""

        history.append({"role": "assistant", "content": full_response})

except KeyboardInterrupt:
    print("\n\n[Shutting down...] Goodbye!")
    running = False
    # A tiny sleep to let the terminal clean up
    time.sleep(0.1)
    # This is the "Heavy Hammer" that forces Fedora to kill the process and all threads
    os._exit(0)