# Local Voice-to-Voice AI Assistant 🎙️🤖

A real-time, low-latency local voice assistant built using LiveKit Agents, Piper TTS, and Silero VAD.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure your virtual environment is activated and you have downloaded your local Piper ONNX voice model into the `./piper/` directory.

### 2. Installation
Install all required Python dependencies and system packages:

```bash
pip install -r requirements.txt
```

### 3. Running the Pipeline

This project utilizes honcho to concurrently run the local Piper HTTP voice server and your LiveKit agent developer loop in a single terminal window.

To start the complete stack, run:
```bash
honcho start
```

[!WARNING]
If the the voice is squicky change the sample rate from piper_tts (.venv/lib64/python3.14/site-packages/livekit/plugins/piper_tts/tts.py)