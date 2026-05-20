# Local Voice-to-Voice AI Assistant 🎙️🤖

An optimized, entirely local pipeline that enables real-time, low-latency spoken conversations with an AI assistant. It is configured out-of-the-box as a creative museum brainstorming companion for kids aged 8–12.

Built explicitly for high-efficiency CPU execution (tested on Fedora/Linux distributions).

## ✨ Features

*   **100% Local & Private:** Zero cloud dependencies or API keys required.
*   **Low-Latency Stream Chaining:** Sentences are spoken via text-to-speech (TTS) *while* the Large Language Model is still streaming the rest of its response.
*   **Smart Interruption Handling:** Suppresses native audio subsystem errors (ALSA) and safely manages background threads.
*   **Concise Memory Management:** Maintained message history window keeps local LLM generations fast and on-topic.

## 🛠️ Built With

*   **STT (Speech-to-Text):** [Faster-Whisper (`tiny.en`)](https://github.com/SYSTRAN/faster-whisper) — Fast, optimized implementation of OpenAI's Whisper model running in INT8 precision.
*   **LLM (Brain):** [Llama.cpp Python](https://github.com/abetlen/llama-cpp-python) running Unsloth's quantized `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.
*   **TTS (Text-to-Speech):** [Piper](https://github.com/rhasspy/piper) (`en_US-lessac-medium.onnx`) — An ultra-fast, local neural text-to-speech system.
*   **Audio Backend:** `speech_recognition` (capture) and `PipeWire` (`pw-play` engine for playback).

## 📦 Prerequisites & System Dependencies

Because this relies on system-level audio devices and specialized Python packages, make sure your environment has the following system dependencies installed (commands tailored for Fedora/Ubuntu):

```bash
# Fedora
sudo dnf install pipewire-utils alsa-lib portaudio-devel python3-devel

# Ubuntu/Debian
sudo apt-get install pipewire-bin libasound2 libasound2-dev portaudio19-dev python3-dev