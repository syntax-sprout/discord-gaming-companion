# Discord Gaming Voice Companion 🎮🎤

A Discord bot that joins your voice channel, listens to you talk, transcribes your speech with Whisper, sends your message to a local Llama model, then replies back **in text and voice** in real time.

Think: lightweight AI gaming buddy / co-pilot that can hang out in call with you.

---

## ✨ Features

- 🎧 Joins your Discord voice channel and continuously listens
- 🗣 Detects speech vs silence using RMS (volume) thresholds
- ✍️ Transcribes speech to text using OpenAI Whisper
- 🤖 Sends conversation history to a local `llama3.2:3b` chat endpoint
- 💬 Replies in text **and** generates TTS audio using `tts-1`
- 🧠 Maintains short conversation history with a configurable system prompt
- ⚙️ Runtime config via Discord commands (no restarts needed)
- 🧪 Built-in mic testing and device listing for easier setup

---

## 🧱 Tech Stack

- Python
- [discord.py](https://discordpy.readthedocs.io/)
- `sounddevice` + `soundfile` for audio recording
- OpenAI API (Whisper + TTS)
- Local Llama endpoint via HTTP (`httpx`)
- NumPy for RMS / silence detection

---

## 📦 Requirements

- Python 3.10+ (recommended)
- FFmpeg installed and available in your PATH
- A working microphone on the machine running the bot
- OpenAI API key
- Discord bot token
- A local Llama chat endpoint (example: Ollama at `http://localhost:11434/api/chat`)

---

## 🔐 Secrets Setup

Secrets are loaded from:

```text
.streamlit/secrets.toml
