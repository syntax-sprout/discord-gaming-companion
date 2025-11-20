Discord Gaming Voice Companion 🎮🗣️

A real‑time AI co‑pilot for Discord voice chat. This Python bot joins your voice channel, listens to your speech, transcribes everything using OpenAI Whisper, and replies instantly—both in text and with synthesized speech—using a local Llama model and TTS. Whether you’re gaming, chilling, or testing AI voice assistants, this bot is your lightweight, always‑on, AI gaming buddy.

✨ Features

🎧 Joins your Discord voice channel and continuously listens

🗣 Speech vs. silence detection (RMS volume threshold)

✍️ Real‑time speech transcription (OpenAI Whisper)

🤖 Sends messages to a local Llama endpoint (Ollama or compatible)

💬 AI replies in text & generates TTS audio (OpenAI TTS)

🧠 Maintains short conversation history (configurable system prompt)

⚙️ Runtime config via Discord commands (no restarts needed)

🧪 Mic/device testing & setup commands for smoother onboarding

🧱 Tech Stack

Python 3.10+

discord.py

sounddevice + soundfile (audio recording)

OpenAI API (Whisper, TTS)

Local Llama endpoint (HTTP, e.g. Ollama)

numpy (RMS/silence detection)

📦 Requirements

Python 3.10+ (recommended)

FFmpeg (in your PATH)

Working microphone on host machine

OpenAI API key

Discord bot token

Local Llama chat endpoint (e.g. Ollama
 at http://localhost:11434/api/chat)

🔐 Secrets Setup

Secrets are loaded from .streamlit/secrets.toml.
