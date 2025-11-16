import discord
from discord.ext import commands
import asyncio
import tomli
from pathlib import Path
from openai import AsyncOpenAI
import io
import sounddevice as sd
import soundfile as sf
import httpx
import numpy as np
import concurrent.futures
import threading

def is_speech(audio_chunk, threshold=500):
    """Detect if audio chunk contains speech based on volume"""
    # Calculate RMS (root mean square) of audio
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > threshold

# Load secrets
secrets_path = Path(".streamlit/secrets.toml")
with open(secrets_path, "rb") as f:
    secrets = tomli.load(f)

DISCORD_TOKEN = secrets["discord_voice"]
OPENAI_API_KEY = secrets["openai"]

client_oai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True
bot = commands.Bot(command_prefix='!', intents=intents)

conversation_history = []
is_listening = False

@bot.command()
async def startchat(ctx):
    """Start continuous conversation mode"""
    global is_listening
    
    if not ctx.voice_client:
        await ctx.send("❌ Use !join first!")
        return
    
    if is_listening:
        await ctx.send("⚠️ Already in chat mode!")
        return
    
    is_listening = True
    await ctx.send("🎤 **CHAT MODE ACTIVE!** Just start talking naturally. Say 'stop chat' or use !stopchat to end.")
    
    # Start the listening loop
    await continuous_listen(ctx)

import concurrent.futures
import threading

async def continuous_listen(ctx):
    """Continuously listen and respond"""
    global is_listening
    
    RATE = 16000
    CHUNK_DURATION = 1  # Record in 1-second chunks
    SILENCE_THRESHOLD = 500  # Adjust based on your mic
    SILENCE_DURATION = 2  # Seconds of silence before processing
    MIC_DEVICE = 2  # Your Razer mic
    
    recorded_audio = []
    silence_chunks = 0
    
    await ctx.send("👂 Listening...")
    
    # Use ThreadPoolExecutor to run blocking audio operations
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    while is_listening:
        # Run audio recording in a separate thread
        loop = asyncio.get_event_loop()
        
        def record_chunk():
            chunk = sd.rec(
                int(CHUNK_DURATION * RATE),
                samplerate=RATE,
                channels=1,
                dtype='int16',
                device=MIC_DEVICE
            )
            sd.wait()
            return chunk
        
        # Record without blocking the event loop
        chunk = await loop.run_in_executor(executor, record_chunk)
        
        # Check if this chunk has speech
        if is_speech(chunk, SILENCE_THRESHOLD):
            # Speech detected!
            recorded_audio.append(chunk)
            silence_chunks = 0
        else:
            # Silence detected
            silence_chunks += 1
            
            # If we have recorded audio AND hit silence threshold, process it
            if len(recorded_audio) > 0 and silence_chunks >= SILENCE_DURATION:
                await ctx.send("🔄 Processing...")
                
                # Combine all recorded chunks
                full_audio = np.concatenate(recorded_audio)
                
                # Save and transcribe
                sf.write('temp_recording.wav', full_audio, RATE)
                
                try:
                    # Transcribe
                    with open('temp_recording.wav', 'rb') as audio_file:
                        transcript = await client_oai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    
                    user_text = transcript.text
                    
                    # Check if user said "stop chat"
                    if "stop chat" in user_text.lower():
                        is_listening = False
                        await ctx.send("✋ Stopping chat mode!")
                        break
                    
                    await ctx.send(f"📝 You: {user_text}")
                    
                    # Get Llama response
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://192.168.12.209:11434/api/chat",
                            json={
                                "model": "llama3.2:3b",
                                "messages": conversation_history,
                                "stream": False
                            },
                            timeout=30.0
                        )
                        
                        llama_response = response.json()["message"]["content"]
                        conversation_history.append({"role": "assistant", "content": llama_response})
                        
                        await ctx.send(f"🤖 Bot: {llama_response}")
                        
                        # Speak response
                        tts_response = await client_oai.audio.speech.create(
                            model="tts-1",
                            voice="echo",
                            input=llama_response
                        )
                        
                        with open("bot_response.mp3", "wb") as f:
                            f.write(tts_response.content)
                        
                        audio_source = discord.FFmpegPCMAudio("bot_response.mp3")
                        ctx.voice_client.play(audio_source)
                        
                        # Wait for audio to finish playing
                        while ctx.voice_client.is_playing():
                            await asyncio.sleep(0.1)
                    
                    # Reset for next input
                    recorded_audio = []
                    silence_chunks = 0
                    await ctx.send("👂 Listening...")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error: {str(e)}")
                    recorded_audio = []
                    silence_chunks = 0
    
    executor.shutdown(wait=False)

@bot.command()
async def stopchat(ctx):
    """Stop continuous conversation mode"""
    global is_listening
    is_listening = False
    await ctx.send("✋ Chat mode stopped!")

@bot.event
async def on_ready():
    print(f'✅ {bot.user} is online and ready!')
    print(f'📋 Commands: !join, !leave, !listen')

@bot.command()
async def join(ctx):
    """Join your voice channel"""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"👋 Joined {channel.name}!")
    else:
        await ctx.send("❌ You need to be in a voice channel first!")

@bot.command()
async def continuous_listen(ctx):
    """Continuously listen and respond"""
    global is_listening
    
    RATE = 16000
    CHUNK_DURATION = 1  # Record in 1-second chunks
    SILENCE_THRESHOLD = 500  # Adjust based on your mic
    SILENCE_DURATION = 2  # Seconds of silence before processing
    MIC_DEVICE = 15
    
    recorded_audio = []
    silence_chunks = 0
    
    await ctx.send("👂 Listening...")
    
    while is_listening:
        # Record 1 second of audio
        chunk = sd.rec(
            int(CHUNK_DURATION * RATE),
            samplerate=RATE,
            channels=1,
            dtype='int16',
            device=MIC_DEVICE
        )
        sd.wait()
        
        # Check if this chunk has speech
        if is_speech(chunk, SILENCE_THRESHOLD):
            # Speech detected!
            recorded_audio.append(chunk)
            silence_chunks = 0
        else:
            # Silence detected
            silence_chunks += 1
            
            # If we have recorded audio AND hit silence threshold, process it
            if len(recorded_audio) > 0 and silence_chunks >= SILENCE_DURATION:
                await ctx.send("🔄 Processing...")
                
                # Combine all recorded chunks
                full_audio = np.concatenate(recorded_audio)
                
                # Save and transcribe
                sf.write('temp_recording.wav', full_audio, RATE)
                
                try:
                    # Transcribe
                    with open('temp_recording.wav', 'rb') as audio_file:
                        transcript = await client_oai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    
                    user_text = transcript.text
                    
                    # Check if user said "stop chat"
                    if "stop chat" in user_text.lower():
                        is_listening = False
                        await ctx.send("✋ Stopping chat mode!")
                        break
                    
                    await ctx.send(f"📝 You: {user_text}")
                    
                    # Get Llama response
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://192.168.12.209:11434/api/chat",
                            json={
                                "model": "llama3.2:3b",
                                "messages": conversation_history,
                                "stream": False
                            },
                            timeout=30.0
                        )
                        
                        llama_response = response.json()["message"]["content"]
                        conversation_history.append({"role": "assistant", "content": llama_response})
                        
                        await ctx.send(f"🤖 Bot: {llama_response}")
                        
                        # Speak response
                        tts_response = await client_oai.audio.speech.create(
                            model="tts-1",
                            voice="echo",
                            input=llama_response
                        )
                        
                        with open("bot_response.mp3", "wb") as f:
                            f.write(tts_response.content)
                        
                        audio_source = discord.FFmpegPCMAudio("bot_response.mp3")
                        ctx.voice_client.play(audio_source)
                        
                        # Wait for audio to finish playing
                        while ctx.voice_client.is_playing():
                            await asyncio.sleep(0.1)
                    
                    # Reset for next input
                    recorded_audio = []
                    silence_chunks = 0
                    await ctx.send("👂 Listening...")
                    
                except Exception as e:
                    await ctx.send(f"❌ Error: {str(e)}")
                    recorded_audio = []
                    silence_chunks = 0

@bot.command()
async def leave(ctx):
    """Leave the voice channel"""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("👋 Left the voice channel!")
    else:
        await ctx.send("❌ I'm not in a voice channel!")

# Run the bot
bot.run(DISCORD_TOKEN)
