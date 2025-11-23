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
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > threshold

# Load secrets
secrets_path = Path(".streamlit/secrets.toml")
with open(secrets_path, "rb") as f:
    secrets = tomli.load(f)

DISCORD_TOKEN = secrets["discord_voice"]
OPENAI_API_KEY = secrets["openai"]
KINDROID_API_KEY = secrets["kindroid"]
LOTTIE_AI_ID = secrets["lottie_ai_id"]

client_oai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True
bot = commands.Bot(command_prefix='!', intents=intents)

conversation_history = []
is_listening = False

# Configuration that can be changed at runtime
config = {
    'mic_device': 2,
    'silence_threshold': 500,
    'silence_duration': 2,
    'kindroid_timeout': 30,
    'max_history': 10,
}

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
    await ctx.send("🎤 **CHAT MODE ACTIVE!** Lottie is ready to chat! Just start talking naturally. Say 'stop chat' or use !stopchat to end.")
    
    # Start the listening loop
    await continuous_listen(ctx)

async def continuous_listen(ctx):
    """Continuously listen and respond"""
    global is_listening, conversation_history

    RATE = 16000
    CHUNK_DURATION = 1

    recorded_audio = []
    silence_chunks = 0

    await ctx.send("👂 Listening...")

    # Use ThreadPoolExecutor to run blocking audio operations
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    while is_listening:
        SILENCE_THRESHOLD = config['silence_threshold']
        SILENCE_DURATION = config['silence_duration']
        MIC_DEVICE = config['mic_device']

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
        chunk_rms = np.sqrt(np.mean(chunk**2))

        # Check if this chunk has speech
        if is_speech(chunk, SILENCE_THRESHOLD):
            print(f"🎤 Speech detected! RMS: {chunk_rms:.2f} (threshold: {SILENCE_THRESHOLD})")
            recorded_audio.append(chunk)
            silence_chunks = 0
        else:
            silence_chunks += 1
            print(f"🔇 Silence {silence_chunks}/{SILENCE_DURATION} - RMS: {chunk_rms:.2f} (threshold: {SILENCE_THRESHOLD})")

            if len(recorded_audio) > 0 and silence_chunks >= SILENCE_DURATION:
                print(f"✅ Processing {len(recorded_audio)} chunks of audio...")
                await ctx.send("🔄 Processing...")
                
                # Combine all recorded chunks
                full_audio = np.concatenate(recorded_audio)
                
                # Save and transcribe
                sf.write('temp_recording.wav', full_audio, RATE)
                
                try:
                    print("🔄 Step 1: Transcribing audio...")
                    with open('temp_recording.wav', 'rb') as audio_file:
                        transcript = await client_oai.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )

                    user_text = transcript.text
                    print(f"✅ Transcription: {user_text}")

                    # Check if user said "stop chat"
                    if "stop chat" in user_text.lower():
                        is_listening = False
                        await ctx.send("✋ Stopping chat mode!")
                        break

                    await ctx.send(f"📝 You: {user_text}")

                    # Get Lottie's response from Kindroid
                    print(f"🔄 Step 2: Getting Lottie's response...")

                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                "https://api.kindroid.ai/v1/send-message",
                                headers={
                                    "Authorization": f"Bearer {KINDROID_API_KEY}",
                                    "Content-Type": "application/json"
                                },
                                json={
                                    "ai_id": LOTTIE_AI_ID,
                                    "message": user_text
                                },
                                timeout=config['kindroid_timeout']
                            )

                            response.raise_for_status()
                            lottie_response = response.text
                            
                            print(f"✅ Lottie's response: {lottie_response[:100]}...")
                            await ctx.send(f"💅 Lottie: {lottie_response}")

                            # Speak response with OpenAI TTS
                            print("🔄 Step 3: Generating TTS...")
                            tts_response = await client_oai.audio.speech.create(
                                model="tts-1",
                                voice="nova",
                                input=lottie_response
                            )

                            with open("bot_response.mp3", "wb") as f:
                                f.write(tts_response.content)

                            print("🔄 Step 4: Playing audio...")
                            audio_source = discord.FFmpegPCMAudio("bot_response.mp3")
                            ctx.voice_client.play(audio_source)

                            # Wait for audio to finish playing
                            while ctx.voice_client.is_playing():
                                await asyncio.sleep(0.1)

                            print("✅ Audio playback complete")

                        # Reset for next input
                        recorded_audio = []
                        silence_chunks = 0
                        await ctx.send("👂 Listening...")

                    except httpx.ReadTimeout:
                        print(f"⏱️ Kindroid API timeout after {config['kindroid_timeout']}s")
                        await ctx.send(f"⏱️ Lottie took too long to respond (>{config['kindroid_timeout']}s).")
                        await ctx.send("👂 Listening... (continuing after timeout)")
                        recorded_audio = []
                        silence_chunks = 0

                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    error_traceback = traceback.format_exc()

                    # Log full error to console
                    print(f"❌ ERROR OCCURRED:")
                    print(error_traceback)

                    # Send truncated error to Discord (max 2000 chars)
                    if len(error_msg) > 1900:
                        error_msg = error_msg[:1900] + "..."

                    await ctx.send(f"❌ Error: {error_msg}")
                    await ctx.send("👂 Listening... (continuing after error)")

                    recorded_audio = []
                    silence_chunks = 0
    
    executor.shutdown(wait=False)

@bot.command()
async def stopchat(ctx):
    """Stop continuous conversation mode"""
    global is_listening
    is_listening = False
    await ctx.send("✋ Chat mode stopped!")

@bot.command()
async def setmic(ctx, device_id: int):
    """Set the microphone device. Usage: !setmic [device_id]"""
    config['mic_device'] = device_id
    if is_listening:
        await ctx.send(f"✅ Microphone device set to: {device_id} (will take effect on next recording chunk)")
    else:
        await ctx.send(f"✅ Microphone device set to: {device_id}")

@bot.command()
async def setthreshold(ctx, threshold: int):
    """Set the silence threshold. Usage: !setthreshold [value]"""
    config['silence_threshold'] = threshold
    if is_listening:
        await ctx.send(f"✅ Silence threshold set to: {threshold} (active immediately)")
    else:
        await ctx.send(f"✅ Silence threshold set to: {threshold}")

@bot.command()
async def settimeout(ctx, seconds: int):
    """Set the Kindroid API timeout. Usage: !settimeout [seconds]"""
    if seconds < 10:
        await ctx.send("❌ Timeout must be at least 10 seconds")
        return
    config['kindroid_timeout'] = seconds
    await ctx.send(f"✅ Kindroid timeout set to: {seconds} seconds")

@bot.command()
async def config_show(ctx):
    """Show current configuration"""
    result = "⚙️ **Current Configuration:**\n"
    result += f"Microphone Device: {config['mic_device']}\n"
    result += f"Silence Threshold: {config['silence_threshold']}\n"
    result += f"Silence Duration: {config['silence_duration']} seconds\n"
    result += f"Kindroid Timeout: {config['kindroid_timeout']} seconds\n"
    await ctx.send(result)

@bot.command()
async def devices(ctx):
    """List all available audio input devices"""
    devices_list = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices_list):
        if device['max_input_channels'] > 0:
            input_devices.append(f"**{i}**: {device['name']} (Channels: {device['max_input_channels']}, SR: {device['default_samplerate']})")

    if input_devices:
        await ctx.send("🎤 **Available Input Devices:**\n" + "\n".join(input_devices))
    else:
        await ctx.send("❌ No input devices found!")

@bot.command()
async def testmic(ctx, device_id: int = 2, duration: int = 5):
    """Test microphone and show audio levels. Usage: !testmic [device_id] [duration]"""
    await ctx.send(f"🎙️ Testing device {device_id} for {duration} seconds...")

    RATE = 16000
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    def record_test():
        recording = sd.rec(
            int(duration * RATE),
            samplerate=RATE,
            channels=1,
            dtype='int16',
            device=device_id
        )
        sd.wait()
        return recording

    try:
        audio = await loop.run_in_executor(executor, record_test)

        # Calculate RMS for the whole recording
        rms = np.sqrt(np.mean(audio**2))
        max_val = np.max(np.abs(audio))

        # Calculate RMS for 1-second chunks
        chunk_size = RATE
        chunk_rms_values = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            chunk_rms = np.sqrt(np.mean(chunk**2))
            chunk_rms_values.append(chunk_rms)

        avg_chunk_rms = np.mean(chunk_rms_values)
        max_chunk_rms = np.max(chunk_rms_values)

        result = f"📊 **Microphone Test Results:**\n"
        result += f"Overall RMS: {rms:.2f}\n"
        result += f"Max Value: {max_val}\n"
        result += f"Average Chunk RMS: {avg_chunk_rms:.2f}\n"
        result += f"Max Chunk RMS: {max_chunk_rms:.2f}\n"
        result += f"Chunk RMS values: {[f'{v:.2f}' for v in chunk_rms_values]}\n\n"
        result += f"💡 **Recommended Threshold:** {avg_chunk_rms * 0.3:.2f} - {avg_chunk_rms * 0.7:.2f}\n"
        result += f"Current threshold: {config['silence_threshold']}"

        await ctx.send(result)

        # Save the test recording
        sf.write('test_recording.wav', audio, RATE)
        await ctx.send("💾 Saved as test_recording.wav")

    except Exception as e:
        await ctx.send(f"❌ Error testing microphone: {str(e)}")
    finally:
        executor.shutdown(wait=False)

@bot.event
async def on_ready():
    print(f'✅ {bot.user} is online and ready!')
    print(f'💅 Lottie (Kindroid) is connected!')
    print(f'📋 Commands: !join, !leave, !startchat, !stopchat')
    print(f'🔧 Debug: !devices, !testmic [device_id] [duration], !config_show')
    print(f'⚙️  Config: !setmic, !setthreshold, !settimeout')

@bot.command()
async def join(ctx):
    """Join your voice channel"""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"👋 Joined {channel.name}! Lottie is ready to chat!")
    else:
        await ctx.send("❌ You need to be in a voice channel first!")

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
