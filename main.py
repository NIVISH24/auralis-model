import numpy as np
import sounddevice as sd
import queue
import threading
from transformers import pipeline

# Set your sampling rate and chunk duration (in seconds)
SAMPLERATE = 16000
CHUNK_DURATION = 10  # seconds
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLERATE

# Initialize the ASR pipeline (chunk_length_s can be set to match your chunk duration)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-medium", chunk_length_s=CHUNK_DURATION)

audio_queue = queue.Queue()
buffer = np.empty((0,), dtype=np.float32)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Flatten and put the recorded audio into the queue
    audio_queue.put(indata[:, 0].copy())

def process_audio():
    global buffer
    while True:
        # Get audio data from the queue
        data = audio_queue.get()
        buffer = np.concatenate([buffer, data])
        # When we have enough samples, process the chunk
        if buffer.shape[0] >= CHUNK_SAMPLES:
            chunk = buffer[:CHUNK_SAMPLES]
            buffer = buffer[CHUNK_SAMPLES:]  # retain leftover samples
            # Transcribe the chunk (input can be a numpy array; samplerate is inferred)
            result = asr({"array": chunk, "sampling_rate": SAMPLERATE})
            print("Recognized:", result["text"].strip())

def main():
    # Start the processing thread
    threading.Thread(target=process_audio, daemon=True).start()
    # Open the input stream from the microphone
    with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=audio_callback):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
