import numpy as np
import sounddevice as sd
import queue
import threading
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from concurrent.futures import ThreadPoolExecutor

def choose_input_device():
    devices = sd.query_devices()
    input_indices = []
    print("Available input devices:")
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"{idx}: {dev['name']}")
            input_indices.append(idx)
    selected_index = int(input("Enter the device index to use: "))
    if selected_index not in input_indices:
        print("Invalid device index selected. Exiting.")
        exit(1)
    return selected_index

# Device and model settings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "distil-whisper/distil-small.en"

# Load model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create ASR pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# Audio stream settings
SAMPLERATE = 16000
CHUNK_DURATION = 3  # seconds per chunk
CHUNK_SAMPLES = SAMPLERATE * CHUNK_DURATION

# Thread-safe queue and buffer for audio samples
audio_queue = queue.Queue()
buffer = np.empty((0,), dtype=np.float32)

def audio_callback(indata, frames, time, status):
    if status:
        print("Stream status:", status)
    # Put mono audio data into the queue
    audio_queue.put(indata[:, 0].copy())

def process_chunk(chunk):
    result = asr({"array": chunk, "sampling_rate": SAMPLERATE})
    print("Recognized:", result["text"].strip())

def process_audio(executor):
    global buffer
    while True:
        data = audio_queue.get()
        buffer = np.concatenate([buffer, data])
        # Process a chunk when we have enough samples
        if buffer.shape[0] >= CHUNK_SAMPLES:
            chunk = buffer[:CHUNK_SAMPLES]
            # For 10% overlap, remove 90% of the chunk from the buffer,
            # so that the last 10% remains.
            shift = int(CHUNK_SAMPLES * 0.9)
            buffer = buffer[shift:]
            executor.submit(process_chunk, chunk)

def main():
    selected_device = choose_input_device()
    
    # Create a thread pool to process chunks concurrently
    executor = ThreadPoolExecutor(max_workers=4)
    
    # Start the audio processing thread
    threading.Thread(target=process_audio, args=(executor,), daemon=True).start()
    
    # Open the input stream with the selected device
    with sd.InputStream(device=selected_device, samplerate=SAMPLERATE, channels=1, callback=audio_callback):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopping...")
            executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
