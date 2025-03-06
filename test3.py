import numpy as np
import sounddevice as sd
import queue
import threading
import torch
import asyncio
import json
from typing import Deque, Tuple, Dict, Any
from collections import deque
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ollama import AsyncClient

# Enhanced Configuration
CONFIG = {
    "stt_model": "distil-whisper/distil-small.en",
    "primary_llm": "llama3.2",
    "validation_llm": "llama3.2",
    "context_window": 5,
    "word_trigger": 8,
    "silence_threshold": 0.0,  # RMS threshold for silence detection
    "silence_duration": 1.5,      # Seconds of silence to consider speech ended
    "sample_rate": 16000,
    "chunk_duration": 2,         # Processing chunk duration in seconds
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
}

def select_mic() -> int:
    devices = sd.query_devices()
    input_devices = [(i, dev) for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]
    if not input_devices:
        raise Exception("No input devices found.")
    print("Available audio input devices:")
    for i, dev in input_devices:
        print(f"{i}: {dev['name']}")
    try:
        selection = int(input("Select microphone device index: "))
        if selection in [i for i, _ in input_devices]:
            return selection
        else:
            print("Invalid selection. Using default device.")
            return None
    except Exception as e:
        print("Error in selection. Using default device.")
        return None

class StateManager:
    def __init__(self):
        self.state = "listening"
        self.lock = threading.Lock()
        self.context: Deque[Tuple[str, str]] = deque(maxlen=CONFIG["context_window"])
        self.current_dialog = []
        self.audio_history = []
        self.silence_counter = 0
        self.last_energy = 0.0

    def update_state(self, new_state: str):
        with self.lock:
            self.state = new_state

    def add_exchange(self, user_input: str, ai_response: str):
        self.context.append((user_input, ai_response))
        
    def update_audio_energy(self, energy: float):
        self.last_energy = energy
        if energy < CONFIG["silence_threshold"]:
            self.silence_counter += 1
        else:
            self.silence_counter = 0

    def speech_ended(self) -> bool:
        return self.silence_counter * (CONFIG["chunk_duration"] / 2) > CONFIG["silence_duration"]

class SpeechProcessor:
    def __init__(self):
        print("Initializing Enhanced Speech Processor...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            CONFIG["stt_model"],
            torch_dtype=CONFIG["torch_dtype"],
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(CONFIG["device"])
        processor = AutoProcessor.from_pretrained(CONFIG["stt_model"])
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            device=CONFIG["device"],
        )
        self.audio_queue = queue.Queue()
        self.buffer = np.empty((0,), dtype=np.float32)

    async def process_audio(self, state: StateManager):
        loop = asyncio.get_running_loop()
        while True:
            data = await loop.run_in_executor(None, self.audio_queue.get)
            energy = np.sqrt(np.mean(data**2))
            print(f"Audio energy: {energy:.5f}, min: {data.min()}, max: {data.max()}")

            state.update_audio_energy(energy)
            
            if energy < CONFIG["silence_threshold"]:
                continue
                
            self.buffer = np.concatenate([self.buffer, data])
            if len(self.buffer) >= CONFIG["sample_rate"] * CONFIG["chunk_duration"]:
                chunk = self.buffer[:CONFIG["sample_rate"] * CONFIG["chunk_duration"]]
                self.buffer = self.buffer[CONFIG["sample_rate"] * CONFIG["chunk_duration"] // 2:]
                result = await loop.run_in_executor(
                    None, lambda: self.asr({"array": chunk, "sampling_rate": CONFIG["sample_rate"]})
                )
                recognized_text = result.get("text", "").strip()
                if recognized_text:
                    print(f"STT: {recognized_text}")
                    yield recognized_text

class ValidationSystem:
    VALIDATION_PROMPT = """Analyze the conversation snippet considering it might be partial input. Return JSON with:
- "validation": "continue" (user is still speaking), "complete" (complete thought), or "junk"
- "corrected_text": cleaned version if needed
- "certainty": 0-1 confidence score
- "reasoning": brief explanation

Context: {context}
Input: {input}"""

    async def validate(self, text: str, context: str) -> Dict[str, Any]:
        prompt = self.VALIDATION_PROMPT.format(context=context, input=text)
        client = AsyncClient()
        try:
            response = await client.chat(
                model=CONFIG["validation_llm"],
                messages=[{"role": "user", "content": prompt}]
            )
            return self._parse_response(response.message.content, text)
        except Exception as e:
            print(f"Validation error: {e}")
            return {"validation": "continue", "corrected_text": text, "certainty": 0.5, "reasoning": str(e)}

    def _parse_response(self, response: str, original: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            required = ["validation", "corrected_text", "certainty", "reasoning"]
            return {k: data.get(k, original if k == "corrected_text" else "continue") for k in required}
        except:
            return {"validation": "continue", "corrected_text": original, "certainty": 0.5, "reasoning": "Parse error"}

class ConversationManager:
    PROMPT_TEMPLATE = """You're an AI tutor helping someone verbalize their thoughts. They're speaking in chunks (may be incomplete). 

Current Context:
{context}

Latest Input:
{input}

Respond with:
1. If input seems complete: Summary + insights
2. If incomplete: Brief encouragement to continue
3. If contradictory: Gentle correction
4. If unclear: Ask clarifying question

Keep responses very short and conversational."""

    async def respond(self, text: str, context: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(context=context, input=text)
        client = AsyncClient()
        try:
            response = await client.chat(
                model=CONFIG["primary_llm"],
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            return await self._stream_response(response)
        except Exception as e:
            print(f"Response error: {e}")
            return "Let me think about that..."

    async def _stream_response(self, response) -> str:
        return "".join([chunk["message"]["content"] async for chunk in response])

async def main_loop():
    state = StateManager()
    speech = SpeechProcessor()
    validator = ValidationSystem()
    conversation = ConversationManager()

    def audio_callback(indata, frames, time, status):
        if indata.any():
            speech.audio_queue.put(indata[:, 0].copy())

    # Ask user to select the microphone before starting the stream
    def audio_thread_fn(device):
        with sd.InputStream(
            samplerate=CONFIG["sample_rate"],
            channels=1,
            device=device,
            callback=audio_callback
        ) as stream:
            print("Audio stream started.")
            while True:
                sd.sleep(1000)  # Keep the stream alive

        # Ask user to select the microphone before starting the stream
    selected_device = select_mic()
    audio_thread = threading.Thread(target=lambda: audio_thread_fn(selected_device), daemon=True)
    audio_thread.start()

    current_input = []
    async for text in speech.process_audio(state):
        current_input.append(text)
        full_input = " ".join(current_input)
        
        if state.speech_ended():
            validation = await validator.validate(full_input, "\n".join([f"U: {u}\nA: {a}" for u, a in state.context]))
            
            if validation["validation"] == "complete":
                response = await conversation.respond(
                    validation["corrected_text"],
                    "\n".join([f"U: {u}\nA: {a}" for u, a in state.context])
                )
                state.add_exchange(full_input, response)
                current_input = []
                print(f"AI: {response}")
            elif validation["validation"] == "junk":
                current_input = []
            else:
                print("AI: Please continue...")
                
        elif len(current_input) >= 3:  # 3 chunks without silence
            validation = await validator.validate(full_input, "")
            if validation["certainty"] > 0.7:
                response = await conversation.respond(
                    validation["corrected_text"],
                    ""
                )
                print(f"AI: {response}")
                current_input = []

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("System shutdown.")
