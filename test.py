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

# Configuration
CONFIG = {
    "stt_model": "distil-whisper/distil-small.en",
    "primary_llm": "llama3.2",
    "validation_llm": "llama3.2",
    "context_window": 5,  # Number of exchanges to retain
    "word_trigger": 10,   # Words needed to trigger validation
    "sample_rate": 16000,
    "chunk_duration": 3,  # seconds per chunk
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
}

def choose_input_device() -> int:
    devices = sd.query_devices()
    input_devices = [idx for idx, dev in enumerate(devices) if dev["max_input_channels"] > 0]
    print("Available input devices:")
    for idx in input_devices:
        print(f"{idx}: {devices[idx]['name']}")
    try:
        selected_index = int(input("Enter the device index to use: "))
    except ValueError:
        print("Invalid input. Using default device.")
        return sd.default.device[0]
    if selected_index not in input_devices:
        print("Invalid device index selected. Using default device.")
        return sd.default.device[0]
    return selected_index

class StateManager:
    def __init__(self):
        self.state = "listen"
        self.lock = threading.Lock()
        self.context: Deque[Tuple[str, str]] = deque(maxlen=CONFIG["context_window"])
        self.current_input = ""

    def update_state(self, new_state: str):
        print(f"State changed from {self.state} to {new_state}")
        self.state = new_state

    def add_exchange(self, user_input: str, ai_response: str):
        self.context.append((user_input, ai_response))
        print(f"Exchange added. Total exchanges: {len(self.context)}")

class SpeechProcessor:
    def __init__(self):
        print("Initializing SpeechProcessor...")
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
            device=CONFIG["device"],
        )
        self.audio_queue = queue.Queue()
        self.buffer = np.empty((0,), dtype=np.float32)

    async def process_audio(self):
        loop = asyncio.get_running_loop()
        while True:
            data = await loop.run_in_executor(None, self.audio_queue.get)
            self.buffer = np.concatenate([self.buffer, data])
            if self.buffer.shape[0] >= CONFIG["sample_rate"] * CONFIG["chunk_duration"]:
                chunk_samples = CONFIG["sample_rate"] * CONFIG["chunk_duration"]
                chunk = self.buffer[:chunk_samples]
                shift = int(chunk_samples * 0.9)
                self.buffer = self.buffer[shift:]
                result = self.asr({"array": chunk, "sampling_rate": CONFIG["sample_rate"]})
                recognized_text = result.get("text", "").strip()
                print(f"ASR recognized: {recognized_text}")
                yield recognized_text

class ValidationSystem:
    VALIDATION_PROMPT = (
        "Analyze the conversation snippet and return ONLY valid JSON with the following keys:\n"
        '- "validation": either "talk" or "listen"\n'
        '- "corrected_text": the corrected version if needed\n'
        '- "reasoning": a brief explanation\n\n'
        "Return ONLY JSON with no additional text.\n\n"
        "Current Context: {context}\n"
        "New Input: {input}\n\n"
        "Consider: factual accuracy, context alignment, completeness, ambiguity, and contradictions."
    )
    
    async def validate_input(self, text: str, context: str) -> Dict[str, Any]:
        prompt = self.VALIDATION_PROMPT.format(context=context, input=text)
        print("Sending validation prompt:")
        print(prompt)
        client = AsyncClient()
        try:
            response = await client.chat(
                model=CONFIG["validation_llm"],
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.message.content
            if not content.strip():
                print("Received empty validation response. Defaulting to listen mode.")
                return {"validation": "listen", "corrected_text": text, "reasoning": "Empty response"}
            print(f"Received validation response: {content}")
            return self._parse_response(content, text)
        except Exception as e:
            print(f"Exception during validation: {e}")
            return {"validation": "listen", "corrected_text": text, "reasoning": str(e)}

    def _parse_response(self, response: str, original_text: str) -> Dict[str, Any]:
        if not response.strip():
            print("Empty response in _parse_response. Defaulting to listen mode.")
            return {"validation": "listen", "corrected_text": original_text, "reasoning": "Empty response"}
        try:
            parsed = json.loads(response)
            print("Parsed validation response:", parsed)
            if "validation" not in parsed or "corrected_text" not in parsed or "reasoning" not in parsed:
                print("Missing keys in parsed response. Defaulting.")
                return {"validation": "listen", "corrected_text": original_text, "reasoning": "Incomplete response"}
            return parsed
        except Exception as e:
            print("Error parsing validation response. Raw response:", response)
            print("Error:", e)
            return {"validation": "listen", "corrected_text": original_text, "reasoning": "Parsing error"}

class ConversationManager:
    PROMPT_TEMPLATE = (
        "You're an AI bot, and the user is telling you things they've"
        " learned to validate their understanding. Your role is to help"
        "them by confirming what's correct, correcting any mistakes, "
        "filling in gaps, and providing additional insights where needed."
         " The user might be preparing for an exam, studying a topic, or "
         "just reinforcing their knowledge.\nCurrent conversation:\n"
        "{context}\n"
        "User: {input}\n"
        "Assistant:"
    )

    async def generate_response(self, text: str, context: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(context=context, input=text)
        print("Sending conversation prompt:")
        print(prompt)
        client = AsyncClient()
        try:
            response = await client.chat(
                model=CONFIG["primary_llm"],
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            full_response = await self._stream_response(response)
            print(f"Full generated response: {full_response}")
            return full_response
        except Exception as e:
            print(f"Exception during response generation: {e}")
            return "I'm sorry, I encountered an error generating a response."

    async def _stream_response(self, response) -> str:
        full_response = []
        async for chunk in response:
            content = chunk["message"]["content"]
            print(content, end='', flush=True)
            full_response.append(content)
        return "".join(full_response)

async def main_loop():
    state_manager = StateManager()
    speech_processor = SpeechProcessor()
    validator = ValidationSystem()
    conversation = ConversationManager()

    def audio_callback(indata, frames, time, status):
        if status:
            print("Audio stream status:", status)
        speech_processor.audio_queue.put(indata[:, 0].copy())

    def start_audio_stream():
        try:
            selected_device = choose_input_device()
            with sd.InputStream(device=selected_device,
                                samplerate=CONFIG["sample_rate"],
                                channels=1,
                                callback=audio_callback):
                print(f"Audio stream started on device {selected_device}. Listening...")
                while True:
                    sd.sleep(1000)
        except Exception as e:
            print("Exception in audio stream:", e)

    audio_thread = threading.Thread(target=start_audio_stream, daemon=True)
    audio_thread.start()

    async for text in speech_processor.process_audio():
        if text:
            state_manager.current_input += " " + text
            print(f"Current accumulated input: {state_manager.current_input.strip()}")
            word_count = len(state_manager.current_input.split())
            if word_count >= CONFIG["word_trigger"]:
                context = "\n".join([f"User: {u}\nAI: {a}" for u, a in state_manager.context])
                print("Validating input with context:")
                print(context)
                validation = await validator.validate_input(state_manager.current_input, context)
                print("Validation result:", validation)
                if validation.get("validation") == "talk":
                    state_manager.update_state("talk")
                    response = await conversation.generate_response(
                        validation.get("corrected_text", state_manager.current_input),
                        context
                    )
                    state_manager.add_exchange(state_manager.current_input, response)
                    state_manager.current_input = ""
                    state_manager.update_state("listen")
                else:
                    print("Validation indicates 'listen' mode. Continuing to accumulate input.")
                    
if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
