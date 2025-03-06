import torch
import asyncio
from ollama import AsyncClient

# Global history to be replaced by database later
history = [{
    "role": "system",
    "content": """For every 10 words typed or after a 4-second debounce, the user's text will be sent to you 
            along with a summarized message history. Respond in the following JSON format:
{
  "summarized_message_history": "<summary>",
  "user_state": "<state>",
  "response": "<output>"
}
Summarize the conversation like:

'The user has discussed [topic] and their statements are factually correct.'
'The last response from the user was: "[latest input]."'
If the user needs to provide more input, include the full text in summarized_message_history instead of a summary.
User State Determination
Analyze the last 10 words to classify the user's state:

Factually Incorrect – If the user makes a mistake, identify the incorrect information.
Needs More Input – If the user has not completed their thought, set summarized_message_history to the full text and wait for more input.
Speech-to-Text (STT) Junk Output – If the input appears garbled or nonsensical.
STT Misinterpretation – If the input suggests a misheard word.
Response Strategy
If the user is incorrect, provide a correction.
If the user needs to type more, respond with "Okay, go on..." while retaining the full text in summarized_message_history.
If the input is junk or misheard, suggest a rephrase or clarify potential errors.
You are assisting the user in reinforcing their knowledge, revising concepts, and preparing for exams by evaluating their understanding as they explain topics to you. Don't ask the user what they would like to explore further, instead let them talk about the topic. and if you feel like they left out any, let them know."""
}]

CONFIG = {
    "stt_model": "distil-whisper/distil-small.en",
    "primary_llm": "llama3.2",
    "validation_llm": "llama3.2",
    "context_window": 5,
    "word_trigger": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
}

async def chat(messages):
    full_response = ""
    # Pass the full history list to maintain conversation context
    async for part in await AsyncClient().chat(model='llama3.2', messages=messages, stream=True, format='json'):
        response_part = part['message']['content']
        print(response_part, end='', flush=True)
        full_response += response_part
    return full_response


def interrupt(user_input: str):
    global history
    # If this is the very first user message, just append it and get a response.
    if len(history) == 1:
        history.append({"role": "user", "content": user_input})
        response_text = asyncio.run(chat(history))
        history.append({"role": "assistant", "content": response_text})
    else:
        # history already contains system prompt + previous conversation pair (3 messages)
        # Append the new user prompt as the fourth element.
        history.append({"role": "user", "content": user_input})
        # Now history has 4 messages:
        # [system prompt, previous user, previous assistant, current user]
        response_text = asyncio.run(chat(history))
        history.append({"role": "assistant", "content": response_text})
        # After receiving the assistant response, update history to only contain:
        # system prompt + latest conversation pair (current user and its assistant response)
        system_prompt = history[0]
        last_user = history[-2]
        last_assistant = history[-1]
        history = [system_prompt, last_user, last_assistant]

if __name__ == "__main__":
    buffer = ""
    print("Start Typing:")
    while True:
        try:
            line = input()
            if not line:
                continue
            buffer += line + " "
            # Trigger if buffer has at least 10 words.
            if len(buffer.split()) >= CONFIG["word_trigger"]:
                interrupt(buffer.strip())
                buffer = ""
        except KeyboardInterrupt:
            break

