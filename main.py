import torch
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ollama import AsyncClient

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global conversation history
history = [{
    "role": "system",
    "content": (
        "For every 10 words typed or after a 4-second debounce, the user's text will be sent to you "
        "along with a summarized message history. Respond in the following JSON format:\n"
        '{\n  "summarized_message_history": "<summary>",\n  "user_state": "<state>",\n  "response": "<output>"\n}\n'
        "Summarize the conversation like:\n\n"
        "'The user has discussed [topic] and their statements are factually correct.'\n"
        'The last response from the user was: "[latest input]."\n'
        "If the user needs to provide more input, include the full text in summarized_message_history instead of a summary.\n"
        "User State Determination\n"
        "Analyze the last 10 words to classify the user's state:\n\n"
        "Factually Incorrect – If the user makes a mistake, identify the incorrect information.\n"
        "Needs More Input – If the user has not completed their thought, set summarized_message_history to the full text and wait for more input.\n"
        "Speech-to-Text (STT) Junk Output – If the input appears garbled or nonsensical.\n"
        "STT Misinterpretation – If the input suggests a misheard word.\n\n"
        "Response Strategy\n"
        "If the user is incorrect, provide a correction.\n"
        'If the user needs to type more, respond with "Okay, go on..." while retaining the full text in summarized_message_history.\n'
        "If the input is junk or misheard, suggest a rephrase or clarify potential errors.\n"
        "You are assisting the user in reinforcing their knowledge, revising concepts, and preparing for exams by evaluating their understanding as they explain topics to you. "
        "Don't ask the user what they would like to explore further, instead let them talk about the topic. and if you feel like they left out any, let them know."
    )
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

# Pydantic models for request payloads
class UserInput(BaseModel):
    user_input: str

class SubtopicRequest(BaseModel):
    topic: str

class ExplainRequest(BaseModel):
    content_json: str
    specific_content: str

# Helper function for streaming chat responses
async def chat(messages, model="llama3.2", stream=True, format="json"):
    full_response = ""
    async for part in await AsyncClient().chat(
        model=model,
        messages=messages,
        stream=stream,
        format=format,
    ):
        full_response += part["message"]["content"]
    return full_response

# Helper to update conversation history (retain system prompt and last pair)
def update_history():
    global history
    if len(history) > 3:
        history = [history[0]] + history[-2:]

@app.post("/interrupt")
async def interrupt_endpoint(input: UserInput):
    global history
    user_text = input.user_input
    history.append({"role": "user", "content": user_text})
    response_text = await chat(history)
    history.append({"role": "assistant", "content": response_text})
    update_history()
    return {"response": response_text}

@app.post("/split")
async def get_subtopics(request: SubtopicRequest):
    messages = [
        {"role": "system", "content": (
            "You will be given a subtopic's response (generated content) by an LLM, give the response such that"
            " you convert the content into json. Make sure the content you're going to respond"
            " is in json format and the json has a key 'splitted_paragraphs'"
            " and the value is a list of paragraphs where each paragraph is a list of sentences."
            " Basically split it wherever you think that it is a different topic that"
            " can be explored/expanded even more. 'Don't expand it, just split it'."
        )},
        {"role": "user", "content": f"Generated Subtopic Content: {request.topic}"}
    ]
    response_text = await chat(messages)
    json_data = json.loads(response_text)
    return JSONResponse(content=json_data)

@app.post("/explain")
async def explain(request: ExplainRequest):
    # Summarize the specific content
    summarize_messages = [
        {"role": "system", "content": "You will be given content to summarize, summarize it. "},
        {"role": "user", "content": f"Content to summarize: {request.content_json}"}
    ]
    summary_response = await chat(summarize_messages, format="")
    
    # Explain using the summarized content as context
    explain_messages = [
        {"role": "system", "content": "You'll be given a specific content to explain, explain it using the summarized content as context. "},
        {"role": "user", "content": f"Context: {summary_response}\nSpecific Content: {request.specific_content}"}
    ]
    explanation_response = await chat(explain_messages, format="")
    
    # Split the explanation into subtopics
    split_messages = [
        {"role": "system", "content": (
            "You will be given a subtopic's response (generated content) by an LLM, give the response such that"
            " you convert the content into json. Make sure the content you're going to respond"
            " is in json format and the json has a key 'splitted_paragraphs'"
            " and the value is a list of paragraphs where each paragraph is a list of sentences."
            " Basically split it wherever you think that it is a different topic that"
            " can be explored/expanded even more. 'Don't expand it, just split it'."
        )},
        {"role": "user", "content": f"Generated Subtopic Content: {explanation_response}"}
    ]
    split_response = await chat(split_messages)
    json_data = json.loads(split_response)
    return JSONResponse(content=json_data)

class QueryRequest(BaseModel):
    topic: str

@app.post("/generate_queries")
async def generate_google_queries(request: QueryRequest):
    messages = [
        {
            "role": "system",
            "content": (
                "You will be given a topic name. Generate a list of query sentences that can be used to search on Google about the topic. "
                "Your response must be in JSON format with a key 'google_queries' and its value should be a list of query sentences."
            )
        },
        {"role": "user", "content": f"Topic: {request.topic}"}
    ]
    response_text = await chat(messages)
    json_data = json.loads(response_text)
    return JSONResponse(content=json_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
