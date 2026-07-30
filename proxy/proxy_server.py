import time
import uuid
import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI()

# Configuration: In Docker, we use the service name 'rag-app' instead of 'localhost'
RAG_APP_URL = "http://rag-app:8000/query"

# --- OpenAI Compatibility Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage

# --- Proxy Logic ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Intercepts OpenAI-style chat completions and routes the 
    last user message to the RAG container.
    """
    # 1. Extract the last user message
    user_query = ""
    if request.messages:
        # We look for the last message in the conversation
        user_query = request.messages[-1].content

    if not user_query:
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="I couldn't find a query to process.")
                )
            ],
            usage=Usage()
        )

    # 2. Call the RAG container
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                RAG_APP_URL,
                json={"query": user_query},
                timeout=60.0
            )
            response.raise_for_status()
            rag_data = response.json()
            answer = rag_data.get("answer", "No answer was returned by the RAG system.")
        except Exception as e:
            answer = f"Error connecting to RAG container: {str(e)}"

    # 3. Wrap the answer in OpenAI format
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=answer)
            )
        ],
        usage=Usage()
    )

@app.get("/v1/models")
async def list_models():
    """Zed often checks for available models."""
    from fastapi.responses import JSONResponse
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": "rag-model", "object": "model", "owned": True}
        ]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
