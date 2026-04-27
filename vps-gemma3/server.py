#!/usr/bin/env python3
"""Gemma3 API server — wraps ollama."""

import json
import os
from typing import AsyncIterator, List

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="Gemma3 API", version="1.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7


class GenerateRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = 0.7


async def _stream_ollama(payload: dict) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                if data.get("done"):
                    yield "data: [DONE]\n\n"
                    break


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(OLLAMA_URL)
            r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {e}")
    return {"status": "ok", "model": MODEL}


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    payload = {
        "model": MODEL,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
        "options": {"temperature": req.temperature},
    }

    if req.stream:
        return StreamingResponse(_stream_ollama(payload), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return {"content": data["message"]["content"], "model": MODEL}


@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    messages = [{"role": "user", "content": req.prompt}]
    return await chat(ChatRequest(messages=[Message(**m) for m in messages],
                                  stream=req.stream,
                                  temperature=req.temperature))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
