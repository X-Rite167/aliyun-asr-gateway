import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Qwen ASR OpenAI Front Gateway", version="0.1.0")
BACKEND_URL = os.getenv(
    "ASR_BACKEND_URL",
    "http://aliyun-asr-gateway.litellm.svc.cluster.local:8091",
).rstrip("/")
BACKEND_MODEL = os.getenv("ASR_BACKEND_MODEL", "qwen-audio-3.0-asr-flash-filetrans")
PUBLIC_MODEL = os.getenv("ASR_PUBLIC_MODEL", "qwen-asr")
DEFAULT_LANGUAGE = os.getenv("ASR_DEFAULT_LANGUAGE", "zh")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str = PUBLIC_MODEL
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = 0
    extra_body: dict[str, Any] | None = None


def _public_ip(address: str) -> bool:
    return ipaddress.ip_address(address).is_global


def _validate_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="input_audio.data must be a public HTTP(S) URL")
    try:
        if _public_ip(parsed.hostname):
            return
    except ValueError:
        pass
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="audio URL hostname could not be resolved") from exc
    if not addresses or any(not _public_ip(address) for address in addresses):
        raise HTTPException(status_code=400, detail="input_audio.data must resolve to a public IP address")


def _extract(messages: list[ChatMessage]) -> tuple[str, str | None]:
    for message in reversed(messages):
        if not isinstance(message.content, list):
            continue
        prompt: list[str] = []
        for item in message.content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                prompt.append(item["text"])
            if item.get("type") == "input_audio":
                audio = item.get("input_audio") or {}
                data = audio.get("data")
                if isinstance(data, str) and data:
                    _validate_url(data)
                    return data, " ".join(prompt).strip() or None
    raise HTTPException(status_code=400, detail="messages must contain a public input_audio URL")


def _response(model: str, text: str) -> dict[str, Any]:
    import time
    import uuid
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _call_backend(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=900) as client:
        response = await client.post(f"{BACKEND_URL}/v1/audio/transcriptions", json=payload)
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"ASR backend error {response.status_code}: {response.text[:2000]}")
    return response.json()


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND_URL, "model": PUBLIC_MODEL}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.model != PUBLIC_MODEL:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
    if request.temperature not in (None, 0):
        raise HTTPException(status_code=400, detail="Only temperature=0 is supported")
    audio_url, prompt = _extract(request.messages)
    payload: dict[str, Any] = {
        "file": audio_url,
        "model": BACKEND_MODEL,
        "language": DEFAULT_LANGUAGE,
        "response_format": "json",
        "temperature": 0,
    }
    if prompt:
        payload["prompt"] = prompt
    result = await _call_backend(payload)
    return _response(request.model, str(result.get("text", "")))
