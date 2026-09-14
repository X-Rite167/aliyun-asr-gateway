import asyncio
import ipaddress
import json
import os
import socket
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Alibaba Qwen ASR OpenAI Gateway", version="0.2.2")

MODEL = os.getenv("ALIYUN_ASR_MODEL", "qwen-audio-3.0-asr-flash-filetrans")
COMPATIBLE_MODEL = os.getenv("ALIYUN_ASR_COMPATIBLE_MODEL", "qwen3-asr-flash")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE = os.getenv("DASHSCOPE_BASE", "https://dashscope.aliyuncs.com")
DASHSCOPE_COMPATIBLE_BASE = os.getenv("DASHSCOPE_COMPATIBLE_BASE")
POLL_INTERVAL = float(os.getenv("ASR_POLL_INTERVAL_SECONDS", "2"))
POLL_TIMEOUT = float(os.getenv("ASR_POLL_TIMEOUT_SECONDS", "900"))


class TranscriptionRequest(BaseModel):
    file: str = Field(..., description="Publicly reachable HTTP(S) audio URL")
    model: str = MODEL
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float = 0


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = MODEL
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float = 0
    extra_body: dict[str, Any] | None = None


COMPATIBLE_MODEL_ALIASES = {"qwen3-asr-flash", "qwen-audio-3.0-asr-flash"}


def _extract_audio_input(messages: list[ChatMessage]) -> tuple[str, str | None]:
    for message in reversed(messages):
        content = message.content
        if isinstance(content, list):
            prompt_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "input_audio":
                    audio = item.get("input_audio") or {}
                    data = audio.get("data")
                    if isinstance(data, str) and data:
                        return data, " ".join(prompt_parts).strip() or None
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    prompt_parts.append(item["text"])
        elif isinstance(content, str):
            # Text-only messages are retained as a prompt, but an audio URL is
            # still required by this file-transcription adapter.
            continue
    raise HTTPException(status_code=400, detail="messages must contain a public input_audio URL")


def _chat_response(model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chat_stream(model: str, text: str):
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    first = {"id": response_id, "object": "chat.completion.chunk", "created": created,
             "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
    last = {"id": response_id, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _required(name: str, value: str | None) -> str:
    if not value:
        raise HTTPException(status_code=503, detail=f"Missing server configuration: {name}")
    return value


def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_global


async def _validate_public_url(file_url: str) -> None:
    try:
        parsed = urlsplit(file_url)
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="file must be a public HTTP(S) URL") from exc

    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="file must be a public HTTP(S) URL")

    hostname = parsed.hostname
    try:
        if not _is_public_ip(hostname):
            raise HTTPException(status_code=400, detail="file must resolve to a public IP address")
        return
    except ValueError:
        pass

    try:
        infos = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: socket.getaddrinfo(
                hostname,
                port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            ),
        )
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="file hostname could not be resolved") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses or any(not _is_public_ip(address) for address in addresses):
        raise HTTPException(status_code=400, detail="file must resolve to a public IP address")


async def _dashscope(method: str, url: str, **kwargs) -> dict[str, Any]:
    api_key = _required("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY)
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.request(method, url, headers=headers, **kwargs)
    if response.status_code >= 400:
        detail = response.text[:2000]
        raise HTTPException(status_code=502, detail=f"DashScope error {response.status_code}: {detail}")
    return response.json()


async def _dashscope_compatible(payload: dict[str, Any]) -> dict[str, Any]:
    base = _required("DASHSCOPE_COMPATIBLE_BASE", DASHSCOPE_COMPATIBLE_BASE)
    return await _dashscope(
        "POST",
        f"{base.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
    )


async def _transcribe(file_url: str, language: str | None, prompt: str | None) -> str:
    parameters: dict[str, Any] = {"channel_id": [0]}
    if language:
        parameters["language_hints"] = [language]
    if prompt:
        parameters["vocabulary"] = {prompt: 5}

    submit = await _dashscope(
        "POST",
        f"{DASHSCOPE_BASE.rstrip('/')}/api/v1/services/audio/asr/transcription",
        headers={"Content-Type": "application/json", "X-DashScope-Async": "enable"},
        json={"model": MODEL, "input": {"file_urls": [file_url]}, "parameters": parameters},
    )
    task_id = submit.get("output", {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=502, detail="DashScope response did not contain output.task_id")

    deadline = asyncio.get_running_loop().time() + POLL_TIMEOUT
    while True:
        result = await _dashscope("GET", f"{DASHSCOPE_BASE.rstrip('/')}/api/v1/tasks/{task_id}")
        output = result.get("output", {})
        status = output.get("task_status")
        if status == "SUCCEEDED":
            results = output.get("results") or []
            if not results or results[0].get("subtask_status") != "SUCCEEDED":
                raise HTTPException(status_code=502, detail="DashScope returned no successful subtask")
            result_url = results[0].get("transcription_url")
            if not result_url:
                raise HTTPException(status_code=502, detail="DashScope result has no transcription_url")
            async with httpx.AsyncClient(timeout=60) as client:
                result_response = await client.get(result_url)
            if result_response.status_code >= 400:
                raise HTTPException(status_code=502, detail="Failed to download transcription result")
            payload = result_response.json()
            transcripts = payload.get("transcripts") or []
            return "\n".join(str(item.get("text", "")) for item in transcripts).strip()
        if status == "FAILED":
            raise HTTPException(status_code=502, detail=json.dumps(output, ensure_ascii=False))
        if asyncio.get_running_loop().time() >= deadline:
            raise HTTPException(status_code=504, detail=f"ASR task timed out; task_id={task_id}")
        await asyncio.sleep(POLL_INTERVAL)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: TranscriptionRequest):
    if request.model != MODEL:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
    if request.response_format not in {"json", "text"}:
        raise HTTPException(status_code=400, detail="Only response_format=json or text is supported")
    if request.temperature != 0:
        raise HTTPException(status_code=400, detail="Only temperature=0 is supported")

    await _validate_public_url(request.file)
    text = await _transcribe(request.file, request.language, request.prompt)

    if request.response_format == "text":
        return PlainTextResponse(text)
    return {"text": text}


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.model not in {MODEL, *COMPATIBLE_MODEL_ALIASES}:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
    if request.temperature != 0:
        raise HTTPException(status_code=400, detail="Only temperature=0 is supported")

    file_url, prompt = _extract_audio_input(request.messages)
    await _validate_public_url(file_url)
    if request.model in COMPATIBLE_MODEL_ALIASES:
        provider_payload = {
            "model": COMPATIBLE_MODEL,
            "messages": [message.model_dump() for message in request.messages],
            "stream": request.stream,
        }
        if request.stream_options is not None:
            provider_payload["stream_options"] = request.stream_options
        if request.extra_body:
            provider_payload.update(request.extra_body)
        return await _dashscope_compatible(provider_payload)

    text = await _transcribe(file_url, None, prompt)
    if request.stream:
        return StreamingResponse(_chat_stream(request.model, text), media_type="text/event-stream")
    return _chat_response(request.model, text)
