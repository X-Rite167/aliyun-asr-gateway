import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import front_gateway

client = TestClient(front_gateway.app)


def test_front_gateway_forwards_messages_to_backend_chat(monkeypatch):
    call = AsyncMock(return_value={
        "object": "chat.completion",
        "model": "qwen-asr",
        "choices": [{"message": {"role": "assistant", "content": "欢迎使用阿里云。"}}],
    })
    monkeypatch.setattr(front_gateway, "_call_litellm", call)
    monkeypatch.setattr(front_gateway, "_validate_url", lambda value: None)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "请准确转写"},
        {"type": "input_audio", "input_audio": {
            "data": "https://media.example.com/a.wav", "format": "wav"
        }},
    ]}]
    response = client.post("/v1/chat/completions", json={
        "model": "qwen-asr",
        "messages": messages,
        "stream": False,
    })
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "欢迎使用阿里云。"
    payload = call.await_args.args[0]
    assert payload["model"] == "qwen-asr"
    assert payload["messages"] == messages


def test_front_gateway_requires_audio_url():
    response = client.post("/v1/chat/completions", json={
        "model": "qwen-asr",
        "messages": [{"role": "user", "content": "only text"}],
    })
    assert response.status_code == 400


def test_front_gateway_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_front_gateway_omits_auth_header_when_no_key(monkeypatch):
    import asyncio
    monkeypatch.setattr(front_gateway, "LITELLM_API_KEY", "")

    captured = {}

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def post(self, url, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            return FakeResponse()

    monkeypatch.setattr(front_gateway.httpx, "AsyncClient", FakeClient)
    asyncio.run(front_gateway._call_litellm({"model": "qwen-asr", "messages": []}))
    assert "Authorization" not in captured["headers"]
