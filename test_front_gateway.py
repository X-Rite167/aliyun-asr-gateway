import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import front_gateway

client = TestClient(front_gateway.app)


def test_front_gateway_transforms_openai_audio_to_backend_json(monkeypatch):
    call = AsyncMock(return_value={"text": "欢迎使用阿里云。"})
    monkeypatch.setattr(front_gateway, "_call_backend", call)
    monkeypatch.setattr(front_gateway, "_validate_url", lambda value: None)
    response = client.post("/v1/chat/completions", json={
        "model": "qwen-asr",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "请准确转写"},
            {"type": "input_audio", "input_audio": {
                "data": "https://media.example.com/a.wav", "format": "wav"
            }},
        ]}],
        "stream": False,
    })
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "欢迎使用阿里云。"
    payload = call.await_args.args[0]
    assert payload == {
        "file": "https://media.example.com/a.wav",
        "model": "qwen-audio-3.0-asr-flash-filetrans",
        "language": "zh",
        "response_format": "json",
        "temperature": 0,
        "prompt": "请准确转写",
    }


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
