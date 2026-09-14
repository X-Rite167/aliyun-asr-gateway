import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import front_gateway

client = TestClient(front_gateway.app)


def test_front_gateway_calls_litellm_with_openai_chat_payload(monkeypatch):
    call = AsyncMock(return_value={
        "object": "chat.completion",
        "model": "qwen-asr",
        "choices": [{"message": {"role": "assistant", "content": "欢迎使用阿里云。"}}],
    })
    monkeypatch.setattr(front_gateway, "_call_litellm", call)
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
    assert payload["model"] == "qwen-asr"
    assert payload["messages"][0]["content"][1]["type"] == "input_audio"
    assert payload["messages"][0]["content"][1]["input_audio"]["data"] == "https://media.example.com/a.wav"


def test_front_gateway_does_not_call_post_gateway_directly(monkeypatch):
    call = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(front_gateway, "_call_litellm", call)
    monkeypatch.setattr(front_gateway, "_validate_url", lambda value: None)
    response = client.post("/chat/completions", json={
        "model": "qwen-asr",
        "messages": [{"role": "user", "content": [{
            "type": "input_audio", "input_audio": {"data": "https://media.example.com/a.wav"}
        }]}],
    })
    assert response.status_code == 200
    call.assert_awaited_once()


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
