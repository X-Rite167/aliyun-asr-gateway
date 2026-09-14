import asyncio
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == app.MODEL


def test_accepts_public_audio_url_in_openai_shaped_json(monkeypatch):
    transcribe = AsyncMock(return_value="你好，世界")
    monkeypatch.setattr(app, "_validate_public_url", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_transcribe", transcribe)
    response = client.post("/v1/audio/transcriptions", json={"file": "https://media.example.com/audio/a.wav?signature=test", "model": app.MODEL, "language": "zh", "prompt": "医学术语", "response_format": "json", "temperature": 0})
    assert response.status_code == 200
    assert response.json() == {"text": "你好，世界"}
    transcribe.assert_awaited_once_with("https://media.example.com/audio/a.wav?signature=test", "zh", "医学术语")


def test_returns_plain_text_for_text_response_format(monkeypatch):
    monkeypatch.setattr(app, "_validate_public_url", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_transcribe", AsyncMock(return_value="测试文本"))
    response = client.post("/v1/audio/transcriptions", json={"file": "https://media.example.com/a.mp3", "model": app.MODEL, "response_format": "text"})
    assert response.status_code == 200
    assert response.text == "测试文本"
    assert response.headers["content-type"].startswith("text/plain")


def test_rejects_unsupported_response_format():
    response = client.post("/v1/audio/transcriptions", json={"file": "https://media.example.com/a.wav", "model": app.MODEL, "response_format": "srt"})
    assert response.status_code == 400


def test_rejects_unsupported_model():
    response = client.post("/v1/audio/transcriptions", json={"file": "https://media.example.com/a.wav", "model": "whisper-1"})
    assert response.status_code == 400


def test_rejects_non_http_audio_url():
    response = client.post("/v1/audio/transcriptions", json={"file": "file:///etc/passwd", "model": app.MODEL})
    assert response.status_code == 400


def test_rejects_private_audio_url():
    response = client.post("/v1/audio/transcriptions", json={"file": "http://192.168.200.128/audio.wav", "model": app.MODEL})
    assert response.status_code == 400


def test_submit_and_poll_contract(monkeypatch):
    responses = [{"output": {"task_id": "task-1", "task_status": "PENDING"}}, {"output": {"task_id": "task-1", "task_status": "SUCCEEDED", "results": [{"subtask_status": "SUCCEEDED", "transcription_url": "https://result.example/r.json"}]}}]
    calls = []
    async def fake_dashscope(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return responses.pop(0)
    class Result:
        status_code = 200
        def json(self): return {"transcripts": [{"text": "测试结果"}]}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url): return Result()
    monkeypatch.setattr(app, "DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(app, "_dashscope", fake_dashscope)
    monkeypatch.setattr(app.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(app, "POLL_INTERVAL", 0)
    assert asyncio.run(app._transcribe("https://audio.example/a.wav", "zh", None)) == "测试结果"
    assert not responses
    assert calls[0][2]["json"]["input"]["file_urls"] == ["https://audio.example/a.wav"]


def test_chat_completions_accepts_openai_input_audio_and_returns_chat_shape(monkeypatch):
    monkeypatch.setattr(app, "_validate_public_url", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_transcribe", AsyncMock(return_value="欢迎使用阿里云。"))
    response = client.post("/v1/chat/completions", json={"model": app.MODEL, "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "https://media.example.com/a.wav", "format": "wav"}}]}], "stream": False})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == app.MODEL
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "欢迎使用阿里云。"}


def test_chat_completions_rejects_audio_without_public_url():
    response = client.post("/v1/chat/completions", json={"model": app.MODEL, "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]})
    assert response.status_code == 400


def test_chat_completions_uses_dashscope_openai_compatible_asr(monkeypatch):
    monkeypatch.setattr(app, "DASHSCOPE_COMPATIBLE_BASE", "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(app, "_validate_public_url", AsyncMock(return_value=None))
    provider = AsyncMock(return_value={"id": "chatcmpl-provider", "object": "chat.completion", "model": "qwen3-asr-flash", "choices": [{"message": {"role": "assistant", "content": "你好"}}]})
    monkeypatch.setattr(app, "_dashscope_compatible", provider)
    response = client.post("/v1/chat/completions", json={"model": "qwen3-asr-flash", "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "https://media.example.com/a.wav"}}]}], "stream": False, "extra_body": {"asr_options": {"enable_itn": False}}})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "你好"
    provider.assert_awaited_once()
    payload = provider.await_args.args[0]
    assert payload["model"] == "qwen3-asr-flash"
    assert payload["messages"][0]["content"][0]["input_audio"]["data"] == "https://media.example.com/a.wav"


def test_compatible_gateway_wraps_request_for_dashscope(monkeypatch):
    monkeypatch.setattr(
        app,
        "DASHSCOPE_COMPATIBLE_BASE",
        "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    dashscope = AsyncMock(return_value={"choices": []})
    monkeypatch.setattr(app, "_dashscope", dashscope)

    result = asyncio.run(app._dashscope_compatible({
        "model": "qwen3-asr-flash",
        "messages": [{"role": "user", "content": "payload"}],
        "stream": False,
        "extra_body": {"asr_options": {"enable_itn": False}},
    }))

    assert result == {"choices": []}
    dashscope.assert_awaited_once()
    method, url = dashscope.await_args.args[:2]
    kwargs = dashscope.await_args.kwargs
    assert method == "POST"
    assert url == "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert kwargs["json"]["model"] == "qwen3-asr-flash"
    assert kwargs["json"]["extra_body"]["asr_options"]["enable_itn"] is False


def test_gateway_maps_litellm_model_to_dashscope_asr_model(monkeypatch):
    monkeypatch.setattr(
        app,
        "DASHSCOPE_COMPATIBLE_BASE",
        "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setattr(app, "_validate_public_url", AsyncMock(return_value=None))
    provider = AsyncMock(return_value={
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "测试转写"}}],
    })
    monkeypatch.setattr(app, "_dashscope_compatible", provider)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-audio-3.0-asr-flash",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": "https://media.example.com/a.wav"},
                }],
            }],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = provider.await_args.args[0]
    assert payload["model"] == "qwen3-asr-flash"
    assert response.json()["choices"][0]["message"]["content"] == "测试转写"


def test_gateway_accepts_litellm_custom_provider_path_without_v1(monkeypatch):
    monkeypatch.setattr(app, "_validate_public_url", AsyncMock(return_value=None))
    provider = AsyncMock(return_value={
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "别名路径"}}],
    })
    monkeypatch.setattr(app, "_dashscope_compatible", provider)
    response = client.post(
        "/chat/completions",
        json={
            "model": "qwen-audio-3.0-asr-flash",
            "messages": [{"role": "user", "content": [{
                "type": "input_audio",
                "input_audio": {"data": "https://media.example.com/a.wav"},
            }]}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "别名路径"


def test_gateway_accepts_litellm_base_url_probe(monkeypatch):
    response = client.post("/v1", json={
        "model": "qwen-asr",
        "params": {"prompt": ["What's 1 + 1?"], "max_tokens": 16},
    })
    assert response.status_code == 200
    assert response.json() == {"data": [{"output": ["ok"]}]}


def test_chat_reconstructs_provider_message_from_openai_audio_content(monkeypatch):
    monkeypatch.setattr(app, "DASHSCOPE_COMPATIBLE_BASE", "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(app, "_validate_public_url", AsyncMock(return_value=None))
    provider = AsyncMock(return_value={"choices": [{"message": {"role": "assistant", "content": "重构成功"}}]})
    monkeypatch.setattr(app, "_dashscope_compatible", provider)
    request_body = {
        "model": "qwen-audio-3.0-asr-flash",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "请转写"},
            {"type": "input_audio", "input_audio": {"data": "https://media.example.com/a.wav", "format": "wav"}},
        ]}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    payload = provider.await_args.args[0]
    assert payload["model"] == "qwen3-asr-flash"
    assert payload["messages"] == request_body["messages"]
