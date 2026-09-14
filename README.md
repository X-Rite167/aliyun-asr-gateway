# 阿里云 Qwen ASR OpenAI 风格网关

该服务接收业务侧 OpenAI Chat Completions 请求，经网关封装后调用阿里云 OpenAI-compatible ASR 接口。

`POST /v1/chat/completions`

调用方直接传入公网可访问的音频 URL。网关不接收、上传或存储音频，只负责校验 URL、将业务模型别名映射到 `qwen3-asr-flash`、转发到 DashScope Compatible API 并返回 Chat Completions 响应。

## 调用

请求使用 OpenAI Chat Completions 格式：

```json
{"model":"qwen-audio-3.0-asr-flash","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"https://example.com/audio.wav"}}]}],"stream":false}
```

## 安装与启动

```bash
cd C:/Users/Lenovo/aliyun-asr-adapter
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8091
```

必须安全设置 `DASHSCOPE_API_KEY`，不要把真实密钥提交到 Git。

网关接口：

```bash
curl -X POST http://127.0.0.1:8091/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-audio-3.0-asr-flash","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"}}]}],"stream":false}'
```

外部模型名 `qwen-audio-3.0-asr-flash` 和 `qwen3-asr-flash` 都会映射到上游真实模型 `ALIYUN_ASR_COMPATIBLE_MODEL`，默认值为已验证的 `qwen3-asr-flash`。

## URL 安全规则

- 仅接受 `http://` 和 `https://`；生产建议只允许 HTTPS。
- 拒绝带用户名或密码的 URL。
- 拒绝解析到 loopback、私网、链路本地、保留地址等非公网 IP 的 URL。
- 签名 URL 的查询参数不应写入日志。

## Kubernetes 部署

网关应作为与 LiteLLM 分离的 Deployment + ClusterIP Service 部署。K8s Secret 只需要 `DASHSCOPE_API_KEY`；同时配置 `DASHSCOPE_COMPATIBLE_BASE`，例如 `https://ws-xxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。

网关提供 OpenAI Chat Completions 兼容请求；业务方通过 `messages[].content[].input_audio.data` 传入公网音频 URL。网关本身不存储或上传音频。
