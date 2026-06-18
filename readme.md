# llm_proxy

大模型API调用代理 （openai_compatible）

## 配置

需要在 config.yaml 中配置 provider
可以在 provider 中配置上游的api_key，并面向下游分发新的 key


## 调用

在端侧，通过 `http://{base_url}/{provider}` 来调用

### 支持的路由

- `GET /` - 健康检查
- `HEAD /{provider}` - Provider 健康检查（HEAD 请求）
- `GET /{provider}` - Provider 健康检查（返回 provider 信息）
- `POST /{provider}/chat/completions` - OpenAI 兼容聊天接口
- `POST /{provider}/embeddings` - Embeddings 接口
- `POST /{provider}/v1/messages` - Anthropic Messages API（支持自动格式转换）

### 示例

```bash
# HEAD 请求检查 provider 是否可用
curl -I http://localhost:8088/arkcodingplan

# GET 请求获取 provider 信息
curl http://localhost:8088/arkcodingplan

# POST 请求调用聊天接口
curl -X POST http://localhost:8088/qwencodingplan/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-plus", "messages": [{"role": "user", "content": "你好"}]}'
```
