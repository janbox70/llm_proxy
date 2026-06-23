import asyncio
import inspect
import yaml
import os
import traceback
from contextlib import asynccontextmanager
from fastapi import Depends, Header, Request, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response
from openai import APIError, AsyncOpenAI as OpenAI, AsyncStream
import httpx
from logger import ChatLogger, AnthropicChatLogger, _remove_image_data, _remove_anthropic_image_data
from converters import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    openai_stream_to_anthropic_stream,
)


async def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_site_config(config, provider: str):
    providers = config.get("providers", {})
    site_config = providers.get(provider, {})
    if "alias" in site_config:
        site_config = providers.get(site_config["alias"])
    return site_config

# 客户端缓存，避免重复创建
_client_cache = {}

# httpx 客户端缓存（用于 Anthropic 等非 OpenAI SDK 的转发）
_httpx_clients: dict[str, httpx.AsyncClient] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 第一阶段
    yield
    
    # 关闭时清理 OpenAI 客户端
    for client in _client_cache.values():
        await client.close()
    _client_cache.clear()
    # 关闭时清理 httpx 客户端
    for client in _httpx_clients.values():
        await client.aclose()
    _httpx_clients.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return "index"


@app.head("/{provider}")
@app.get("/{provider}")
async def provider_health_check(provider: str):
    """Provider 健康检查端点，支持 HEAD 和 GET 请求"""
    config = await load_config()
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    providers = config.get("providers", {})
    if provider not in providers:
        raise HTTPException(
            status_code=404, detail=f"Provider '{provider}' not found"
        )
    
    # 对于 HEAD 请求，FastAPI 会自动处理为空响应
    # 对于 GET 请求，返回 provider 基本信息
    provider_info = {
        "provider": provider,
        "status": "available",
        "base_url": providers[provider].get("base_url"),
        "api_type": providers[provider].get("api_type", "openai-compatible")
    }
    return provider_info


@app.exception_handler(HTTPException)
async def http_exception_handler(request, ex):
    print(f"http_exception_handler: {ex}.")
    return JSONResponse(
        status_code=ex.status_code, content={"message": ex.detail}
    )


@app.exception_handler(APIError)
async def openai_exception_handler(request, ex: APIError):
    async def _iter_data(response):
        async for chunk in response.aiter_bytes():
            yield chunk

    print(f"openai_exception_handler: {ex}. \n{traceback.format_exc()}")

    if getattr(ex, 'response', None) is not None:
        return StreamingResponse(_iter_data(ex.response), status_code=ex.status_code, headers=ex.response.headers)
    return JSONResponse(status_code=400, content={'error': f'{ex}'}, media_type='application/json')


async def _iter_events(response: AsyncStream, logger: ChatLogger):
    try:
        async for sse in response.response.aiter_bytes():
            yield sse
            logger.add_chunk(sse)
    finally:
        # 记录日志
        await logger.write_log()


async def get_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=401, detail="Authorization header is missing")
    parts = authorization.split()
    if len(parts) == 1:
        return parts[0]
    if len(parts) > 1 and parts[1]:
        return parts[1]
    raise HTTPException(
        status_code=401, detail="Invalid authorization header")


def get_client(base_url: str, api_key: str) -> OpenAI:
    """获取或创建 OpenAI 客户端（带缓存）"""
    cache_key = f"{base_url}:{api_key}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    return _client_cache[cache_key]


@app.post("/{provider}/chat/completions", summary="标准OPEN AI聊天接口")
async def standard_chat(provider: str, request: Request, api_key=Depends(get_api_key)):

    data = await request.json()

    config = await load_config()
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")

    logger = ChatLogger(config.get("log_path", "logs"), provider, api_key, data,
                        path="v1/chat/completions", upstream_path="v1/chat/completions")

    try:
        site_config = get_site_config(config, provider)
        if not site_config or 'base_url' not in site_config:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}")

        # 模型映射：如果配置了 models_mapping，则替换模型名称
        models_mapping = site_config.get("models_mapping", {})
        original_model = data.get("model")
        if original_model and models_mapping and original_model in models_mapping:
            data["model"] = models_mapping[original_model]
            print(f"Model mapped: {original_model} -> {data['model']} for provider {provider}")

        # 如果在白名单上，则使用预配置的 api_key
        if api_key in site_config.get("authorized_keys", []):
            api_key = site_config.get("api_key", api_key)

        base_url = site_config.get("base_url", "")
        client = get_client(base_url, api_key)
        params = inspect.signature(client.chat.completions.create).parameters

        unsupport_params = ['frequency_penalty',
                            'presence_penalty']  # gemini unsupport

        body, extra_body = {}, {}
        for name, value in data.items():
            if name in unsupport_params:
                continue
            if name in params:
                body[name] = value
            else:
                extra_body[name] = value
        if 'stop' in body and not body['stop']:
            del body['stop']

        response = await client.chat.completions.create(**body, extra_body=extra_body)

        if isinstance(response, AsyncStream):
            return StreamingResponse(_iter_events(response, logger), media_type="text/event-stream")

        await logger.write_log(response.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR!!: provider {provider}")
        print(_remove_image_data(data))
        raise HTTPException(status_code=500, detail=str(e))

    return response


@app.post("/{provider}/embeddings", summary="标准OPEN AI embeddings接口")
async def standard_embeddings(provider: str, request: Request, api_key=Depends(get_api_key)):

    data = await request.json()

    logger = ChatLogger(provider, api_key, data)

    try:
        site_config = get_site_config(await load_config(), provider)
        if 'base_url' not in site_config:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}")

        # 如果在白名单上，则使用预配置的 api_key
        if api_key in site_config.get("authorized_keys", []):
            api_key = site_config.get("api_key", api_key)

        client = OpenAI(api_key=api_key,
                        base_url=site_config.get("base_url", ""))

        # 获取embeddings.create方法的参数
        params = inspect.signature(client.embeddings.create).parameters

        body, extra_body = {}, {}
        for name, value in data.items():
            if name in params:
                body[name] = value
            else:
                extra_body[name] = value

        response = await client.embeddings.create(**body, extra_body=extra_body)

        await logger.write_log(response.model_dump())
    except Exception as e:
        print(f"ERROR!!: provider {provider}, api-key {api_key}")
        print(data)
        raise e

    return response


# ──────────────────────────────────────────────────────────────
# Anthropic Messages API 代理
# ──────────────────────────────────────────────────────────────


async def get_anthropic_api_key(request: Request) -> str:
    """提取 Anthropic API Key：优先 x-api-key，回退 Authorization: Bearer"""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key
    authorization = request.headers.get("authorization")
    if authorization:
        parts = authorization.split()
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        if len(parts) == 1 and parts[0]:
            return parts[0]
    raise HTTPException(
        status_code=401, detail="Missing x-api-key or Authorization header"
    )


def get_httpx_client(provider: str, base_url: str, api_key: str) -> httpx.AsyncClient:
    """获取或创建 httpx 客户端（per provider 缓存）"""
    cache_key = f"{provider}:{api_key}"
    if cache_key not in _httpx_clients:
        _httpx_clients[cache_key] = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )
    return _httpx_clients[cache_key]


async def _iter_anthropic_events(
    response: httpx.Response, logger: AnthropicChatLogger
):
    """透传 Anthropic SSE 字节流，同时记录日志"""
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
            logger.add_chunk(chunk)
    finally:
        await logger.write_log()


@app.post("/{provider}/v1/messages", summary="Anthropic Messages API 代理")
async def anthropic_messages(
    provider: str, request: Request, api_key: str = Depends(get_anthropic_api_key)
):
    data = await request.json()

    config = await load_config()
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")

    logger = AnthropicChatLogger(
        config.get("log_path", "logs"), provider, api_key, data,
        path="v1/messages", upstream_path=""
    )

    try:
        site_config = get_site_config(config, provider)
        if not site_config or "base_url" not in site_config:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}"
            )

        # 模型映射：如果配置了 models_mapping，则替换模型名称
        models_mapping = site_config.get("models_mapping", {})
        original_model = data.get("model")
        if original_model and models_mapping and original_model in models_mapping:
            data["model"] = models_mapping[original_model]
            print(f"Model mapped (Anthropic): {original_model} -> {data['model']} for provider {provider}")

        # 如果在白名单上，替换为预配置的 api_key
        if api_key in site_config.get("authorized_keys", []):
            real_key = site_config.get("api_key", api_key)
        else:
            real_key = api_key

        # 获取 provider 的 api_type，默认 openai-compatible
        api_type = site_config.get("api_type", "openai-compatible")
        base_url = site_config["base_url"]
        is_stream = data.get("stream", False)

        if api_type == "anthropic":
            # 直接转发到 Anthropic API
            logger.upstream_path = "v1/messages"
            return await _forward_to_anthropic(
                provider, base_url, real_key, data, is_stream, request, logger
            )
        else:
            # 转换为 OpenAI 格式，调用 OpenAI 兼容接口
            logger.upstream_path = "v1/chat/completions"
            return await _forward_to_openai_as_anthropic(
                provider, base_url, real_key, data, is_stream, logger
            )
    except Exception as e:
        print(f"ERROR!!: anthropic provider {provider}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        print(_remove_anthropic_image_data(data))
        raise HTTPException(status_code=500, detail=str(e))


async def _forward_to_anthropic(
    provider: str,
    base_url: str,
    real_key: str,
    data: dict,
    is_stream: bool,
    request: Request,
    logger: AnthropicChatLogger,
):
    """直接转发请求到 Anthropic API"""
    client = get_httpx_client(provider, base_url, real_key)

    # 转发 anthropic-version header
    anthropic_version = request.headers.get("anthropic-version")
    headers = {}
    if anthropic_version:
        headers["anthropic-version"] = anthropic_version
    client.headers["x-api-key"] = real_key
    client.headers["Authorization"] = f"Bearer {real_key}"

    if is_stream:
        upstream = await client.send(
            client.build_request("POST", "/v1/messages", json=data, headers=headers),
            stream=True,
        )
        if upstream.status_code != 200:
            body = await upstream.aread()
            await upstream.aclose()
            return Response(
                content=body,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )
        return StreamingResponse(
            _iter_anthropic_events(upstream, logger),
            media_type="text/event-stream",
        )
    else:
        upstream = await client.post("/v1/messages", json=data, headers=headers)
        if upstream.status_code != 200:
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )
        resp_data = upstream.json()
        await logger.write_log(resp_data)
        return JSONResponse(content=resp_data, status_code=200)


async def _forward_to_openai_as_anthropic(
    provider: str,
    base_url: str,
    real_key: str,
    data: dict,
    is_stream: bool,
    logger: AnthropicChatLogger,
):
    """
    将 Anthropic 格式请求转换为 OpenAI 格式，调用 OpenAI 兼容接口，
    再将响应转换回 Anthropic 格式返回给客户端
    """
    # 转换请求格式
    openai_req = anthropic_to_openai_request(data)
    # 确保 stream 参数正确设置
    openai_req["stream"] = is_stream

    client = get_client(base_url, real_key)

    # 获取 SDK 支持的参数
    params = inspect.signature(client.chat.completions.create).parameters

    # 过滤不支持的参数
    unsupport_params = ['frequency_penalty', 'presence_penalty']

    body, extra_body = {}, {}
    for name, value in openai_req.items():
        if name in unsupport_params:
            continue
        if name in params:
            body[name] = value
        else:
            extra_body[name] = value

    try:
        if is_stream:
            response = await client.chat.completions.create(**body, extra_body=extra_body, stream=True)

            if not isinstance(response, AsyncStream):
                # 某些客户端可能不支持流式，降级为非流式
                anthropic_resp = openai_to_anthropic_response(response.model_dump(), data.get("model"))
                await logger.write_log(anthropic_resp)
                return JSONResponse(content=anthropic_resp, status_code=200)

            # 流式：将 OpenAI SSE 转换为 Anthropic SSE
            async def _convert_and_log_stream():
                collected_output = {"content": "", "tool_calls": {}, "stop_reason": "", "usage": None}
                async for sse_bytes in openai_stream_to_anthropic_stream(response, data.get("model")):
                    # 同时收集日志数据
                    _collect_stream_log(sse_bytes, collected_output)
                    yield sse_bytes
                await logger.write_log(collected_output)

            return StreamingResponse(
                _convert_and_log_stream(),
                media_type="text/event-stream",
            )
        else:
            response = await client.chat.completions.create(**body, extra_body=extra_body)

            if isinstance(response, AsyncStream):
                # 请求了非流式但返回了流式，需要聚合
                collected_output = {"content": "", "tool_calls": {}, "stop_reason": "", "usage": None}
                async for chunk in response:
                    # 这里需要处理流式聚合并转换为 Anthropic 格式
                    # 简化处理：只取最后一个 chunk 的信息
                    pass
                anthropic_resp = openai_to_anthropic_response(
                    {"choices": [{"message": {"content": collected_output["content"]}}]},
                    data.get("model")
                )
                await logger.write_log(anthropic_resp)
                return JSONResponse(content=anthropic_resp, status_code=200)

            # 正常非流式响应
            openai_resp = response.model_dump()
            anthropic_resp = openai_to_anthropic_response(openai_resp, data.get("model"))
            await logger.write_log(anthropic_resp)
            return JSONResponse(content=anthropic_resp, status_code=200)

    except Exception as e:
        # 将 OpenAI 错误转换为 Anthropic 格式错误
        raise e
        return _format_anthropic_error(e)


def _format_anthropic_error(error: Exception) -> JSONResponse:
    """将 OpenAI API 错误转换为 Anthropic 格式错误响应"""
    # 尝试提取状态码
    status_code = getattr(error, 'status_code', 500)
    if status_code < 400:
        status_code = 500

    # 提取错误信息
    error_message = str(error)

    # 尝试解析更详细的错误信息
    error_type = "api_error"
    if "401" in error_message or "invalid_api_key" in error_message.lower():
        error_type = "authentication_error"
        status_code = 401
    elif "403" in error_message or "forbidden" in error_message.lower():
        error_type = "permission_error"
        status_code = 403
    elif "404" in error_message or "not_found" in error_message.lower():
        error_type = "not_found_error"
        status_code = 404
    elif "429" in error_message or "rate" in error_message.lower():
        error_type = "rate_limit_error"
        status_code = 429
    elif "400" in error_message or "invalid" in error_message.lower():
        error_type = "invalid_request_error"
        status_code = 400

    # 限制错误信息长度
    if len(error_message) > 500:
        error_message = error_message[:500] + "..."

    anthropic_error = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": error_message
        }
    }
    return JSONResponse(content=anthropic_error, status_code=status_code)


def _collect_stream_log(sse_bytes: bytes, output: dict):
    """从转换后的 SSE 流中收集日志数据"""
    import json
    text = sse_bytes.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        content = line[5:].strip()
        if not content:
            continue
        try:
            data = json.loads(content)
            event_type = data.get("type", "")

            if event_type == "message_start":
                msg = data.get("message", {})
                if "usage" in msg and msg["usage"]:
                    output["usage"] = dict(msg["usage"])

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    output["content"] += delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    idx = data.get("index", 0)
                    if idx not in output["tool_calls"]:
                        output["tool_calls"][idx] = {"function": {"name": "", "arguments": ""}}
                    output["tool_calls"][idx]["function"]["arguments"] += delta.get("partial_json", "")

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                if "stop_reason" in delta:
                    output["stop_reason"] = delta["stop_reason"]
                if "usage" in delta and delta["usage"]:
                    if output["usage"] is None:
                        output["usage"] = {}
                    output["usage"].update(delta["usage"])
        except json.JSONDecodeError:
            pass