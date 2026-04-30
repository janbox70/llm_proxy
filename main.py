from datetime import datetime
import inspect
import json
import yaml
import os
import traceback
from fastapi import Depends, Header, Request, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError, AsyncOpenAI as OpenAI, AsyncStream
from logger import ChatLogger, _remove_image_data


app = FastAPI()


@app.get("/")
async def index():
    return "index"


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
    return parts[1]

# 记录日志


async def load_config():
    path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path, f"config.yaml"), "r") as f:
        return yaml.safe_load(f)


@app.post("/{provider}/chat/completions", summary="标准OPEN AI聊天接口")
async def standard_chat(provider: str, request: Request, api_key=Depends(get_api_key)):

    data = await request.json()

    config = await load_config()
    
    logger = ChatLogger(config.get("log_path", "logs"), provider, api_key, data)

    try:
        site_config = config.get("providers", {}).get(provider, {})
        if 'base_url' not in site_config:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}")

        # 如果在白名单上，则使用预配置的 api_key
        if api_key in site_config.get("authorized_keys", []):
            api_key = site_config.get("api_key", api_key)

        client = OpenAI(api_key=api_key,
                        base_url=site_config.get("base_url", ""))
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
    except Exception as e:
        print(f"ERROR!!: provider {provider}, api-key {api_key}")
        print(_remove_image_data(data))
        raise e

    return response
