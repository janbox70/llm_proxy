"""
Anthropic ↔ OpenAI 格式转换模块

实现跨 API 格式的请求/响应转换，使客户端可以通过 Anthropic Messages API 接口
调用 OpenAI 兼容的模型。
"""
import json
from typing import AsyncGenerator, Any
from openai import AsyncStream


def anthropic_to_openai_request(anthropic_req: dict) -> dict:
    """
    将 Anthropic Messages 请求转换为 OpenAI Chat Completions 格式

    主要转换：
    - system → messages 中的 system role
    - content blocks (text/image) → OpenAI content 格式
    - tools 定义格式
    - 参数名称映射
    """
    openai_req = {}

    # model
    openai_req["model"] = anthropic_req.get("model", "")

    # messages: Anthropic 的 system + messages → OpenAI 的 messages
    messages = []

    # system prompt
    system = anthropic_req.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # system 可能是 content block 数组
            text_parts = _extract_text_from_content_blocks(system)
            if text_parts:
                messages.append({"role": "system", "content": text_parts})

    # messages 转换
    for msg in anthropic_req.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # 转换 content 格式
        openai_content = _convert_anthropic_content_to_openai(content)
        messages.append({"role": role, "content": openai_content})

    openai_req["messages"] = messages

    # tools 转换
    if "tools" in anthropic_req:
        openai_req["tools"] = _convert_anthropic_tools_to_openai(anthropic_req["tools"])

    # tool_choice 转换
    if "tool_choice" in anthropic_req:
        openai_req["tool_choice"] = _convert_anthropic_tool_choice_to_openai(
            anthropic_req["tool_choice"]
        )

    # stream
    if "stream" in anthropic_req:
        openai_req["stream"] = anthropic_req["stream"]

    # 参数映射
    if "max_tokens" in anthropic_req:
        openai_req["max_tokens"] = anthropic_req["max_tokens"]
    if "temperature" in anthropic_req:
        openai_req["temperature"] = anthropic_req["temperature"]
    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]
    if "stop_sequences" in anthropic_req:
        openai_req["stop"] = anthropic_req["stop_sequences"]

    return openai_req


def _extract_text_from_content_blocks(blocks: list) -> str:
    """从 Anthropic content blocks 中提取纯文本"""
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _convert_anthropic_content_to_openai(content: Any) -> Any:
    """
    将 Anthropic content 转换为 OpenAI content 格式

    Anthropic 格式：
    - 字符串: "Hello"
    - content blocks 数组: [{"type": "text", "text": "..."}, {"type": "image", "source": {...}}]

    OpenAI 格式：
    - 字符串: "Hello"
    - content parts 数组: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content) if content else ""

    # 转换 content blocks
    openai_parts = []
    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            openai_parts.append({
                "type": "text",
                "text": block.get("text", "")
            })

        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                # 转换为 OpenAI 的 data URL 格式
                openai_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}"
                    }
                })
            elif source.get("type") == "url":
                openai_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": source.get("url", "")
                    }
                })

        elif block_type == "tool_result":
            # tool_result 需要特殊处理，这里简化为文本
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = _extract_text_from_content_blocks(tool_content)
            # tool_result 在 OpenAI 中是通过 tool messages 传递的
            # 这里先作为普通文本处理
            if tool_content:
                openai_parts.append({
                    "type": "text",
                    "text": f"[Tool result: {block.get('tool_use_id', '')}] {tool_content}"
                })

        else:
            # 其他类型保持原样或转换为文本
            pass

    # 如果只有一个文本 part，返回字符串
    if len(openai_parts) == 1 and openai_parts[0].get("type") == "text":
        return openai_parts[0]["text"]

    return openai_parts if openai_parts else ""


def _convert_anthropic_tools_to_openai(tools: list) -> list:
    """
    将 Anthropic tools 转换为 OpenAI tools 格式

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools


def _convert_anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    """
    将 Anthropic tool_choice 转换为 OpenAI tool_choice 格式

    Anthropic:
    - "auto" | "any" | "none"
    - {"type": "tool", "name": "tool_name"}

    OpenAI:
    - "auto" | "none"
    - {"type": "function", "function": {"name": "tool_name"}}
    """
    if tool_choice == "auto":
        return "auto"
    elif tool_choice == "any":
        return "required"  # OpenAI 没有 "any"，用 "required" 近似
    elif tool_choice == "none":
        return "none"
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")}
        }
    return "auto"


def openai_to_anthropic_response(openai_resp: dict, model: str) -> dict:
    """
    将 OpenAI Chat Completions 响应转换为 Anthropic Messages 格式
    """
    choices = openai_resp.get("choices", [])
    if not choices:
        return {
            "id": f"msg_{openai_resp.get('id', 'unknown')}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }

    choice = choices[0]
    message = choice.get("message", {})

    # 构建 content blocks
    content = []

    # 文本内容
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})

    # tool_calls
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        arguments = func.get("arguments", "{}")
        try:
            input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            input_data = {}

        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": input_data
        })

    # stop_reason 映射
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    # usage
    usage = openai_resp.get("usage", {})

    return {
        "id": f"msg_{openai_resp.get('id', 'unknown')}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    }


async def openai_stream_to_anthropic_stream(
    openai_stream: AsyncStream,
    model: str,
    request_id: str = None
) -> AsyncGenerator[bytes, None]:
    """
    将 OpenAI 流式响应转换为 Anthropic 格式的 SSE 流

    OpenAI SSE 事件：
    - data: {"choices": [{"delta": {"role": "assistant"}}], "id": "...", "model": "..."}
    - data: {"choices": [{"delta": {"content": "..."}}]}
    - data: {"choices": [{"delta": {"tool_calls": [...]}}]}
    - data: [DONE]

    Anthropic SSE 事件序列：
    1. message_start
    2. content_block_start (text or tool_use)
    3. content_block_delta (多次)
    4. content_block_stop
    5. message_delta
    6. message_stop
    """
    import time

    msg_id = f"msg_{request_id or 'unknown'}"
    content_block_index = 0
    in_text_block = False
    in_tool_block = False
    current_tool_index = -1
    tool_blocks_started = set()

    # 追踪 usage
    input_tokens = 0
    output_tokens = 0
    first_chunk = True
    stream_model = model

    async for chunk in openai_stream:
        # 解析 chunk
        chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk

        # 获取基本信息
        if "id" in chunk_dict:
            msg_id = f"msg_{chunk_dict['id']}"
        if "model" in chunk_dict:
            stream_model = chunk_dict["model"]

        # 获取 usage（部分提供商在第一个 chunk 返回）
        if "usage" in chunk_dict and chunk_dict["usage"]:
            usage = chunk_dict["usage"]
            if "prompt_tokens" in usage:
                input_tokens = usage["prompt_tokens"]

        choices = chunk_dict.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})

        # 发送 message_start（第一个 chunk 时）
        if first_chunk:
            first_chunk = False
            message_start = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": stream_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0
                    }
                }
            }
            yield _format_sse_event("message_start", message_start)

        # 处理 role（忽略，Anthropic 不需要在 delta 中传递 role）
        if "role" in delta:
            continue

        # 处理文本内容
        if "content" in delta and delta["content"]:
            if not in_text_block:
                # 如果之前在 tool block，先关闭它
                if in_tool_block:
                    yield _format_sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": content_block_index
                    })
                    content_block_index += 1
                    in_tool_block = False

                # 开始新的 text block
                yield _format_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {"type": "text", "text": ""}
                })
                in_text_block = True

            # 发送文本 delta
            yield _format_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": content_block_index,
                "delta": {"type": "text_delta", "text": delta["content"]}
            })

        # 处理 tool_calls
        if "tool_calls" in delta and delta["tool_calls"]:
            for tc in delta["tool_calls"]:
                tc_index = tc.get("index", 0)

                # 如果之前在 text block，先关闭它
                if in_text_block:
                    yield _format_sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": content_block_index
                    })
                    content_block_index += 1
                    in_text_block = False
                    tool_blocks_started.clear()

                # 新的 tool call
                if tc_index not in tool_blocks_started:
                    tool_blocks_started.add(tc_index)

                    # 开始新的 tool_use block
                    function = tc.get("function", {})
                    yield _format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": function.get("name", ""),
                            "input": {}
                        }
                    })

                # 发送 input_json_delta
                function = tc.get("function", {})
                if "arguments" in function and function["arguments"]:
                    yield _format_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": function["arguments"]
                        }
                    })

                current_tool_index = content_block_index

                # 如果是最后一个 tool chunk，增加 index
                # 注意：这里简化处理，实际需要根据下一个 chunk 判断
                content_block_index += 1
                in_tool_block = True

        # 处理 finish_reason（在最后一个 chunk）
        finish_reason = choices[0].get("finish_reason")
        if finish_reason:
            # 关闭当前 block
            if in_text_block or in_tool_block:
                yield _format_sse_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": max(0, content_block_index - 1)
                })

            # 获取 output usage（如果在最后一个 chunk）
            if "usage" in chunk_dict and chunk_dict["usage"]:
                usage = chunk_dict["usage"]
                if "completion_tokens" in usage:
                    output_tokens = usage["completion_tokens"]

            # 映射 stop_reason
            stop_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "content_filter": "end_turn",
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            # 发送 message_delta
            yield _format_sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens}
            })

    # 发送 message_stop
    yield _format_sse_event("message_stop", {"type": "message_stop"})


def _format_sse_event(event_type: str, data: dict) -> bytes:
    """将事件格式化为 SSE 字节流"""
    event_line = f"event: {event_type}"
    data_line = f"data: {json.dumps(data)}"
    return f"{event_line}\n{data_line}\n\n".encode("utf-8")
