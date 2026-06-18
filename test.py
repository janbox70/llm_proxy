"""
测试 LLM Proxy 网关功能
"""
import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 测试配置
BASE_URL = "http://127.0.0.1:8088"
API_KEY = os.getenv("TEST_API_KEY", "cli-xyz")  # 从 .env 读取， fallback 到默认值
PROVIDER = "qwencodingplan"  # 使用配置中的 provider
MODEL = "qwen3.5-plus"  # 测试使用的模型

# Anthropic 测试配置
ANTHROPIC_PROVIDER = "anthropic"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY = os.getenv("TEST_ANTHROPIC_API_KEY", API_KEY)


async def test_health_check():
    """测试健康检查"""
    print("\n=== 测试 1: 健康检查 ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        assert response.status_code == 200
        print("✅ 健康检查通过")


async def test_provider_head_request():
    """测试 Provider HEAD 请求（修复 404 错误）"""
    print("\n=== 测试 1.5: Provider HEAD 请求 ===")
    url = f"{BASE_URL}/arkcodingplan"
    
    async with httpx.AsyncClient() as client:
        # 测试 HEAD 请求
        response = await client.head(url)
        print(f"HEAD 请求 - 状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Provider HEAD 请求测试通过")
        elif response.status_code == 404:
            print("❌ Provider HEAD 请求返回 404，未找到该 provider")
            raise Exception(f"Provider 'arkcodingplan' not found")
        else:
            print(f"⚠️  预期 200，实际: {response.status_code}")
            raise Exception(f"Unexpected status code: {response.status_code}")
        
        # 测试 GET 请求
        response = await client.get(url)
        print(f"GET 请求 - 状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Provider 信息: {json.dumps(data, ensure_ascii=False)}")
            print("✅ Provider GET 请求测试通过")
        else:
            print(f"❌ GET 请求失败: {response.status_code}")


async def test_chat_completion_non_streaming():
    """测试非流式聊天完成"""
    print("\n=== 测试 2: 非流式聊天完成 ===")
    url = f"{BASE_URL}/{PROVIDER}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ],
        "stream": False
    }

    print(f"请求 URL: {url}")
    print(f"请求 payload: {json.dumps(payload, ensure_ascii=False)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}...")

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"回复内容: {content}")
            print("✅ 非流式聊天完成测试通过")
        else:
            print(f"❌ 测试失败: {response.text}")
            raise Exception(f"请求失败: {response.status_code}")


async def test_chat_completion_streaming():
    """测试流式聊天完成"""
    print("\n=== 测试 3: 流式聊天完成 ===")
    url = f"{BASE_URL}/{PROVIDER}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "请数1到5"}
        ],
        "stream": True
    }

    print(f"请求 URL: {url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"状态码: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")

            full_content = ""
            chunk_count = 0
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    content = line[5:].strip()
                    if content and content != "[DONE]":
                        try:
                            data = json.loads(content)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            chunk_content = delta.get("content", "")
                            if chunk_content:
                                full_content += chunk_content
                                chunk_count += 1
                        except json.JSONDecodeError:
                            pass

            print(f"接收到的 chunk 数量: {chunk_count}")
            print(f"完整内容: {full_content}")
            print("✅ 流式聊天完成测试通过")


async def test_unauthorized():
    """测试未授权访问"""
    print("\n=== 测试 4: 未授权访问 ===")
    url = f"{BASE_URL}/{PROVIDER}/chat/completions"
    headers = {
        "Authorization": "Bearer invalid-key",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"状态码: {response.status_code}")

        if response.status_code == 400:
            print("✅ 未授权访问被正确拒绝")
        else:
            print(f"⚠️  预期 400，实际: {response.status_code}")


async def test_unsupported_provider():
    """测试不支持的 provider"""
    print("\n=== 测试 5: 不支持的 provider ===")
    url = f"{BASE_URL}/unknown_provider/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")

        if response.status_code == 400:
            print("✅ 不支持的 provider 被正确拒绝")
        else:
            print(f"⚠️  预期 400，实际: {response.status_code}")


async def test_missing_auth_header():
    """测试缺少 Authorization header"""
    print("\n=== 测试 6: 缺少 Authorization header ===")
    url = f"{BASE_URL}/{PROVIDER}/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"状态码: {response.status_code}")

        if response.status_code == 401:
            print("✅ 缺少 Authorization header 被正确拒绝")
        else:
            print(f"⚠️  预期 401，实际: {response.status_code}")


async def test_anthropic_non_streaming():
    """测试 Anthropic 非流式消息"""
    print("\n=== 测试 7: Anthropic 非流式消息 ===")
    url = f"{BASE_URL}/{ANTHROPIC_PROVIDER}/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ]
    }

    print(f"请求 URL: {url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            content_blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            usage = data.get("usage", {})
            print(f"回复内容: {text[:200]}")
            print(f"Usage: input={usage.get('input_tokens')}, output={usage.get('output_tokens')}")
            print(f"Stop reason: {data.get('stop_reason')}")
            print("✅ Anthropic 非流式消息测试通过")
        else:
            print(f"❌ 测试失败: {response.text[:500]}")
            raise Exception(f"请求失败: {response.status_code}")


async def test_anthropic_streaming():
    """测试 Anthropic 流式消息"""
    print("\n=== 测试 8: Anthropic 流式消息 ===")
    url = f"{BASE_URL}/{ANTHROPIC_PROVIDER}/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "请数1到5"}
        ],
        "stream": True
    }

    print(f"请求 URL: {url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"状态码: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")

            full_content = ""
            event_count = 0
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    content = line[5:].strip()
                    if content:
                        try:
                            data = json.loads(content)
                            event_type = data.get("type", "")
                            event_count += 1
                            if event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    full_content += delta.get("text", "")
                        except json.JSONDecodeError:
                            pass

            print(f"接收到的事件数量: {event_count}")
            print(f"完整内容: {full_content}")
            print("✅ Anthropic 流式消息测试通过")


async def test_anthropic_missing_api_key():
    """测试 Anthropic 缺少 x-api-key"""
    print("\n=== 测试 9: Anthropic 缺少 API Key ===")
    url = f"{BASE_URL}/{ANTHROPIC_PROVIDER}/v1/messages"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"状态码: {response.status_code}")

        if response.status_code == 401:
            print("✅ 缺少 x-api-key 被正确拒绝")
        else:
            print(f"⚠️  预期 401，实际: {response.status_code}")


async def test_cross_api_non_streaming():
    """测试跨 API 转换：通过 /v1/messages 调用 openai-compatible provider"""
    print("\n=== 测试 10: 跨 API 转换非流式 ===")
    # 使用 qwencodingplan（api_type: openai-compatible）
    url = f"{BASE_URL}/{PROVIDER}/v1/messages"
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ]
    }

    print(f"请求 URL: {url}")
    print(f"(通过 Anthropic 接口调用 OpenAI 兼容 provider)")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            # 验证返回的是 Anthropic 格式
            assert data.get("type") == "message", "响应类型应为 message"
            assert data.get("role") == "assistant", "角色应为 assistant"

            content_blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            usage = data.get("usage", {})
            stop_reason = data.get("stop_reason")

            print(f"回复内容: {text[:200]}")
            print(f"Usage: input={usage.get('input_tokens')}, output={usage.get('output_tokens')}")
            print(f"Stop reason: {stop_reason}")
            print(f"响应格式: Anthropic Messages API ✓")
            print("✅ 跨 API 转换非流式测试通过")
        else:
            print(f"❌ 测试失败: {response.text[:500]}")
            raise Exception(f"请求失败: {response.status_code}")


async def test_cross_api_streaming():
    """测试跨 API 转换流式：通过 /v1/messages 流式调用 openai-compatible provider"""
    print("\n=== 测试 11: 跨 API 转换流式 ===")
    url = f"{BASE_URL}/{PROVIDER}/v1/messages"
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "请数1到5"}
        ],
        "stream": True
    }

    print(f"请求 URL: {url}")
    print(f"(通过 Anthropic 流式接口调用 OpenAI 兼容 provider)")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"状态码: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")

            full_content = ""
            event_types = []
            has_message_start = False
            has_message_stop = False

            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    event_types.append(event_type)
                    if event_type == "message_start":
                        has_message_start = True
                    elif event_type == "message_stop":
                        has_message_stop = True

                elif line.startswith("data:"):
                    content = line[5:].strip()
                    if content:
                        try:
                            data = json.loads(content)
                            event_type = data.get("type", "")
                            if event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    full_content += delta.get("text", "")
                        except json.JSONDecodeError:
                            pass

            print(f"接收到的事件类型: {event_types}")
            print(f"完整内容: {full_content}")
            print(f"包含 message_start: {has_message_start}")
            print(f"包含 message_stop: {has_message_stop}")

            # 验证 Anthropic SSE 事件序列
            if has_message_start and has_message_stop and full_content:
                print("✅ 跨 API 转换流式测试通过")
            else:
                raise Exception("SSE 事件序列不完整")


async def main():
    """运行所有测试"""
    print("=" * 50)
    print("LLM Proxy 网关测试")
    print("=" * 50)
    print(f"目标地址: {BASE_URL}")
    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL}")
    print(f"API Key: {API_KEY}")

    tests = [
        test_health_check,
        test_provider_head_request,
        test_chat_completion_non_streaming,
        test_chat_completion_streaming,
        test_unauthorized,
        test_unsupported_provider,
        test_missing_auth_header,
        test_anthropic_non_streaming,
        test_anthropic_streaming,
        test_anthropic_missing_api_key,
        test_cross_api_non_streaming,
        test_cross_api_streaming,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 失败: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())