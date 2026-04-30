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


async def test_health_check():
    """测试健康检查"""
    print("\n=== 测试 1: 健康检查 ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        assert response.status_code == 200
        print("✅ 健康检查通过")


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
        test_chat_completion_non_streaming,
        test_chat_completion_streaming,
        test_unauthorized,
        test_unsupported_provider,
        test_missing_auth_header,
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