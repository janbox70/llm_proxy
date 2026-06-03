import base64
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import aiofiles
import mmh3


class ChatLogger:
    def __init__(self, log_path, provider: str, api_key: str, input_data: dict):
        self.log_path = log_path
        self.provider = provider
        self.api_key = api_key
        self.input_data = input_data
        self.output = {"content": "", "reasoning_content": "",
                       "usage": None, "tool_calls": {}}
        self._image_index = 0
        self.start_time = datetime.now()
        self.first_token_time = None
        self._image_ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        # MurmurHash3(data) -> saved filename (LRU, keep last 200)
        self._image_hash_index: "OrderedDict[int, str]" = OrderedDict()
        self._image_hash_index_limit = 200

    async def write(self, data: dict):
        """将一条日志写入 jsonl 文件，并在超出大小时做简单滚动。"""
        log_dir = Path(self.log_path) / str(self.provider)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "chat.jsonl"

        self._rotate_log_if_needed(log_dir, log_file)

        line = json.dumps(data, ensure_ascii=False)
        async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
            await f.write(line + "\n")

    def _rotate_log_if_needed(self, log_dir: Path, log_file: Path) -> None:
        """根据大小限制做简单文件滚动。"""
        max_bytes = 10 * 1024 * 1024  # 10MB
        max_history = 100

        if not (log_file.exists() and log_file.stat().st_size >= max_bytes):
            return

        # 删除最老的历史文件
        oldest = log_dir / f"chat.{max_history}.jsonl"
        if oldest.exists():
            oldest.unlink()

        # 从后往前依次重命名历史文件
        for i in range(max_history - 1, 0, -1):
            src = log_dir / f"chat.{i}.jsonl"
            if src.exists():
                src.rename(log_dir / f"chat.{i + 1}.jsonl")

        # 当前文件变为最新的历史文件
        log_file.rename(log_dir / "chat.1.jsonl")

    def add_chunk(self, sse: bytes):
        """
        解析 SSE 字节数据，将内容累加到 output（content / usage / tool_calls）
        """
        if self.first_token_time is None:
            self.first_token_time = datetime.now()
        for line in sse.decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            content = line[5:].strip()
            if content == "[DONE]":
                continue

            try:
                data = json.loads(content)

                if "usage" in data and data["usage"]:
                    self.output["usage"] = data["usage"]

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # A. 合并文本内容
                if "content" in delta and delta["content"]:
                    self.output["content"] += delta["content"]

                # B. 新增：合并思考过程 (Reasoning/Thinking)
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    self.output["reasoning_content"] += delta["reasoning_content"]
                # 适配某些其他模型可能使用的 thought 字段
                elif "thought" in delta and delta["thought"]:
                    self.output["reasoning_content"] += delta["thought"]

                if delta.get("tool_calls"):
                    for tool in delta["tool_calls"]:
                        idx = tool.get("index", 0)
                        if idx not in self.output["tool_calls"]:
                            self.output["tool_calls"][idx] = {
                                "function": {"name": "", "arguments": ""}
                            }

                        f = tool["function"]
                        if "name" in f:
                            self.output["tool_calls"][idx]["function"]["name"] += f["name"]
                        if "arguments" in f:
                            self.output["tool_calls"][idx]["function"]["arguments"] += f[
                                "arguments"
                            ]
            except json.JSONDecodeError:
                self.output["content"] += line

    async def write_log(self, output=None):
        # 先解析并保存 input_data 中的 image_url，将 url 替换为本地文件名
        await self._process_input_images(self.input_data)

        # 丢弃输入的tools字段
        tools = self.input_data.pop("tools", None)
        if tools is not None:
            self.input_data['tools_list'] = [
                t.get("function").get("name", "") for t in tools]

        data = {
            "st" : self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ft" : (self.first_token_time - self.start_time).total_seconds() if self.first_token_time else 0,
            "time": (datetime.now() - self.start_time).total_seconds(),
            "api_key": self.api_key,
            "input": self.input_data,
            "output": self.output if output is None else output,
        }
        await self.write(data)

    async def _process_input_images(self, obj):
        """
        递归遍历 input_data，查找 OpenAI 风格的 image_url 字段，
        将其中的 url 对应的图片保存到本地文件，并把 url 替换成本地文件名。
        """
        if isinstance(obj, dict):
            # 命中形如 {"type": "image_url" / "input_image", "image_url": {"url": "..."}}
            image_obj = obj.get("image_url")
            if isinstance(image_obj, dict):
                url = image_obj.get("url")
                # 只处理 data:image/...;base64,...，其他 URL 保持不变
                if isinstance(url, str) and url.startswith("data:"):
                    image_obj["url"] = await self._save_image(url)

            # 继续递归其他键（避免对 image_url 再次递归，防止重复处理）
            for key, value in obj.items():
                if key == "image_url":
                    continue
                obj[key] = await self._process_input_images(value)

        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                obj[idx] = await self._process_input_images(item)

        return obj

    async def _save_image(self, url: str) -> str:
        """
        把 image_url 指向的图片保存到 {log_path}/images/ 目录，
        返回保存时使用的本地文件名。如果保存失败，则返回原始 url。
        """
        h = None
        try:
            # 在 try 之前建立索引并查表：重复图片直接复用旧文件名
            # 这里只处理 data:URL，例如 data:image/png;base64,xxxxx
            if not url.startswith("data:"):
                return url
            h = mmh3.hash128(url, signed=False)
            existing = self._image_hash_index.get(h)
            if existing:
                self._image_hash_index.move_to_end(h)
                return existing
            # 不存在则创建索引（占位，文件名会在 return filename 之前写入）
            self._image_hash_index[h] = ""
            self._image_hash_index.move_to_end(h)
            while len(self._image_hash_index) > self._image_hash_index_limit:
                self._image_hash_index.popitem(last=False)
        except Exception:
            # 索引失败不影响主流程（继续走原有 try 保存逻辑）
            h = None
        try:
            image_dir = Path(self.log_path) / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            index = self._image_index
            self._image_index += 1

            # 这里只处理 data:URL，例如 data:image/png;base64,xxxxx
            if not url.startswith("data:"):
                return url

            header, b64data = url.split(",", 1)
            if "jpeg" in header or "jpg" in header:
                ext = "jpg"
            elif "webp" in header:
                ext = "webp"
            elif "gif" in header:
                ext = "gif"
            else:
                ext = "png"

            data = base64.b64decode(b64data)

            filename = f"{self._image_ts}-{index}.{ext}"
            file_path = image_dir / filename

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)

            # 记录索引，避免重复保存
            if h is not None:
                self._image_hash_index[h] = filename
            return filename
        except Exception:
            # 出错时，不影响主流程，直接返回原始 url
            return url


def _remove_image_data(obj):
    if isinstance(obj, dict):
        if "image_url" in obj:
            obj["image_url"]["url"] = ""
        return {k: _remove_image_data(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_remove_image_data(item) for item in obj]
    return obj


def _remove_anthropic_image_data(obj):
    """清除 Anthropic 格式中的 base64 图片数据，避免日志爆炸"""
    if isinstance(obj, dict):
        if obj.get("type") == "image" and isinstance(obj.get("source"), dict):
            if "data" in obj["source"]:
                obj["source"]["data"] = ""
        return {k: _remove_anthropic_image_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_remove_anthropic_image_data(item) for item in obj]
    return obj


class AnthropicChatLogger(ChatLogger):
    """Anthropic Messages API 专用日志记录器"""

    def __init__(self, log_path, provider: str, api_key: str, input_data: dict):
        # 跳过 ChatLogger.__init__，直接调用更上层的初始化
        self.log_path = log_path
        self.provider = provider
        self.api_key = api_key
        self.input_data = input_data
        self.output = {"content": "", "tool_calls": {}, "stop_reason": "", "usage": None}
        self._image_index = 0
        self.start_time = datetime.now()
        self.first_token_time = None
        self._image_ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self._image_hash_index: "OrderedDict[int, str]" = OrderedDict()
        self._image_hash_index_limit = 200
        # SSE 解析缓冲
        self._buffer = ""

    def add_chunk(self, sse: bytes):
        """
        解析 Anthropic SSE 字节数据，累加到 output。
        Anthropic SSE 每个事件由空行分隔，包含 event: 和 data: 行。
        data: 行的 JSON 中有 type 字段标识事件类型。
        """
        if self.first_token_time is None:
            self.first_token_time = datetime.now()

        self._buffer += sse.decode("utf-8", errors="ignore")

        # 按双换行分割完整事件块
        while "\n\n" in self._buffer:
            event_str, self._buffer = self._buffer.split("\n\n", 1)
            self._process_event(event_str)

    def _process_event(self, event_str: str):
        """处理一个完整的 SSE 事件块"""
        for line in event_str.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            content = line[5:].strip()
            if not content:
                continue

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "message_start":
                msg = data.get("message", {})
                if "usage" in msg and msg["usage"]:
                    self.output["usage"] = dict(msg["usage"])

            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    idx = data.get("index", 0)
                    self.output["tool_calls"][idx] = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": "",
                    }

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type", "")

                if delta_type == "text_delta":
                    self.output["content"] += delta.get("text", "")

                elif delta_type == "input_json_delta":
                    idx = data.get("index", 0)
                    if idx in self.output["tool_calls"]:
                        self.output["tool_calls"][idx]["input"] += delta.get(
                            "partial_json", ""
                        )

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                if "stop_reason" in delta:
                    self.output["stop_reason"] = delta["stop_reason"]
                if "usage" in delta and delta["usage"]:
                    if self.output["usage"] is None:
                        self.output["usage"] = {}
                    self.output["usage"].update(delta["usage"])

    async def write_log(self, output=None):
        # 处理 Anthropic 格式的图片（base64 内嵌在 source.data 中）
        await self._process_anthropic_images(self.input_data)

        # 丢弃 tools 完整定义，仅保留名称列表
        tools = self.input_data.pop("tools", None)
        if tools is not None:
            self.input_data["tools_list"] = [t.get("name", "") for t in tools]

        data = {
            "st": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ft": (
                (self.first_token_time - self.start_time).total_seconds()
                if self.first_token_time
                else 0
            ),
            "time": (datetime.now() - self.start_time).total_seconds(),
            "api_key": self.api_key,
            "input": self.input_data,
            "output": self.output if output is None else output,
        }
        await self.write(data)

    async def _process_anthropic_images(self, obj):
        """
        递归遍历 Anthropic 格式的 input_data，将 image content block 中的
        base64 data 保存为本地文件，替换为文件名。
        """
        if isinstance(obj, dict):
            if obj.get("type") == "image" and isinstance(obj.get("source"), dict):
                source = obj["source"]
                if source.get("type") == "base64" and "data" in source:
                    media_type = source.get("media_type", "image/png")
                    b64data = source["data"]
                    filename = await self._save_b64_image(b64data, media_type)
                    source["data"] = filename

            for key, value in obj.items():
                obj[key] = await self._process_anthropic_images(value)

        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                obj[idx] = await self._process_anthropic_images(item)

        return obj

    async def _save_b64_image(self, b64data: str, media_type: str) -> str:
        """
        将 base64 图片数据保存为文件，返回文件名。
        使用 hash 去重避免重复保存。
        """
        h = None
        try:
            h = mmh3.hash128(b64data, signed=False)
            existing = self._image_hash_index.get(h)
            if existing:
                self._image_hash_index.move_to_end(h)
                return existing
            self._image_hash_index[h] = ""
            self._image_hash_index.move_to_end(h)
            while len(self._image_hash_index) > self._image_hash_index_limit:
                self._image_hash_index.popitem(last=False)
        except Exception:
            h = None

        try:
            image_dir = Path(self.log_path) / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            index = self._image_index
            self._image_index += 1

            if "jpeg" in media_type or "jpg" in media_type:
                ext = "jpg"
            elif "webp" in media_type:
                ext = "webp"
            elif "gif" in media_type:
                ext = "gif"
            else:
                ext = "png"

            data = base64.b64decode(b64data)
            filename = f"{self._image_ts}-{index}.{ext}"
            file_path = image_dir / filename

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)

            if h is not None:
                self._image_hash_index[h] = filename
            return filename
        except Exception:
            return b64data
