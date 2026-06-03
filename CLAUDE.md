# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`llm_proxy` is a FastAPI gateway that proxies LLM API requests to multiple upstream providers. It supports two API formats:

- **OpenAI-compatible**: downstream clients call `POST http://{host}:8088/{provider}/chat/completions`
- **Anthropic Messages**: downstream clients call `POST http://{host}:8088/{provider}/v1/messages`

The proxy swaps credentials (via `authorized_keys` whitelist), forwards to the provider's `base_url`, and streams (or returns) the response while logging every call.

### Cross-API Conversion

Each provider has an `api_type` field in config (`openai-compatible` or `anthropic`). When a client calls `/{provider}/v1/messages` but the provider's `api_type` is `openai-compatible`, the proxy automatically:
1. Converts the Anthropic request to OpenAI format (`converters.anthropic_to_openai_request`)
2. Calls the OpenAI `chat/completions` endpoint
3. Converts the response back to Anthropic format (`converters.openai_to_anthropic_response`)

This allows Anthropic SDK clients to seamlessly use OpenAI-compatible models. Streaming is supported — OpenAI SSE events are converted to Anthropic SSE event sequence (`converters.openai_stream_to_anthropic_stream`).

## Running

```bash
# Start the server (port 8088, auto-reload)
./startup.sh
# or directly:
uvicorn main:app --port 8088 --host 127.0.0.1 --reload

# Run integration tests (server must be running)
python test.py
```

Tests read `TEST_API_KEY` from `.env` (falls back to `cli-xyz`) and hit `http://127.0.0.1:8088`. There is no unit test harness — `test.py` is a sequential async integration script using `httpx`.

## Dependencies

`pip install -r requirements.txt` — `fastapi`, `uvicorn`, `openai`, `pyyaml`, `aiofiles`, `mmh3`, `httpx`, `python-dotenv`.

## Architecture

### Request flow (`main.py`)

**OpenAI path** (`POST /{provider}/chat/completions`):
1. `get_api_key` extracts the bearer token from the `Authorization` header.
2. The `provider` path param is looked up in `config.yaml` under `providers`.
3. If the downstream key appears in the provider's `authorized_keys` list, it is replaced with the provider's real `api_key`; otherwise the downstream key is forwarded as-is (letting the upstream provider validate it).
4. Unknown request params are split into `extra_body`; `frequency_penalty` and `presence_penalty` are dropped unconditionally (Gemini does not support them). Empty `stop` is stripped.
5. `AsyncOpenAI` clients are cached in `_client_cache` keyed by `base_url:api_key` and closed on shutdown via the FastAPI lifespan.
6. Streaming responses pass through `_iter_events`, which yields raw SSE bytes while feeding them to `ChatLogger.add_chunk`. Non-streaming responses are logged via `response.model_dump()`.

**Anthropic path** (`POST /{provider}/v1/messages`):
1. `get_anthropic_api_key` extracts the key from `x-api-key` header (falls back to `Authorization: Bearer`).
2. Same provider lookup and `authorized_keys` credential swap.
3. Uses `httpx.AsyncClient` (cached per provider in `_httpx_clients`) with `x-api-key` and `anthropic-version` headers.
4. Forwards `anthropic-version` from the downstream client if present (otherwise defaults to `2023-06-01`).
5. Upstream URL is `{base_url}/v1/messages`. Non-2xx upstream responses are proxied verbatim (status + body).
6. Streaming passes through `_iter_anthropic_events`; non-streaming returns `JSONResponse`.

### Configuration (`config.yaml`)

- `log_path`: directory for JSONL logs and saved images (default `./output`, gitignored).
- `providers.<name>.base_url`: upstream endpoint.
- `providers.<name>.api_type`: `openai-compatible` (default) or `anthropic`. Determines which API format the provider uses. When a client calls `/v1/messages` but `api_type` is `openai-compatible`, the proxy auto-converts.
- `providers.<name>.api_key`: real upstream key (used when the downstream key is authorized).
- `providers.<name>.authorized_keys`: list of downstream keys allowed to consume this provider's `api_key`.

`ConfigManager` loads the YAML once at startup. The docstring claims hot-reload support, but `load()` is only invoked from `lifespan` — edits require a server restart (`--reload` handles this in dev).

### Logging (`logger.py`)

**`ChatLogger`** (OpenAI):
- Writes one JSONL line per request to `{log_path}/{provider}/chat.jsonl`.
- Rotation: when a file exceeds 10 MB, shift `chat.N.jsonl` → `chat.N+1.jsonl` up to `N=100` and start fresh.
- For streaming, `add_chunk` incrementally merges `content`, `reasoning_content`/`thought`, `tool_calls` (keyed by `index`), and `usage` from SSE deltas.
- `tools` is stripped from the logged input and replaced with `tools_list` (function names only) to keep logs compact.
- Base64 `data:` image URLs in `image_url` fields are decoded to `{log_path}/images/{ts}-{idx}.{ext}`; the URL in the log is replaced with the filename. A MurmurHash3 LRU (cap 200) deduplicates identical images within a request.
- `_remove_image_data` is a helper used in error paths to avoid printing large base64 payloads.

**`AnthropicChatLogger`** (Anthropic, inherits from `ChatLogger`):
- Reuses file writing, rotation, and image saving infrastructure from `ChatLogger`.
- `add_chunk` parses Anthropic SSE format: events carry a `type` field in their JSON payload (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`). Accumulates `text_delta` into `content`, `input_json_delta` into `tool_calls`, and captures `stop_reason` / `usage` from `message_delta`.
- Handles Anthropic's inline base64 image format (`{"type": "image", "source": {"type": "base64", "data": "..."}}`): saves to file and replaces `data` with the filename.
- `_remove_anthropic_image_data` clears inline base64 data in error paths.

### Format conversion (`converters.py`)

Handles Anthropic ↔ OpenAI bidirectional conversion:
- `anthropic_to_openai_request`: Converts messages (system + content blocks → OpenAI messages format), tools (Anthropic `input_schema` → OpenAI `parameters`), images (Anthropic `source.data` → OpenAI `data:` URL), and parameters.
- `openai_to_anthropic_response`: Converts OpenAI response to Anthropic format, including `stop_reason` mapping (`stop`→`end_turn`, `tool_calls`→`tool_use`, `length`→`max_tokens`).
- `openai_stream_to_anthropic_stream`: Async generator that converts OpenAI SSE to Anthropic SSE event sequence (`message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop`).

### Error handling

- `HTTPException` → JSON `{message}`.
- `openai.APIError` → if the exception carries a `response`, its body is streamed back verbatim with the original status/headers; otherwise a 400 JSON error is returned.
- Anthropic upstream errors (non-2xx) → the response body is proxied verbatim (Anthropic returns structured JSON errors).
- Cross-API conversion errors (OpenAI call fails when handling Anthropic request) → converted to Anthropic-formatted error response via `_format_anthropic_error`, with proper `type` field (`authentication_error`, `rate_limit_error`, etc.) and status code.

## Conventions / gotchas

- `output/`, `logs/`, and `.env` are gitignored. Logs land in `output/{provider}/chat.jsonl`.
- Served routes: `POST /{provider}/chat/completions` (OpenAI) and `POST /{provider}/v1/messages` (Anthropic). There is no models endpoint, no embeddings, no non-chat completion route.
- `config_manager.config` is `None` until `lifespan` runs — any code path that might execute before startup must handle that (both endpoints raise 500).
- When adding new request parameters that upstream providers may not support, add them to `unsupport_params` or route them through `extra_body`.
- Anthropic requires the `anthropic-version` header; the proxy defaults to `2023-06-01` but forwards the downstream client's value if provided.
- `httpx.AsyncClient` instances are cached per `provider:api_key` in `_httpx_clients`. When the api_key changes (via `authorized_keys` swap), a new client is created with updated `x-api-key` header.
