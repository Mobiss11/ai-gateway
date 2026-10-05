"""Тесты отчёта usage в журнал Борта. Сети нет: MockTransport ловит и чат, и POST в Борт."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import gateway
import usage_report

TOKEN = "test-gateway-token"
DEEPSEEK_KEY = "fake-deepseek-key"
KRAUBE_KEY = "fake-kraube-key"
BORT_URL = "http://127.0.0.1:9100/api/v1/ai/usage"

AUTH = {"Authorization": f"Bearer {TOKEN}"}

CONFIG: dict[str, Any] = {
    "listen": {"host": "127.0.0.1", "port": 8130},
    "auth": {"token": TOKEN, "token_env": "AI_GATEWAY_TOKEN"},
    "request_timeout_seconds": 30,
    "bort_usage_url": BORT_URL,
    "upstreams": {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "headers": {},
        },
        "kraube": {
            "base_url": "http://127.0.0.1:8787",
            "api_key_env": "KRAUBE_SERVE_KEY",
            "api_format": "anthropic",
            "headers": {},
        },
    },
    "models": {
        "cheap": {"upstream": "deepseek", "model": "deepseek-chat", "name": "Cheap"},
        "claude-opus-5-5": {"upstream": "kraube", "model": "claude-opus-5-5"},
    },
}


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY)
    monkeypatch.setenv("KRAUBE_SERVE_KEY", KRAUBE_KEY)


def chat_ok() -> dict[str, Any]:
    return {
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_cache_hit_tokens": 80,
        },
    }


def chat_stream_sse() -> bytes:
    return (
        b'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-s1","choices":[],"usage":{"prompt_tokens":50,"completion_tokens":3,"total_tokens":53,"prompt_cache_hit_tokens":40,"cost":0.000123}}\n\n'
        b"data: [DONE]\n\n"
    )


def anthropic_message() -> dict[str, Any]:
    return {
        "id": "msg_bort",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 90,
            "cache_creation_input_tokens": 25,
            "output_tokens": 7,
        },
    }


def anthropic_stream_sse() -> bytes:
    parts = []
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_stream", "usage": {"input_tokens": 5, "cache_read_input_tokens": 7}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "hi"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 4}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    for name, data in events:
        parts.append(f"event: {name}")
        parts.append(f"data: {json.dumps(data)}")
        parts.append("")
        parts.append("")
    return "\n".join(parts).encode("utf-8")


def make_client(handler, config: dict | None = None) -> TestClient:
    app = gateway.create_app(config or CONFIG, transport=httpx.MockTransport(handler))
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Юнит: сборка payload и сниффер
# --------------------------------------------------------------------------- #
def test_build_payload_normalizes_usage_variants() -> None:
    # OpenAI-форма
    payload = usage_report.build_payload(
        provider="openrouter", model="m",
        usage={"prompt_tokens": 100, "completion_tokens": 10,
               "prompt_tokens_details": {"cached_tokens": 60}, "cost": 0.0005},
        request_id="req-1", duration_ms=1234.5,
    )
    assert payload == {
        "provider": "openrouter", "agent": "opencode", "model": "m",
        "input_tokens": 100, "output_tokens": 10,
        "cache_read_tokens": 60, "cache_write_tokens": 0,
        "cost_microusd": 500, "duration_ms": 1234,
        "status": "success", "error_code": None,
        "external_request_id": "req-1",
    }
    # DeepSeek-форма кэша
    payload = usage_report.build_payload(
        provider="deepseek", model="m",
        usage={"prompt_tokens": 10, "completion_tokens": 1, "prompt_cache_hit_tokens": 9},
    )
    assert payload["cache_read_tokens"] == 9
    # Anthropic-форма
    payload = usage_report.build_payload(
        provider="kraube", model="m",
        usage={"prompt_tokens": 125, "completion_tokens": 7,
               "cache_read_input_tokens": 90, "cache_creation_input_tokens": 25},
        status="error", error_code="http_429",
    )
    assert payload["cache_read_tokens"] == 90
    assert payload["cache_write_tokens"] == 25
    assert payload["status"] == "error"
    assert payload["error_code"] == "http_429"
    # стоимость выдумывать нельзя: нет cost → 0
    assert payload["cost_microusd"] == 0


def test_sniffer_extracts_usage_and_id_from_stream() -> None:
    sniffer = usage_report.OpenAIUsageSniffer()
    raw = chat_stream_sse()
    for i in range(0, len(raw), 13):  # произвольные границы чанков
        sniffer.feed(raw[i : i + 13])
    assert sniffer.request_id == "chatcmpl-s1"
    assert sniffer.usage is not None
    assert sniffer.usage["prompt_tokens"] == 50
    assert sniffer.usage["cost"] == 0.000123


# --------------------------------------------------------------------------- #
# Интеграция: шлюз → Борт
# --------------------------------------------------------------------------- #
def test_non_stream_reports_usage_to_bort() -> None:
    bort: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BORT_URL:
            bort.append(json.loads(request.content))
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(200, json=chat_ok())

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert len(bort) == 1
    event = bort[0]
    assert event["provider"] == "deepseek"
    assert event["model"] == "deepseek-chat"
    assert event["agent"] == "opencode"
    assert event["input_tokens"] == 100
    assert event["output_tokens"] == 5
    assert event["cache_read_tokens"] == 80
    assert event["external_request_id"] == "chatcmpl-test-1"
    assert event["status"] == "success"
    assert event["duration_ms"] is not None


def test_openai_stream_reports_sniffed_usage() -> None:
    bort: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BORT_URL:
            bort.append(json.loads(request.content))
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(
            200, content=chat_stream_sse(), headers={"content-type": "text/event-stream"}
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert b"[DONE]" in response.content
    assert len(bort) == 1
    event = bort[0]
    assert event["input_tokens"] == 50
    assert event["cache_read_tokens"] == 40
    assert event["cost_microusd"] == 123
    assert event["external_request_id"] == "chatcmpl-s1"


def test_anthropic_non_stream_reports_cache_fields() -> None:
    bort: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BORT_URL:
            bort.append(json.loads(request.content))
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(200, json=anthropic_message())

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert len(bort) == 1
    event = bort[0]
    assert event["provider"] == "kraube"
    assert event["model"] == "claude-opus-5-5"
    assert event["input_tokens"] == 125  # input + cache_read + cache_creation
    assert event["cache_read_tokens"] == 90
    assert event["cache_write_tokens"] == 25
    assert event["output_tokens"] == 7
    assert event["external_request_id"] == "chatcmpl-msg_bort"


def test_anthropic_stream_reports_translator_usage() -> None:
    bort: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BORT_URL:
            bort.append(json.loads(request.content))
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(
            200, content=anthropic_stream_sse(), headers={"content-type": "text/event-stream"}
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus-5-5", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert len(bort) == 1
    event = bort[0]
    assert event["input_tokens"] == 12  # 5 input + 7 cache_read
    assert event["cache_read_tokens"] == 7
    assert event["output_tokens"] == 4
    assert event["external_request_id"] == "chatcmpl-msg_stream"


def test_upstream_error_is_reported_with_error_code() -> None:
    bort: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BORT_URL:
            bort.append(json.loads(request.content))
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 429
    assert len(bort) == 1
    assert bort[0]["status"] == "error"
    assert bort[0]["error_code"] == "http_429"
    assert bort[0]["provider"] == "deepseek"


def test_bort_downtime_never_breaks_chat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BORT_URL:
            raise httpx.ConnectError("bort down", request=request)
        return httpx.Response(200, json=chat_ok())

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"


def test_reporting_disabled_by_default() -> None:
    config = json.loads(json.dumps(CONFIG))
    config.pop("bort_usage_url")
    bort: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == BORT_URL:
            bort.append(json.loads(request.content))
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(200, json=chat_ok())

    with make_client(handler, config=config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert bort == []
