"""Тесты обогащения /v1/models, транспорта и прозрачности проксирования."""

from __future__ import annotations

import contextlib
import gzip
import json
import os
from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import gateway
import test_catalog

TOKEN = "test-gateway-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

CONFIG = {
    "auth": {"token": TOKEN},
    "request_timeout_seconds": 30,
    "upstreams": {
        "deepseek": {"base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    },
    "models": {
        "deepseek-flash": {
            "upstream": "deepseek",
            "model": "deepseek-flash",
            "name": "DeepSeek Flash (homelab)",
        },
        "flash-alias": {"upstream": "deepseek", "model": "flash-real-model"},
    },
}


def ok_response(request: httpx.Request, content: dict | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json=content
        or {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": request.url.path,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "prompt_cache_hit_tokens": 5,
                "prompt_cache_miss_tokens": 6,
            },
        },
    )


@contextlib.contextmanager
def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    config: dict | None = None,
    catalog: dict | None = None,
) -> Iterator[TestClient]:
    transport = httpx.MockTransport(handler)
    app = gateway.create_app(config or CONFIG, transport=transport, catalog=catalog)
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")


# --------------------------------------------------------------------------- #
def test_models_enriched_from_catalog() -> None:
    with make_client(ok_response, catalog=test_catalog.build()) as client:
        response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    entries = {m["id"]: m for m in response.json()["data"]}
    flash = entries["deepseek-flash"]
    assert flash["context_window"] == 1048576
    assert flash["max_output_tokens"] == 393216
    assert flash["limit"] == {"context": 1048576, "output": 393216}
    assert flash["capabilities"]["input"] == ["text", "image"]
    assert flash["reasoning"]["supported_efforts"] == ["low", "high", "max"]
    assert flash["package"] == "@opencode/ai/providers/deepseek"
    assert flash["metadata"]["quality"] == "exact"
    # alias без записи в каталоге остаётся минимальным
    assert "context_window" not in entries["flash-alias"]


def test_invalid_catalog_path_does_not_break_models(tmp_path) -> None:
    config = json.loads(json.dumps(CONFIG))
    config["catalog_path"] = str(tmp_path / "missing.json")
    with make_client(ok_response, config=config) as client:
        response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    assert all("context_window" not in m for m in response.json()["data"])


def test_decompressed_response_drops_content_encoding() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(b'{"ok": true}'),
            headers={"content-encoding": "gzip", "content-type": "application/json"},
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "deepseek-flash", "messages": []},
        )
    assert response.status_code == 200
    assert "content-encoding" not in {k.lower() for k in response.headers}
    assert response.json() == {"ok": True}


def test_reasoning_and_tool_history_forwarded_unchanged() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return ok_response(request)

    messages = [
        {"role": "user", "content": "run the tool"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should call the tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "deepseek-flash", "messages": messages, "tools": tools},
        )
    assert response.status_code == 200
    forwarded = json.loads(captured["request"].content)
    assert forwarded["messages"] == messages
    assert forwarded["messages"][1]["reasoning_content"] == "I should call the tool"
    assert forwarded["tools"] == tools


def test_alias_rewrites_to_upstream_model_but_not_vice_versa() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return ok_response(request)

    with make_client(handler) as client:
        client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "flash-alias", "messages": []},
        )
    forwarded = json.loads(captured["request"].content)
    assert forwarded["model"] == "flash-real-model"


def test_streaming_passthrough_preserves_usage_tools_and_headers() -> None:
    sse = (
        b'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"id":"c1"}]}}]}\n\n'
        b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(sse),
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "deepseek-flash", "stream": True, "messages": []},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert "content-encoding" not in {k.lower() for k in response.headers}
    assert response.content == sse


def test_transport_error_text_is_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.HTTPError("bad gateway: Authorization Bearer sk-secret123456")

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "deepseek-flash", "messages": []},
        )
    text = response.text
    assert response.status_code == 502
    assert "sk-secret123456" not in text
    assert "<redacted>" in text


def test_catalog_route_mismatch_is_not_applied() -> None:
    cat = test_catalog.build()
    cat["models"]["deepseek-flash"]["upstream_model"] = "some-other-model"
    with make_client(ok_response, catalog=cat) as client:
        body = client.get("/v1/models", headers=AUTH).json()
    flash = next(m for m in body["data"] if m["id"] == "deepseek-flash")
    assert "context_window" not in flash
    assert flash["metadata"]["route_mismatch"] is True


def test_stale_provenance_is_exposed_honestly() -> None:
    cat = test_catalog.build()
    entry = cat["models"]["deepseek-flash"]
    entry["stale"] = True
    entry["retained_fields"] = ["limit.context"]
    entry["fetched_at"] = "2026-10-03T00:00:00+00:00"
    entry["stale_fields_from"] = "2026-10-01T00:00:00+00:00"
    entry["source_type"] = "fixture"
    with make_client(ok_response, catalog=cat) as client:
        body = client.get("/v1/models", headers=AUTH).json()
    flash = next(m for m in body["data"] if m["id"] == "deepseek-flash")
    assert flash["metadata"]["stale"] is True
    assert flash["metadata"]["fetched_at"] == "2026-10-03T00:00:00+00:00"
    assert flash["metadata"]["source"] == "deepseek"
    assert flash["metadata"]["source_type"] == "fixture"
    assert flash["metadata"]["retained_fields"] == ["limit.context"]


def test_models_reload_catalog_on_mtime_change(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.json"
    cat = test_catalog.build()
    catalog_path.write_text(json.dumps(cat), encoding="utf-8")
    config = json.loads(json.dumps(CONFIG))
    config["catalog_path"] = str(catalog_path)

    with make_client(ok_response, config=config) as client:
        first = next(
            m
            for m in client.get("/v1/models", headers=AUTH).json()["data"]
            if m["id"] == "deepseek-flash"
        )
        assert first["context_window"] == 1048576

        cat["models"]["deepseek-flash"]["limit"]["context"] = 999999
        catalog_path.write_text(json.dumps(cat), encoding="utf-8")
        os.utime(catalog_path, ns=(catalog_path.stat().st_atime_ns, catalog_path.stat().st_mtime_ns + 10**9))

        second = next(
            m
            for m in client.get("/v1/models", headers=AUTH).json()["data"]
            if m["id"] == "deepseek-flash"
        )
    assert second["context_window"] == 999999


def test_missing_catalog_file_does_not_crash_models(tmp_path) -> None:
    config = json.loads(json.dumps(CONFIG))
    config["catalog_path"] = str(tmp_path / "nope.json")
    with make_client(ok_response, config=config) as client:
        assert client.get("/v1/models", headers=AUTH).status_code == 200
