"""Тесты AI Gateway. Сети нет: upstream httpx подменяется MockTransport."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import gateway

TOKEN = "test-gateway-token"
DEEPSEEK_KEY = "fake-deepseek-key"
OPENROUTER_KEY = "fake-openrouter-key"

CONFIG = {
    "listen": {"host": "127.0.0.1", "port": 8130},
    "auth": {"token": TOKEN, "token_env": "AI_GATEWAY_TOKEN"},
    "request_timeout_seconds": 30,
    "upstreams": {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "headers": {"X-Test-Header": "yes"},
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "headers": {},
        },
    },
    "models": {
        "cheap": {"upstream": "deepseek", "model": "deepseek-chat", "name": "Cheap Chat"},
        "deepseek-v4-pro": {
            "upstream": "deepseek",
            "model": "deepseek-v4-pro",
            "name": "DeepSeek V4 Pro",
        },
    },
}

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def ok_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "deepseek-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        },
    )


@contextlib.contextmanager
def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    config: dict | None = None,
) -> Iterator[TestClient]:
    transport = httpx.MockTransport(handler)
    app = gateway.create_app(config or CONFIG, transport=transport)
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    monkeypatch.delenv("AI_GATEWAY_TEST_MISSING_KEY", raising=False)


# --------------------------------------------------------------------------- #
def test_healthz_without_auth() -> None:
    with make_client(ok_response) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "upstreams": ["deepseek", "openrouter"],
    }


def test_missing_and_wrong_token_gives_401() -> None:
    with make_client(ok_response) as client:
        no_token = client.get("/v1/models")
        wrong = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        bad_scheme = client.get("/v1/models", headers={"Authorization": TOKEN})
    for response in (no_token, wrong, bad_scheme):
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "invalid_request_error"


def test_models_endpoint_lists_configured_aliases() -> None:
    with make_client(ok_response) as client:
        response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["cheap", "deepseek-v4-pro"]
    assert body["data"][0] == {
        "id": "cheap",
        "object": "model",
        "owned_by": "deepseek",
        "name": "Cheap Chat",
    }


def test_alias_rewrites_model_and_routes_to_upstream() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return ok_response(request)

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    upstream_request = captured["request"]
    assert str(upstream_request.url) == "https://api.deepseek.com/v1/chat/completions"
    assert upstream_request.headers["authorization"] == f"Bearer {DEEPSEEK_KEY}"
    assert upstream_request.headers["x-test-header"] == "yes"
    forwarded = json.loads(upstream_request.content)
    assert forwarded["model"] == "deepseek-chat"
    assert forwarded["messages"] == [{"role": "user", "content": "hi"}]


def test_passthrough_routes_to_openrouter() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return ok_response(request)

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "openrouter/anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    upstream_request = captured["request"]
    assert str(upstream_request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert upstream_request.headers["authorization"] == f"Bearer {OPENROUTER_KEY}"
    assert json.loads(upstream_request.content)["model"] == "anthropic/claude-3.5-sonnet"


def test_unknown_model_returns_404_with_available_list() -> None:
    with make_client(ok_response) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "does-not-exist", "messages": []},
        )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert "cheap" in error["message"]
    assert "deepseek-v4-pro" in error["message"]


def test_streaming_is_passed_through_unchanged() -> None:
    sse = b'data: {"delta":"a"}\n\ndata: {"delta":"b"}\n\ndata: [DONE]\n\n'

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"}
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "stream": True, "messages": []},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == sse


def test_upstream_error_is_propagated_as_is() -> None:
    provider_body = {"error": {"message": "rate limited", "type": "rate_limit_error"}}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=provider_body)

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": []},
        )
    assert response.status_code == 429
    assert response.json() == provider_body


def test_missing_upstream_key_is_reported_without_crashing(monkeypatch) -> None:
    config = json.loads(json.dumps(CONFIG))
    config["upstreams"]["deepseek"]["api_key_env"] = "AI_GATEWAY_TEST_MISSING_KEY"
    monkeypatch.delenv("AI_GATEWAY_TEST_MISSING_KEY", raising=False)

    with make_client(ok_response, config=config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": []},
        )
        alive = client.get("/healthz")

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    assert "AI_GATEWAY_TEST_MISSING_KEY" in error["message"]
    assert alive.status_code == 200


def test_network_failure_becomes_502(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "cheap", "messages": []},
        )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


def test_create_app_requires_token() -> None:
    config = {
        "auth": {},
        "upstreams": {},
        "models": {},
    }
    with pytest.raises(RuntimeError, match="token is not configured"):
        gateway.create_app(config)


def test_load_env_file_does_not_overwrite_existing(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / "env"
    env_file.write_text(
        "# comment\nEXISTING=from-file\nNEW_VALUE=hello\nQUOTED=\"quoted value\"\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING", "from-env")
    monkeypatch.delenv("NEW_VALUE", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    gateway.load_env_file(env_file)

    assert os.environ["EXISTING"] == "from-env"
    assert os.environ["NEW_VALUE"] == "hello"
    assert os.environ["QUOTED"] == "quoted value"