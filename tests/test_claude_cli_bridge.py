"""Тесты моста claude-cli: перевод протокола и запуск через gateway.

CLI подменяется стаб-скриптом: он читает stdin (промпт) и печатает
заранее заготовленные stream-json события. Сети нет.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import claude_cli_bridge
import gateway

TOKEN = "test-gateway-token"

STUB_TEMPLATE = """\
#!/usr/bin/env python3
import json, os, sys

prompt = sys.stdin.read()
mode = os.environ.get("CLAUDE_STUB_MODE", "success")

if mode == "sleep":
    import time
    time.sleep(30)
    sys.exit(0)

if mode == "crash":
    sys.stderr.write("boom sk-1234567890\\n")
    sys.exit(3)

events = []
if mode == "error":
    events.append({
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "result": "Your account is on hold",
        "session_id": "sess-err",
        "total_cost_usd": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    })
elif mode == "cwd":
    events.append({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": os.getcwd(),
        "session_id": "sess-cwd",
        "total_cost_usd": 0,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
else:
    events.append({
        "type": "system",
        "subtype": "init",
        "cwd": os.getcwd(),
        "session_id": "sess-ok",
    })
    events.append({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Привет, "},
        },
    })
    events.append({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "мир!"},
        },
    })
    events.append({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Привет, мир!",
        "session_id": "sess-ok",
        "total_cost_usd": 0,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 7,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 10,
        },
    })

for event in events:
    print(json.dumps(event, ensure_ascii=False), flush=True)
"""


def write_stub(tmp_path: Path) -> Path:
    stub = tmp_path / "claude-stub.py"
    stub.write_text(STUB_TEMPLATE, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


def make_config(stub: Path, project_root: Path | None, **overrides) -> dict:
    upstream = {
        "api_format": "claude-cli",
        "cli_path": str(stub),
        "permission_mode": "acceptEdits",
        "max_concurrent": 1,
    }
    if project_root is not None:
        upstream["allowed_project_roots"] = [str(project_root)]
    upstream.update(overrides.pop("upstream", {}))
    config = {
        "listen": {"host": "127.0.0.1", "port": 8131},
        "auth": {"token": TOKEN},
        "request_timeout_seconds": 20,
        "upstreams": {"claude-code": upstream},
        "models": {
            "claude-opus-cc": {"upstream": "claude-code", "model": "opus"}
        },
    }
    config.update(overrides)
    return config


def opencode_request(project_dir: Path | None = None) -> dict:
    system = "Env:\nWorking directory: /nonexistent/project\nPlatform: darwin"
    if project_dir is not None:
        system = f"Env:\nWorking directory: {project_dir}\nPlatform: darwin"
    return {
        "model": "claude-opus-cc",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Ответь одним словом: ОК"},
        ],
    }


AUTH = {"Authorization": f"Bearer {TOKEN}"}


# --------------------------------------------------------------------------- #
# Чистые функции моста
# --------------------------------------------------------------------------- #
def test_build_prompt_renders_roles() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "System instructions here"},
            {"role": "user", "content": "Сделай задачу"},
            {
                "role": "assistant",
                "content": "Делаю",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": '{"path": "/tmp/x"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "file edited",
            },
        ]
    }
    prompt = claude_cli_bridge.build_prompt(body)
    assert "## Instructions" in prompt
    assert "System instructions here" in prompt
    assert "### User" in prompt
    assert "Сделай задачу" in prompt
    assert "### Assistant tool calls" in prompt
    assert "edit(" in prompt
    assert "### Tool result (edit)" in prompt
    assert "file edited" in prompt


def test_extract_working_directory() -> None:
    body = {
        "messages": [
            {
                "role": "system",
                "content": "Env:\nWorking directory: /Users/demo/project\nOther: 1",
            },
            {"role": "user", "content": "hi"},
        ]
    }
    assert (
        claude_cli_bridge.extract_working_directory(body)
        == "/Users/demo/project"
    )
    assert claude_cli_bridge.extract_working_directory({"messages": []}) is None


def test_resolve_cwd_whitelist(tmp_path: Path) -> None:
    body = {
        "messages": [
            {"role": "system", "content": f"Working directory: {tmp_path}"}
        ]
    }
    upstream = {"allowed_project_roots": [str(tmp_path)]}
    resolved = claude_cli_bridge._resolve_cwd(body, upstream)
    assert resolved is not None and Path(resolved) == tmp_path

    # путь вне whitelist → None (fallback), даже если существует
    upstream_outside = {"allowed_project_roots": [str(tmp_path / "nested")]}
    (tmp_path / "nested").mkdir()
    assert claude_cli_bridge._resolve_cwd(body, upstream_outside) is None

    # несуществующий путь → None
    body_missing = {
        "messages": [{"role": "system", "content": "Working directory: /no/such/dir"}]
    }
    assert claude_cli_bridge._resolve_cwd(body_missing, upstream) is None


def test_build_command() -> None:
    command = claude_cli_bridge.build_command({"cli_path": "claude"}, "opus")
    assert command[0] == "claude"
    assert "-p" in command
    assert "stream-json" in command
    assert "--include-partial-messages" in command
    assert command[command.index("--model") + 1] == "opus"
    assert "--permission-mode" in command
    assert command[command.index("--permission-mode") + 1] == "acceptEdits"

    bypass = claude_cli_bridge.build_command(
        {"cli_path": "claude", "permission_mode": "bypassPermissions"}, "sonnet"
    )
    assert "--dangerously-skip-permissions" in bypass
    assert "--permission-mode" not in bypass

    extra = claude_cli_bridge.build_command(
        {"cli_path": "claude", "permission_mode": "default", "extra_args": ["--flag"]},
        "haiku",
    )
    assert "--flag" in extra
    assert "--permission-mode" not in extra


# --------------------------------------------------------------------------- #
# Переводчик событий
# --------------------------------------------------------------------------- #
def test_translator_success_flow() -> None:
    translator = claude_cli_bridge.CliEventTranslator("claude-opus-cc")
    for line in [
        json.dumps({"type": "system", "subtype": "init", "session_id": "sess-ok"}),
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Привет, "},
                },
            }
        ),
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "мир!"},
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Привет, мир!",
                "session_id": "sess-ok",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 10,
                },
            }
        ),
    ]:
        translator.feed_line(line)

    assert translator.error_text() is None
    assert translator.text() == "Привет, мир!"
    assert translator.session_id == "sess-ok"
    usage = translator.usage_for_report()
    assert usage is not None
    assert usage["prompt_tokens"] == 160
    assert usage["completion_tokens"] == 7

    response = translator.full_response()
    assert response["choices"][0]["message"]["content"] == "Привет, мир!"
    assert response["usage"]["prompt_tokens"] == 160


def test_translator_error_result() -> None:
    translator = claude_cli_bridge.CliEventTranslator("claude-opus-cc")
    translator.feed_line(
        json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "Your account is on hold",
                "session_id": "sess-err",
            }
        )
    )
    error = translator.error_text()
    assert error is not None and "on hold" in error


def test_translator_sanitizes_error_text() -> None:
    translator = claude_cli_bridge.CliEventTranslator("m")
    translator.feed_line(
        json.dumps(
            {
                "type": "result",
                "is_error": True,
                "result": "failed with bearer sk-1234567890abc key",
            }
        )
    )
    error = translator.error_text() or ""
    assert "sk-1234567890abc" not in error
    assert "<redacted>" in error


# --------------------------------------------------------------------------- #
# Интеграция через gateway (стаб-CLI)
# --------------------------------------------------------------------------- #
def test_gateway_claude_cli_nonstream(tmp_path: Path) -> None:
    stub = write_stub(tmp_path)
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(tmp_path), headers=AUTH
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Привет, мир!"
    assert payload["usage"]["prompt_tokens"] == 160
    assert payload["usage"]["completion_tokens"] == 7
    assert payload["usage"]["cache_read_input_tokens"] == 50


def test_gateway_claude_cli_stream(tmp_path: Path) -> None:
    stub = write_stub(tmp_path)
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**opencode_request(tmp_path), "stream": True},
            headers=AUTH,
        )
    assert response.status_code == 200
    body = response.text
    assert "data: " in body
    assert "Привет, " in body
    assert "data: [DONE]" in body
    # без include_usage usage-чанк не приходит
    assert '"usage"' not in body


def test_gateway_claude_cli_stream_with_usage(tmp_path: Path) -> None:
    stub = write_stub(tmp_path)
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    request = {
        **opencode_request(tmp_path),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request, headers=AUTH)
    assert response.status_code == 200
    assert '"usage"' in response.text
    assert "prompt_tokens" in response.text


def test_gateway_claude_cli_error_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = write_stub(tmp_path)
    monkeypatch.setenv("CLAUDE_STUB_MODE", "error")
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(tmp_path), headers=AUTH
        )
    assert response.status_code == 502
    assert "on hold" in response.json()["error"]["message"]


def test_gateway_claude_cli_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = write_stub(tmp_path)
    monkeypatch.setenv("CLAUDE_STUB_MODE", "crash")
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(tmp_path), headers=AUTH
        )
    assert response.status_code == 502
    message = response.json()["error"]["message"]
    assert "exited with code 3" in message
    # stderr санитизирован: ключ не вытекает
    assert "sk-1234567890" not in message


def test_gateway_claude_cli_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = write_stub(tmp_path)
    monkeypatch.setenv("CLAUDE_STUB_MODE", "sleep")
    config = make_config(
        stub, tmp_path, upstream={"timeout_seconds": 0.5}
    )
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(tmp_path), headers=AUTH
        )
    assert response.status_code == 504
    assert "timed out" in response.json()["error"]["message"]


def test_gateway_claude_cli_spawn_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path / "no-such-cli", tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(tmp_path), headers=AUTH
        )
    assert response.status_code == 502
    assert "failed to start" in response.json()["error"]["message"]


def test_gateway_claude_cli_unauthorized(tmp_path: Path) -> None:
    stub = write_stub(tmp_path)
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(tmp_path)
        )
    assert response.status_code == 401


def test_gateway_claude_cli_unknown_model(tmp_path: Path) -> None:
    stub = write_stub(tmp_path)
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "no-such-model", "messages": []},
            headers=AUTH,
        )
    assert response.status_code == 404


def test_gateway_claude_cli_reports_usage_to_bort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = write_stub(tmp_path)
    config = make_config(stub, tmp_path)
    config["bort_usage_url"] = "http://bort.test/api/v1/ai/usage"

    reported: list[dict] = []

    def handler(request: object) -> object:
        import httpx

        if getattr(request, "url").host == "bort.test":
            reported.append(json.loads(getattr(request, "content")))
            return httpx.Response(200, json={"ok": True})
        raise AssertionError("unexpected external request")

    import httpx

    app = gateway.create_app(config, transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(tmp_path), headers=AUTH
        )
    assert response.status_code == 200
    assert reported, "usage-событие должно уйти в Борт"
    event = reported[0]
    assert event["provider"] == "claude-code"
    assert event["model"] == "opus"
    assert event["input_tokens"] == 160
    assert event["output_tokens"] == 7
    assert event["cache_read_tokens"] == 50
    assert event["status"] == "success"


def test_gateway_claude_cli_cwd_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI получает cwd из env-блока и печатает его в result."""
    stub = write_stub(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CLAUDE_STUB_MODE", "cwd")
    config = make_config(stub, tmp_path)
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(project), headers=AUTH
        )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == str(project)


def test_gateway_claude_cli_cwd_outside_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Каталог вне allowed_project_roots → CLI работает в каталоге сервиса."""
    stub = write_stub(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    monkeypatch.setenv("CLAUDE_STUB_MODE", "cwd")
    config = make_config(stub, other_root)  # root не включает project
    app = gateway.create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=opencode_request(project), headers=AUTH
        )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content != str(project)
