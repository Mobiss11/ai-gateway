"""Мост OpenAI chat/completions ↔ Claude Code CLI (api_format: "claude-cli").

Upstream — официальный ``claude`` CLI в скриптуемом режиме ``-p``. В отличие
от kraube (подмена клиента), запросы здесь реально выполняет официальный
клиент со своим system prompt, своими инструментами и своим учётом: шлюз
только переводит протокол. Сообщения OpenCode собираются в текст задачи,
а stream-json события CLI — обратно в OpenAI SSE/JSON.

Ограничение протокола: CLI — агент со своим циклом инструментов, а не
stateless-модель. OpenAI ``tools`` не транслируются: CLI выполняет задачу
своими инструментами в рабочем каталоге запроса и возвращает итоговый
текст. Рабочий каталог извлекается из env-блока системного промпта
OpenCode (``Working directory: /path``) и допускается только внутри
``allowed_project_roots`` (по умолчанию — домашний каталог пользователя).

Конфиг upstream::

    {"api_format": "claude-cli", "cli_path": "claude",
     "permission_mode": "acceptEdits", "max_concurrent": 2,
     "allowed_project_roots": ["/Users/you"]}

Модели: ``models.<alias>.model`` передаётся в ``--model`` как есть
(алиасы CLI: ``opus``/``sonnet``/``haiku`` или полное имя модели).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import anthropic_bridge
from fastapi.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("ai_gateway")

# Санитизация текста ошибок CLI: токены/ключи не должны покидать шлюз.
_TOKEN_IN_TEXT_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_KEY_IN_TEXT_RE = re.compile(r"\b(sk-[A-Za-z0-9._-]{6,})")

# «Working directory: /Users/...» в env-блоке системного промпта OpenCode.
_WORKDIR_RE = re.compile(r"(?m)^\s*Working directory:\s*(/\S+)\s*$")

_PERMISSION_MODES = ("default", "acceptEdits", "bypassPermissions")

# grace-период на корректное завершение CLI после EOF stdout / terminate.
_EXIT_GRACE_SECONDS = 10.0

CliReportFn = Callable[..., Awaitable[None]]


def sanitize_error_text(text: str) -> str:
    """Убирает возможные токены/ключи из текста исключения (как gateway)."""
    text = _TOKEN_IN_TEXT_RE.sub(r"\1<redacted>", text)
    text = _KEY_IN_TEXT_RE.sub("<redacted>", text)
    return text


# --------------------------------------------------------------------------- #
# Промпт: OpenAI-сообщения → текст задачи для CLI
# --------------------------------------------------------------------------- #
def _content_text(content: Any) -> str:
    """Текстовая часть content (строка или список блоков)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "image_url":
                    parts.append("[image]")
        return "\n".join(p for p in parts if p)
    return ""


def extract_working_directory(body: dict[str, Any]) -> str | None:
    """Путь из env-блока системного промпта OpenCode, если он есть."""
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") in ("system", "developer"):
            match = _WORKDIR_RE.search(_content_text(message.get("content")))
            if match:
                return match.group(1)
    return None


def _resolve_cwd(body: dict[str, Any], upstream: dict[str, Any]) -> str | None:
    """Рабочий каталог CLI: из промпта, но только внутри allowed roots."""
    requested = extract_working_directory(body) or upstream.get("default_cwd")
    if not requested:
        return None
    roots = upstream.get("allowed_project_roots")
    if not isinstance(roots, list) or not roots:
        roots = [str(Path.home())]
    try:
        path = Path(str(requested)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    path_str = str(path)
    for root in roots:
        try:
            root_str = str(Path(str(root)).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            continue
        if path_str == root_str or path_str.startswith(root_str.rstrip("/") + "/"):
            if path.is_dir():
                return path_str
    return None


def _tool_call_summary(call: Any) -> str:
    """Компактная строка tool_call для транскрипта."""
    if not isinstance(call, dict):
        return ""
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = fn.get("name") or call.get("name")
    raw_args = fn.get("arguments") if fn.get("arguments") is not None else call.get("input")
    args_text = ""
    if isinstance(raw_args, str):
        args_text = raw_args
    elif isinstance(raw_args, dict):
        try:
            args_text = json.dumps(raw_args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_text = str(raw_args)
    if len(args_text) > 2000:
        args_text = args_text[:2000] + "…"
    return f"{name}({args_text})" if name else args_text


def build_prompt(body: dict[str, Any]) -> str:
    """Транскрипт диалога OpenAI → текст задачи для ``claude -p``.

    Системный промпт остаётся инструкцией внутри задачи: system prompt CLI
    не подменяется — официальный клиент отправляет свой.
    """
    sections: list[str] = []
    conversation: list[str] = []
    tool_names: dict[str, str] = {}  # tool_call_id → имя (для tool-результатов)

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("system", "developer"):
            text = _content_text(message.get("content"))
            if text:
                sections.append("## Instructions\n\n" + text)
        elif role == "user":
            text = _content_text(message.get("content"))
            if text:
                conversation.append("### User\n\n" + text)
        elif role == "assistant":
            text = _content_text(message.get("content"))
            calls = [
                _tool_call_summary(c)
                for c in message.get("tool_calls") or []
                if isinstance(c, dict)
            ]
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    fn = call.get("function")
                    if isinstance(fn, dict) and fn.get("name"):
                        tool_names[str(call["id"])] = str(fn["name"])
            parts: list[str] = []
            if text:
                parts.append("### Assistant\n\n" + text)
            if calls:
                parts.append(
                    "### Assistant tool calls\n\n"
                    + "\n".join(f"- {c}" for c in calls if c)
                )
            if parts:
                conversation.append("\n\n".join(parts))
        elif role == "tool":
            name = tool_names.get(str(message.get("tool_call_id")), "tool")
            text = _content_text(message.get("content"))
            if len(text) > 8000:
                text = text[:8000] + "…"
            conversation.append(f"### Tool result ({name})\n\n{text}")

    if conversation:
        sections.append("## Conversation\n\n" + "\n\n".join(conversation))

    sections.append("Continue this task as asked. Reply with the final result.")
    return "\n\n".join(sections)


def build_command(upstream: dict[str, Any], provider_model: str) -> list[str]:
    """argv запуска CLI: -p, stream-json, частичные сообщения, модель."""
    cli_path = str(upstream.get("cli_path") or "claude")
    command = [
        cli_path,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        str(provider_model),
    ]
    permission_mode = str(upstream.get("permission_mode") or "acceptEdits")
    if permission_mode not in _PERMISSION_MODES:
        permission_mode = "acceptEdits"
    if permission_mode == "bypassPermissions":
        command.append("--dangerously-skip-permissions")
    elif permission_mode != "default":
        command.extend(["--permission-mode", permission_mode])
    extra = upstream.get("extra_args")
    if isinstance(extra, list):
        command.extend(str(a) for a in extra)
    return command


# --------------------------------------------------------------------------- #
# Ответ: stream-json CLI → OpenAI
# --------------------------------------------------------------------------- #
class CliEventTranslator:
    """Перевод stream-json строк CLI в OpenAI-чанки.

    Используется и для SSE (``drain_text_deltas``/``final_chunks``), и для
    не-стримового режима (аккумуляция текста + ``full_response``).
    """

    def __init__(self, model_alias: str, *, include_usage: bool = False) -> None:
        self._model_alias = str(model_alias)
        self._include_usage = include_usage
        self._id: str | None = None
        self._usage: dict[str, int] | None = None
        self._cost: float | None = None
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._saw_deltas = False
        self._result_text: str | None = None
        self._is_error = False
        self._subtype: str | None = None

    # -- разбор одной JSON-строки CLI ------------------------------------- #
    def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")

        if kind == "system" and event.get("subtype") == "init":
            session = event.get("session_id")
            if isinstance(session, str) and not self._id:
                self._id = session
        elif kind == "stream_event":
            self._handle_stream_event(event.get("event"))
        elif kind == "result":
            self._handle_result(event)

    def _handle_stream_event(self, event: Any) -> None:
        if not isinstance(event, dict) or event.get("type") != "content_block_delta":
            return
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return
        if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
            self._saw_deltas = True
            self._text_parts.append(delta["text"])
        elif delta.get("type") == "thinking_delta" and isinstance(
            delta.get("thinking"), str
        ):
            self._reasoning_parts.append(delta["thinking"])

    def _handle_result(self, event: dict[str, Any]) -> None:
        session = event.get("session_id")
        if isinstance(session, str):
            self._id = session
        self._subtype = str(event.get("subtype")) if event.get("subtype") else None
        self._is_error = bool(event.get("is_error")) or (self._subtype or "").startswith(
            "error"
        )
        if isinstance(event.get("result"), str):
            self._result_text = event["result"]
        if isinstance(event.get("usage"), dict):
            self._usage = anthropic_bridge.map_usage(event["usage"])
            raw_cost = event.get("total_cost_usd")
            if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
                self._cost = float(raw_cost)

    # -- доступ к результатам --------------------------------------------- #
    @property
    def session_id(self) -> str | None:
        return self._id

    def text(self) -> str:
        """Итоговый текст ответа (дельты либо result-текст)."""
        if self._saw_deltas and self._text_parts:
            return "".join(self._text_parts)
        return self._result_text or "".join(self._text_parts)

    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    def usage_for_report(self) -> dict[str, Any] | None:
        """usage для отчёта в Борт: OpenAI-форма + честный cost, если был."""
        if not self._usage and not self._cost:
            return None
        usage: dict[str, Any] = dict(self._usage or {})
        if self._cost:
            usage["cost"] = self._cost
        return usage

    def error_text(self) -> str | None:
        if not self._is_error:
            return None
        text = self._result_text or self._subtype or "claude-cli error"
        return sanitize_error_text(text)

    # -- OpenAI-формы ------------------------------------------------------ #
    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{(self._id or 'cli')[:32]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self._model_alias,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    def content_chunk(self, text: str) -> dict[str, Any]:
        return self._chunk({"role": "assistant", "content": text})

    def drain_text_deltas(self) -> list[str]:
        """Новые куски текста с прошлого вызова (для инкрементального SSE)."""
        parts = self._text_parts
        self._text_parts = []
        return parts

    def final_chunks(self) -> list[dict[str, Any]]:
        """Финальные чанки: result-текст (без дельт), usage, стоп."""
        chunks: list[dict[str, Any]] = []
        if not self._saw_deltas and self._result_text:
            chunks.append(self.content_chunk(self._result_text))
        if self._include_usage and (self._usage or self._cost):
            usage: dict[str, Any] = dict(self._usage or {})
            if self._cost:
                usage["cost"] = self._cost
            chunks.append(
                {
                    "id": f"chatcmpl-{(self._id or 'cli')[:32]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self._model_alias,
                    "choices": [],
                    "usage": usage,
                }
            )
        chunks.append(self._chunk({}, finish_reason="stop"))
        return chunks

    def full_response(self) -> dict[str, Any]:
        """Не-стримовый OpenAI-ответ целиком."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text()}
        if self._reasoning_parts:
            message["reasoning_content"] = self.reasoning()
        return {
            "id": f"chatcmpl-{(self._id or 'cli')[:32]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_alias,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": dict(self._usage or {}),
        }


# --------------------------------------------------------------------------- #
# Запуск CLI и обработка запроса
# --------------------------------------------------------------------------- #
async def _drain_stderr(stream: Any) -> str:
    chunks: list[bytes] = []
    while True:
        try:
            line = await stream.readline()
        except Exception:  # pragma: no cover - защита от битого пайпа
            break
        if not line:
            break
        chunks.append(line)
    return b"".join(chunks).decode("utf-8", "replace")


async def _finish_proc(proc: asyncio.subprocess.Process) -> int:
    """Дождаться выхода CLI с grace-периодом, при зависании — kill."""
    try:
        return await asyncio.wait_for(proc.wait(), timeout=_EXIT_GRACE_SECONDS)
    except asyncio.TimeoutError:  # pragma: no cover - зависший CLI
        proc.kill()
        return await proc.wait()


def _sse_frame(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _sse_error_frame(message: str) -> bytes:
    payload = {
        "error": {
            "message": message,
            "type": "upstream_error",
            "code": "claude_cli_error",
        }
    }
    return _sse_frame(payload)


def _cli_error(message: str, *, status_code: int = 502) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "upstream_error",
                "code": "claude_cli_error",
            }
        },
    )


async def handle_chat(
    client: Any,
    body: dict[str, Any],
    *,
    upstream: dict[str, Any],
    upstream_name: str,
    provider_model: str,
    requested_model: Any,
    started: float,
    report_bort: CliReportFn,
    timeout_seconds: float,
    semaphore: asyncio.Semaphore | None = None,
) -> Response:
    """Точка входа из gateway: chat/completions → ``claude -p``.

    Отвечает за подпроцесс, таймаут, отмену, логи и отчёт в Борт.
    """
    alias = str(requested_model)
    command = build_command(upstream, provider_model)
    prompt = build_prompt(body)
    cwd = _resolve_cwd(body, upstream)
    is_stream = bool(body.get("stream"))
    stream_options = body.get("stream_options")
    include_usage = bool(
        isinstance(stream_options, dict) and stream_options.get("include_usage")
    )
    timeout_seconds = float(upstream.get("timeout_seconds") or timeout_seconds or 300)

    async def _bort(
        status: str = "success",
        usage: dict[str, Any] | None = None,
        request_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        await report_bort(
            client,
            upstream_name=upstream_name,
            provider_model=provider_model,
            started=started,
            status=status,
            error_code=error_code,
            usage=usage,
            request_id=request_id,
        )

    def _log(status: int) -> None:
        duration_ms = (time.perf_counter() - started) * 1000
        log.info(
            "chat_completions requested=%s routed=%s upstream=claude-cli "
            "status=%s duration_ms=%.1f",
            requested_model,
            provider_model,
            status,
            duration_ms,
        )

    ctx = semaphore if semaphore is not None else contextlib.nullcontext()
    async with ctx:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=os.environ.copy(),
            )
        except (OSError, ValueError) as exc:
            _log(502)
            await _bort(status="error", error_code="cli_spawn")
            return _cli_error(
                f"claude-cli upstream failed to start: "
                f"{sanitize_error_text(str(exc))}"
            )

        stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))
        deadline = time.monotonic() + timeout_seconds
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except (OSError, RuntimeError) as exc:  # pragma: no cover
            proc.terminate()
            _log(502)
            await _bort(status="error", error_code="cli_stdin")
            return _cli_error(
                f"claude-cli stdin failed: {sanitize_error_text(str(exc))}"
            )

        translator = CliEventTranslator(alias, include_usage=include_usage)

        async def _read_stdout() -> AsyncIterator[str]:
            assert proc.stdout is not None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=max(0.1, remaining)
                    )
                except asyncio.TimeoutError:
                    raise
                if not line:
                    break
                yield line.decode("utf-8", "replace")

        async def _timeout_response() -> Response:
            proc.terminate()
            await _finish_proc(proc)
            stderr_task.cancel()
            _log(504)
            await _bort(status="error", error_code="timeout")
            return _cli_error(
                f"claude-cli upstream timed out after {timeout_seconds:.0f}s",
                status_code=504,
            )

        if is_stream:

            async def stream_body() -> AsyncIterator[bytes]:
                try:
                    async for line in _read_stdout():
                        translator.feed_line(line)
                        for delta_text in translator.drain_text_deltas():
                            yield _sse_frame(translator.content_chunk(delta_text))
                        if translator.error_text() is not None:
                            yield _sse_error_frame(translator.error_text() or "")
                            await _bort(
                                status="error",
                                error_code="cli_error",
                                request_id=translator.session_id,
                            )
                            _log(502)
                            await _finish_proc(proc)
                            return
                    for chunk in translator.final_chunks():
                        yield _sse_frame(chunk)
                    yield b"data: [DONE]\n\n"
                    await _finish_proc(proc)
                    _log(200)
                    await _bort(
                        usage=translator.usage_for_report(),
                        request_id=translator.session_id,
                    )
                except asyncio.TimeoutError:
                    yield _sse_error_frame(
                        f"claude-cli upstream timed out after {timeout_seconds:.0f}s"
                    )
                    _log(504)
                    await _bort(status="error", error_code="timeout")
                    proc.terminate()
                except GeneratorExit:
                    # клиент ушёл: останавливаем CLI, не тратим лимиты
                    proc.terminate()
                    try:
                        await asyncio.wait_for(
                            proc.wait(), timeout=_EXIT_GRACE_SECONDS
                        )
                    except asyncio.TimeoutError:  # pragma: no cover
                        proc.kill()
                        await proc.wait()
                    raise
                finally:
                    stderr_task.cancel()
                    if proc.returncode is None:
                        proc.terminate()

            return StreamingResponse(
                stream_body(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # --- не-стримовый режим ------------------------------------------ #
        try:
            async for line in _read_stdout():
                translator.feed_line(line)
        except asyncio.TimeoutError:
            return await _timeout_response()

        exit_code = await _finish_proc(proc)
        stderr = (await stderr_task) if not stderr_task.done() else ""

        error = translator.error_text()
        if error:
            _log(502)
            await _bort(
                status="error",
                error_code="cli_error",
                request_id=translator.session_id,
            )
            return _cli_error(error)
        if exit_code != 0 and not translator.text():
            tail = (stderr or "").strip().splitlines()[-1:] or ["<no stderr>"]
            _log(502)
            await _bort(status="error", error_code=f"cli_exit_{exit_code}")
            return _cli_error(
                f"claude-cli exited with code {exit_code}: "
                f"{sanitize_error_text(tail[0][:500])}"
            )

        response = translator.full_response()
        _log(200)
        await _bort(
            usage=translator.usage_for_report(),
            request_id=translator.session_id,
        )
        return JSONResponse(status_code=200, content=response)
