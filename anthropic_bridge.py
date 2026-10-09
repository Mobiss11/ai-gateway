"""Мост OpenAI Chat Completions ↔ Anthropic Messages API.

Переводит запросы и ответы между OpenAI-форматом (единый API шлюза для
клиентов) и Anthropic Messages (upstream'ы с ``api_format: "anthropic"``,
например локальный демон ``kraube serve``).

Модуль не ходит в сеть: это чистые функции над dict/bytes. HTTP-часть
остаётся в ``gateway.py``.

Поддерживается: текст, system-промпты, картинки (data: и http(s)-URL),
tools / tool_choice, tool-результаты, stop-последовательности, streaming
SSE (включая потоковые tool_calls), usage с cache-полями и extended
thinking (``reasoning_effort`` → ``thinking.budget_tokens``, блоки
thinking → ``reasoning_content`` в стиле DeepSeek). Для Claude 5+
(adaptive thinking) ``reasoning_effort`` уходит как ``output_config.effort``,
legacy-бюджет там не форвардится (см. ``_is_effort_capable``).

**Env-секция агентных клиентов** (``move_env_to_user``): Anthropic
классифицирует env-секцию OpenCode в system-промпте (``Today's date: …``,
строка «Here is some useful information about the environment you are
running in:», блок ``<env>…</env>``) как «third-party app»: такие запросы
тарифицируются через extra-usage кредиты, а не лимиты плана, и без
кредитов детерминированно 400-ят — идентичность агента при этом
безразлична. Флаг upstream'а переносит секцию из system в первое
user-сообщение (содержимое и ``<env>``-теги сохраняются): контекст
остаётся у модели, классификация не срабатывает. По умолчанию выключено —
system уходит upstream'у как есть.

**Prompt caching** (как в Claude Code): Anthropic-кэш неавтоматический —
мост сам ставит до трёх ``cache_control: ephemeral`` брейкпоинтов: на
system-блоке (кэшируется весь префикс с identity-преамбулой kraube), на
последнем инструменте и на последнем блоке предпоследнего сообщения
(вся история, кроме нового хода). Повторяющийся префикс тарифицируется
как cache read (~0.1x) и заметно экономит окна подписки; usage-поля
``cache_read_input_tokens``/``cache_creation_input_tokens`` пробрасываются
честно.

Отбрасывается молча (Anthropic-аналогов нет): ``n``, ``logprobs``,
``response_format``, ``seed``, ``frequency_penalty``, ``presence_penalty``,
``logit_bias``, ``user``, ``store``. Неподдерживаемые роли/типы контента
дают явную ошибку перевода (400), а не молчаливую потерю.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

DEFAULT_MAX_TOKENS = 8192

# Adaptive thinking (Claude 5+): глубина задаётся output_config.effort, а не
# legacy-бюджетом thinking. Дефолтный потолок ответа для таких моделей —
# 32768, если клиент не прислал max_tokens/max_completion_tokens.
EFFORT_DEFAULT_MAX_TOKENS = 32768

# reasoning_effort → output_config.effort. xhigh/max передаются как есть.
EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

# Extended thinking: бюджет в tokens по reasoning_effort. Минимум API — 1024.
MIN_THINKING_BUDGET = 1024
THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "xhigh": 32768,
    "max": 32768,
}
_DISABLED_EFFORTS = {"", "none", "off", "disabled", "false", "default"}

# Adaptive-thinking модели: claude-<family>-<major>…, major >= 5.
_EFFORT_MODEL_RE = re.compile(r"(?i)^claude-(?:opus|sonnet|haiku|fable)-(\d+)(?:$|-)")

# env-секция OpenCode в system-промпте: «Today's date: …», строка-интро
# «Here is some useful information about the environment you are running in:»
# и блок <env>…</env>. В system Anthropic тарифицирует её как third-party app;
# в user-сообщении она — обычный контекст. Маркеры матчатся по отдельным
# строкам (только в связке с <env>-блоком), формулировки интро — толерантно.
_ENV_INTRO_LINE_RE = re.compile(
    r"(?i)here is some useful information about the environment"
)
_ENV_DATE_LINE_RE = re.compile(r"(?i)^\s*today'?s date:")

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class TranslationError(ValueError):
    """Запрос/ответ нельзя перевести в целевой формат."""


def _cache_control() -> dict[str, str]:
    """Брейкпоинт Anthropic prompt cache (5-минутный ephemeral, как Claude Code)."""
    return {"type": "ephemeral"}


def _apply_prompt_cache(
    request: dict[str, Any], *, cache: bool = True
) -> dict[str, Any]:
    """Ставит cache_control-брейкпоинты: system → tools → история.

    Максимум 4 брейкпоинта на запрос (лимит API) — здесь всегда ≤3:
    1) последний system-блок (кэширует identity-преамбулу kraube + system);
    2) последний инструмент (кэширует system + все tools);
    3) последний блок предпоследнего сообщения (вся история, кроме нового
    хода). Каждая следующая реплика продлевает кэш инкрементально —
    повторяющийся префикс идёт как cache read (~0.1x цены).
    Префиксы короче минимума модели (1024/2048 токенов) просто не
    кэшируются — это не ошибка.
    """
    if not cache:
        return request
    used = 0
    system = request.get("system")
    if isinstance(system, list) and system and isinstance(system[-1], dict):
        system[-1] = {**system[-1], "cache_control": _cache_control()}
        used += 1
    tools = request.get("tools")
    if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
        tools[-1] = {**tools[-1], "cache_control": _cache_control()}
        used += 1
    messages = request.get("messages")
    if (
        used < 3
        and isinstance(messages, list)
        and len(messages) >= 2
        and isinstance(messages[-2], dict)
        and isinstance(messages[-2].get("content"), list)
        and messages[-2]["content"]
    ):
        blocks = messages[-2]["content"]
        if isinstance(blocks[-1], dict):
            blocks[-1] = {**blocks[-1], "cache_control": _cache_control()}
    return request


# --------------------------------------------------------------------------- #
# Запрос: OpenAI → Anthropic
# --------------------------------------------------------------------------- #
def _content_text(content: Any) -> str:
    """Текст из OpenAI-контента (строка или массив текстовых частей)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return ""


def _image_block(url: Any) -> dict[str, Any]:
    if not isinstance(url, str) or not url:
        raise TranslationError("image_url must be a non-empty string")
    if url.startswith("data:"):
        # data:<mediatype>[;base64],<data>
        header, _, data = url.partition(",")
        if not data or ";base64" not in header:
            raise TranslationError("image_url data-URL must be base64-encoded")
        media_type = header[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise TranslationError(f"unsupported image_url: {url[:64]!r}")


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        text = content
        return [{"type": "text", "text": text}] if text else []
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise TranslationError("user content parts must be objects")
            ptype = part.get("type")
            if ptype == "text":
                text = str(part.get("text", ""))
                if text:
                    blocks.append({"type": "text", "text": text})
            elif ptype == "image_url":
                blocks.append(_image_block((part.get("image_url") or {}).get("url")))
            else:
                raise TranslationError(
                    f"unsupported user content part type: {ptype!r}"
                )
        return blocks
    if content is None:
        return []
    raise TranslationError(f"unsupported user content: {type(content).__name__}")


def _tool_use_block(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise TranslationError("tool_calls items must be objects")
    function = call.get("function") or {}
    if not isinstance(function, dict):
        raise TranslationError("tool_call.function must be an object")
    raw_args = function.get("arguments", "")
    try:
        parsed = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else {}
    except (ValueError, TypeError):
        parsed = {"_unparsed": str(raw_args)}
    if not isinstance(parsed, dict):
        parsed = {"_unparsed": json.dumps(parsed, ensure_ascii=False)}
    return {
        "type": "tool_use",
        "id": str(call.get("id") or ""),
        "name": str(function.get("name") or ""),
        "input": parsed,
    }


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    text = _content_text(content)
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        blocks.append(_tool_use_block(call))
    return blocks


def _tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
    tool_use_id = message.get("tool_call_id")
    if not isinstance(tool_use_id, str) or not tool_use_id:
        raise TranslationError("tool message requires a non-empty tool_call_id")
    content = message.get("content")
    if isinstance(content, str):
        mapped: Any = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append({"type": "text", "text": str(part.get("text", ""))})
            else:
                raise TranslationError(
                    "tool result content parts must be text objects"
                )
        mapped = parts
    else:
        raise TranslationError(
            f"unsupported tool result content: {type(content).__name__}"
        )
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": mapped,
    }
    if message.get("is_error"):
        block["is_error"] = True
    return block


def _convert_messages(messages: Any) -> tuple[list[dict[str, Any]], str | None]:
    """OpenAI messages → Anthropic messages + system-промпт."""
    if not isinstance(messages, list):
        raise TranslationError("messages must be a list")
    system_parts: list[str] = []
    items: list[tuple[str, list[dict[str, Any]]]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            items.append(("user", list(pending_tool_results)))
            pending_tool_results.clear()

    for message in messages:
        if not isinstance(message, dict):
            raise TranslationError("messages items must be objects")
        role = message.get("role")
        if role in ("system", "developer"):
            text = _content_text(message.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            pending_tool_results.append(_tool_result_block(message))
            continue
        flush_tool_results()
        if role == "user":
            items.append(("user", _user_blocks(message.get("content"))))
        elif role == "assistant":
            items.append(("assistant", _assistant_blocks(message)))
        else:
            raise TranslationError(f"unsupported message role: {role!r}")
    flush_tool_results()

    # Anthropic не допускает подряд идущие одинаковые роли — сливаем блоки.
    merged: list[tuple[str, list[dict[str, Any]]]] = []
    for role, blocks in items:
        if not blocks:
            continue
        if merged and merged[-1][0] == role:
            merged[-1][1].extend(blocks)
        else:
            merged.append((role, list(blocks)))
    if not merged:
        raise TranslationError("no translatable messages (all were empty)")
    if merged[0][0] == "assistant":
        raise TranslationError(
            "conversation must start with a user message (Anthropic requirement)"
        )

    result = [{"role": role, "content": blocks} for role, blocks in merged]
    system = "\n\n".join(system_parts) if system_parts else None
    return result, system


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise TranslationError("tools must be a list")
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise TranslationError("tools items must be objects")
        if tool.get("type") == "function":
            function = tool.get("function") or {}
            if not isinstance(function, dict) or not function.get("name"):
                raise TranslationError("function tool requires a name")
            schema = function.get("parameters")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            converted.append(
                {
                    "name": str(function["name"]),
                    "description": str(function.get("description") or ""),
                    "input_schema": schema,
                }
            )
        else:
            # Не-function инструменты передаём как есть (вдруг они уже
            # Anthropic-формата) — upstream сам разберётся. Копия, чтобы не
            # мутировать тело запроса клиента.
            converted.append(dict(tool))
    return converted


def _convert_tool_choice(
    tool_choice: Any, parallel_tool_calls: Any
) -> dict[str, Any] | None:
    choice: dict[str, Any] | None = None
    if tool_choice is None:
        choice = None
    elif isinstance(tool_choice, str):
        mapping = {"auto": "auto", "none": "none", "required": "any"}
        if tool_choice not in mapping:
            raise TranslationError(f"unsupported tool_choice: {tool_choice!r}")
        choice = {"type": mapping[tool_choice]}
    elif isinstance(tool_choice, dict):
        ctype = tool_choice.get("type")
        if ctype == "function":
            name = (tool_choice.get("function") or {}).get("name")
            if not name:
                raise TranslationError("tool_choice function requires a name")
            choice = {"type": "tool", "name": str(name)}
        elif ctype in ("auto", "any", "tool", "none"):
            choice = dict(tool_choice)
        else:
            raise TranslationError(f"unsupported tool_choice type: {ctype!r}")
    else:
        raise TranslationError("tool_choice must be a string or an object")
    if parallel_tool_calls is False and choice is not None:
        choice["disable_parallel_tool_use"] = True
    return choice


def _resolve_effort(body: dict[str, Any]) -> str | None:
    """reasoning_effort из OpenAI- или OpenRouter-формы."""
    effort = body.get("reasoning_effort")
    if not isinstance(effort, str):
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
    if isinstance(effort, str):
        return effort.strip().lower()
    return None


def _is_effort_capable(provider_model: str) -> bool:
    """True для Claude-моделей с adaptive thinking (major-версия >= 5).

    Такие модели принимают ``output_config: {"effort": …}`` вместо legacy
    ``thinking``/``budget_tokens``. Мажорная версия берётся из
    ``provider_model`` (case-insensitive): ``claude-opus-5-5``,
    ``claude-sonnet-5-5``, ``claude-opus-5``, ``claude-fable-5-1``,
    ``claude-haiku-5-5`` — True. ``claude-opus-4-6``,
    ``claude-sonnet-4-6``, ``claude-haiku-4-5``, ``"m"`` и прочие имена —
    False.
    """
    if not isinstance(provider_model, str):
        return False
    match = _EFFORT_MODEL_RE.match(provider_model)
    if match is None:
        return False
    return int(match.group(1)) >= 5


def _resolve_adaptive_thinking(
    body: dict[str, Any],
) -> tuple[dict[str, Any] | None, None]:
    """(output_config, None) для effort-моделей.

    Явный ``output_config``-dict из тела клиента форвардится как есть и
    имеет приоритет над ``reasoning_effort``. Иначе ``reasoning_effort``
    (или ``reasoning.effort``) мапится в ``{"effort": …}``; выключенные
    значения (none/off/…) дают ``None``. thinking всегда ``None``:
    legacy-бюджет на этих моделях не поддерживается и не форвардится.
    """
    explicit = body.get("output_config")
    if isinstance(explicit, dict):
        return dict(explicit), None
    effort = _resolve_effort(body)
    if effort is None or effort in _DISABLED_EFFORTS:
        return None, None
    mapped = EFFORT_MAP.get(effort)
    if mapped is None:
        raise TranslationError(f"unsupported reasoning_effort: {effort!r}")
    return {"effort": mapped}, None


def _resolve_thinking(
    body: dict[str, Any], max_tokens: int
) -> tuple[dict[str, Any] | None, int]:
    """(thinking-конфиг, скорректированный max_tokens).

    Приоритет: явный Anthropic-``thinking`` из тела, затем
    ``reasoning_effort``/``reasoning.effort``. Бюджет всегда < max_tokens.
    """
    thinking: dict[str, Any] | None = None
    explicit = body.get("thinking")
    effort = _resolve_effort(body)

    if isinstance(explicit, dict) and explicit.get("type") == "enabled":
        thinking = dict(explicit)
    elif isinstance(explicit, dict):
        # {"type": "disabled"} или иное — выключено явно.
        thinking = None
    elif effort is not None and effort not in _DISABLED_EFFORTS:
        budget = THINKING_BUDGETS.get(effort)
        if budget is None:
            raise TranslationError(f"unsupported reasoning_effort: {effort!r}")
        thinking = {"type": "enabled", "budget_tokens": budget}

    if thinking is None:
        return None, max_tokens

    budget = thinking.get("budget_tokens")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        budget = THINKING_BUDGETS["medium"]
    budget = max(MIN_THINKING_BUDGET, min(budget, THINKING_BUDGETS["max"]))
    if max_tokens <= budget:
        # API требует max_tokens > budget_tokens; поднимаем, а не режем бюджет.
        max_tokens = budget + 1024
    thinking = {"type": "enabled", "budget_tokens": budget}
    return thinking, max_tokens


def _split_env_section(system: str) -> tuple[str, str]:
    """Отделяет env-секцию OpenCode от остального system-промпта.

    Возвращает ``(остаток system, секция)``. Секция — блок ``<env>…</env>``
    плюс непосредственно стоящие перед ним строка-интро и строка
    ``Today's date: …`` (пустые строки между ними не мешают). Незакрытый
    ``<env>`` не трогается — system уходит как есть. Несколько секций
    переносятся все, в исходном порядке.
    """
    if "<env>" not in system:
        return system, ""
    lines = system.split("\n")
    keep = [True] * len(lines)
    sections: list[str] = []
    i = 0
    while i < len(lines):
        if "<env>" not in lines[i]:
            i += 1
            continue
        j = i
        while j < len(lines) and "</env>" not in lines[j]:
            j += 1
        if j == len(lines):  # незакрытый <env> — нечего безопасно переносить
            return system, ""
        start = i
        # строка-интро непосредственно перед блоком
        p = i - 1
        while p >= 0 and not lines[p].strip():
            p -= 1
        if p >= 0 and _ENV_INTRO_LINE_RE.search(lines[p]):
            start = p
            # опционально «Today's date: …» перед интро
            q = p - 1
            while q >= 0 and not lines[q].strip():
                q -= 1
            if q >= 0 and _ENV_DATE_LINE_RE.match(lines[q]):
                start = q
        sections.append("\n".join(lines[start : j + 1]))
        for k in range(start, j + 1):
            keep[k] = False
        i = j + 1
    if not sections:
        return system, ""
    rest = "\n".join(line for line, flag in zip(lines, keep) if flag)
    rest = re.sub(r"\n{3,}", "\n\n", rest).strip()
    return rest, "\n\n".join(sections)


def _prepend_env_block(messages: list[dict[str, Any]], env_section: str) -> None:
    """Вставляет env-секцию текст-блоком в начало первого user-сообщения.

    tool_result-блоки должны оставаться первыми в сообщении — текст
    вставляется сразу после них.
    """
    first = messages[0]
    content = first.get("content")
    if not isinstance(content, list):  # защита: _convert_messages даёт list
        content = [{"type": "text", "text": str(content or "")}]
        first["content"] = content
    insert_at = 0
    while (
        insert_at < len(content)
        and isinstance(content[insert_at], dict)
        and content[insert_at].get("type") == "tool_result"
    ):
        insert_at += 1
    content.insert(insert_at, {"type": "text", "text": env_section})


def openai_to_anthropic(
    body: dict[str, Any],
    provider_model: str,
    *,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    move_env_to_user: bool = False,
) -> dict[str, Any]:
    """Тело POST /v1/chat/completions → тело POST /v1/messages.

    ``move_env_to_user`` — перенос env-секции OpenCode из system в первое
    user-сообщение (см. модульный docstring); по умолчанию выключено.
    """
    if not isinstance(body, dict):
        raise TranslationError("request body must be an object")

    messages, system = _convert_messages(body.get("messages"))
    if move_env_to_user and system:
        system, env_section = _split_env_section(system)
        if env_section:
            _prepend_env_block(messages, env_section)

    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        max_tokens = body.get("max_completion_tokens")
    client_max_tokens = (
        isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0
    )
    effort_capable = _is_effort_capable(provider_model)
    if not client_max_tokens:
        max_tokens = EFFORT_DEFAULT_MAX_TOKENS if effort_capable else default_max_tokens

    output_config: dict[str, Any] | None = None
    if effort_capable:
        # Adaptive thinking: thinking не форвардим, max_tokens клиента не трогаем.
        output_config, thinking = _resolve_adaptive_thinking(body)
    else:
        thinking, max_tokens = _resolve_thinking(body, max_tokens)

    request: dict[str, Any] = {
        "model": provider_model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        # Блоки, а не строка: на последнем блоке _apply_prompt_cache поставит
        # cache_control-брейкпоинт (строка брейкпоинт нести не может).
        request["system"] = [{"type": "text", "text": system}]
    if body.get("stream"):
        request["stream"] = True

    # thinking всегда включён у effort-моделей и включается legacy-бюджетом
    # у остальных: API требует temperature=1 / top_p=1 — в обоих случаях убираем.
    if thinking is None and not effort_capable:
        temperature = body.get("temperature")
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            request["temperature"] = temperature
        top_p = body.get("top_p")
        if isinstance(top_p, (int, float)) and not isinstance(top_p, bool):
            request["top_p"] = top_p

    stop = body.get("stop")
    if isinstance(stop, str) and stop:
        request["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        request["stop_sequences"] = [str(item) for item in stop if item]

    tools = body.get("tools")
    if tools:
        request["tools"] = _convert_tools(tools)
        tool_choice = _convert_tool_choice(
            body.get("tool_choice"), body.get("parallel_tool_calls")
        )
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
    elif body.get("tool_choice") not in (None, "none"):
        raise TranslationError("tool_choice specified without tools")

    if thinking is not None:
        request["thinking"] = thinking
    if output_config is not None:
        request["output_config"] = output_config
    return _apply_prompt_cache(request)


# --------------------------------------------------------------------------- #
# Ответ: Anthropic → OpenAI
# --------------------------------------------------------------------------- #
def map_usage(usage: Any) -> dict[str, int]:
    """usage Anthropic → OpenAI (prompt включает cache-токены)."""
    source = usage if isinstance(usage, dict) else {}
    input_tokens = source.get("input_tokens") or 0
    cache_read = source.get("cache_read_input_tokens") or 0
    cache_creation = source.get("cache_creation_input_tokens") or 0
    output_tokens = source.get("output_tokens") or 0
    prompt_tokens = input_tokens + cache_read + cache_creation
    mapped: dict[str, int] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }
    if cache_read:
        mapped["prompt_tokens_details"] = {"cached_tokens": cache_read}
        mapped["cache_read_input_tokens"] = cache_read
    if cache_creation:
        mapped["cache_creation_input_tokens"] = cache_creation
    return mapped


def _finish_reason(stop_reason: Any) -> str:
    if isinstance(stop_reason, str):
        return _STOP_REASON_MAP.get(stop_reason, "stop")
    return "stop"


def anthropic_to_openai(
    payload: Any, model_alias: str, *, created: int | None = None
) -> dict[str, Any]:
    """Тело ответа /v1/messages → тело ответа /v1/chat/completions."""
    if not isinstance(payload, dict):
        raise TranslationError("anthropic response must be an object")

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "thinking":
            thinking_parts.append(str(block.get("thinking", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(
                            block.get("input") or {}, ensure_ascii=False
                        ),
                    },
                }
            )
        # redacted_thinking и неизвестные блоки не переводим.

    message: dict[str, Any] = {"role": "assistant"}
    if text_parts or not tool_calls:
        message["content"] = "".join(text_parts)
    else:
        message["content"] = None
    if thinking_parts:
        message["reasoning_content"] = "".join(thinking_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    message_id = str(payload.get("id") or "")
    return {
        "id": f"chatcmpl-{message_id}" if message_id else "chatcmpl-anthropic",
        "object": "chat.completion",
        "created": int(created if created is not None else time.time()),
        "model": model_alias,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_reason(payload.get("stop_reason")),
            }
        ],
        "usage": map_usage(payload.get("usage")),
    }


def anthropic_error_to_openai(payload: Any) -> dict[str, Any]:
    """Ошибка Anthropic/kraube → OpenAI-форма {"error": {...}}."""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return {
                "error": {
                    "message": str(error.get("message") or "upstream error"),
                    "type": str(error.get("type") or "api_error"),
                    "code": error.get("code"),
                }
            }
        if error is not None:
            return {"error": {"message": str(error), "type": "api_error"}}
    return {"error": {"message": "upstream error", "type": "api_error"}}


# --------------------------------------------------------------------------- #
# Streaming: Anthropic SSE → OpenAI chat.completion.chunk SSE
# --------------------------------------------------------------------------- #
class AnthropicStreamTranslator:
    """Инкрементальный перевод SSE-потока Anthropic в OpenAI-чанки.

    ``feed(chunk)`` принимает сырые байты из upstream (границы чанков
    произвольные), возвращает готовые SSE-кадры ``data: {...}\\n\\n``.
    ``finish()`` вызывается после EOF upstream'а и докидывает финальные
    кадры, если событие ``message_stop`` не пришло.
    """

    def __init__(self, model_alias: str, *, include_usage: bool = False) -> None:
        self._model_alias = model_alias
        self._include_usage = include_usage
        self._created = int(time.time())
        self._id = "chatcmpl-anthropic"
        self._buffer = b""
        self._separator_len = 0
        self._tool_index = -1
        self._finish_reason: str | None = None
        self._usage: dict[str, int] | None = None
        self._done = False

    # -- публичное ------------------------------------------------------- #
    def feed(self, chunk: bytes) -> list[bytes]:
        if self._done or not chunk:
            return []
        self._buffer += chunk
        frames: list[bytes] = []
        while True:
            separator = self._find_separator()
            if separator < 0:
                break
            block, self._buffer = (
                self._buffer[:separator],
                self._buffer[separator + self._separator_len:],
            )
            frames.extend(self._handle_block(block))
        return frames

    def finish(self) -> list[bytes]:
        """EOF upstream: добить поток, если message_stop не пришёл."""
        self._buffer = b""
        if self._done:
            return []
        self._done = True
        frames = [self._sse(self._chunk({}, finish_reason=self._finish_reason or "stop"))]
        frames.extend(self._final_usage_frames())
        frames.append(b"data: [DONE]\n\n")
        return frames

    # -- доступ для usage-отчётов --------------------------------------- #
    def last_usage(self) -> dict[str, int] | None:
        """Финальный usage потока (из message_start + message_delta)."""
        return self._usage

    def last_id(self) -> str | None:
        """id ответа (chatcmpl-…), как он уходил клиенту."""
        return self._id

    # -- внутреннее ------------------------------------------------------ #
    def _find_separator(self) -> int:
        index = self._buffer.find(b"\n\n")
        if index >= 0:
            self._separator_len = 2
            return index
        index = self._buffer.find(b"\r\n\r\n")
        if index >= 0:
            self._separator_len = 4
            return index
        self._separator_len = 0
        return -1

    def _handle_block(self, block: bytes) -> list[bytes]:
        if not block.strip():
            return []
        event_name: str | None = None
        data_lines: list[bytes] = []
        for raw_line in block.split(b"\n"):
            line = raw_line.rstrip(b"\r")
            if line.startswith(b"event:"):
                event_name = line[6:].strip().decode("ascii", "replace")
            elif line.startswith(b"data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return []
        try:
            data = json.loads(b"\n".join(data_lines).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            # Битый кадр не должен ронять весь поток — пропускаем его.
            return []
        if not isinstance(data, dict):
            return []
        return self._handle_event(event_name, data)

    def _handle_event(self, event: str | None, data: dict[str, Any]) -> list[bytes]:
        name = event or str(data.get("type") or "")
        if name == "message_start":
            message = data.get("message") or {}
            if isinstance(message, dict) and message.get("id"):
                self._id = f"chatcmpl-{message['id']}"
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict) and usage:
                self._usage = map_usage(usage)
            return [self._sse(self._chunk({"role": "assistant", "content": ""}))]
        if name == "content_block_start":
            block = data.get("content_block") or {}
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self._tool_index += 1
                return [
                    self._sse(
                        self._chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": self._tool_index,
                                        "id": str(block.get("id") or ""),
                                        "type": "function",
                                        "function": {
                                            "name": str(block.get("name") or ""),
                                            "arguments": "",
                                        },
                                    }
                                ]
                            }
                        )
                    )
                ]
            return []
        if name == "content_block_delta":
            delta = data.get("delta") or {}
            if not isinstance(delta, dict):
                return []
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                return [self._sse(self._chunk({"content": str(delta.get("text") or "")}))]
            if delta_type == "thinking_delta":
                return [
                    self._sse(
                        self._chunk({"reasoning_content": str(delta.get("thinking") or "")})
                    )
                ]
            if delta_type == "input_json_delta":
                if self._tool_index < 0:
                    return []
                return [
                    self._sse(
                        self._chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": self._tool_index,
                                        "function": {
                                            "arguments": str(delta.get("partial_json") or "")
                                        },
                                    }
                                ]
                            }
                        )
                    )
                ]
            return []  # signature_delta и прочее
        if name == "message_delta":
            delta = data.get("delta") or {}
            if isinstance(delta, dict) and delta.get("stop_reason"):
                self._finish_reason = _finish_reason(delta.get("stop_reason"))
            usage = data.get("usage")
            if isinstance(usage, dict) and usage:
                merged = map_usage(usage)
                if self._usage:
                    merged["prompt_tokens"] = max(
                        self._usage.get("prompt_tokens", 0), merged.get("prompt_tokens", 0)
                    )
                    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                        if self._usage.get(key):
                            merged[key] = max(
                                self._usage.get(key, 0), merged.get(key, 0)
                            )
                    if self._usage.get("prompt_tokens_details"):
                        merged.setdefault(
                            "prompt_tokens_details", self._usage["prompt_tokens_details"]
                        )
                merged["total_tokens"] = (
                    merged.get("prompt_tokens", 0) + merged.get("completion_tokens", 0)
                )
                self._usage = merged
            return []
        if name == "message_stop":
            self._done = True
            frames = [self._sse(self._chunk({}, finish_reason=self._finish_reason or "stop"))]
            frames.extend(self._final_usage_frames())
            frames.append(b"data: [DONE]\n\n")
            return frames
        if name == "error":
            self._done = True
            error = data.get("error") or {}
            message = str(
                (error.get("message") if isinstance(error, dict) else None) or data.get("message") or "upstream stream error"
            )
            error_type = str(
                (error.get("type") if isinstance(error, dict) else None) or "api_error"
            )
            frames = [self._sse({"error": {"message": message, "type": error_type}})]
            frames.append(b"data: [DONE]\n\n")
            return frames
        return []  # ping / content_block_stop / неизвестные

    def _final_usage_frames(self) -> list[bytes]:
        if not self._include_usage or not self._usage:
            return []
        return [self._sse(self._chunk({}, usage=self._usage, with_choices=False))]

    def _chunk(
        self,
        delta: dict[str, Any],
        *,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
        with_choices: bool = True,
    ) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model_alias,
        }
        if with_choices:
            chunk["choices"] = [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ]
        else:
            chunk["choices"] = []
        chunk["usage"] = usage
        return chunk

    @staticmethod
    def _sse(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
