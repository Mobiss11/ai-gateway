"""Отчёт usage-метаданных шлюза в журнал Борта (POST /api/v1/ai/usage).

Включается конфигом ``bort_usage_url`` (см. README). Событие шлётся после
каждого chat/completions — успех и ошибка, стрим и не-стрим — и содержит
только метаданные: провайдер, модель, токены, кэш-поля, длительность,
статус. Промпты и ответы не покидают шлюз.

Считаем честно: ``cost_microusd`` заполняется только если upstream сам
сообщил цену (OpenRouter кладёт ``usage.cost``); тарифы DeepSeek/Anthropic
шлюз не знает и не выдумывает — подписочные запросы идут с нулевой ценой.

Формат usage-полей различается между провайдерами; :func:`build_payload`
нормализует известные варианты (OpenAI ``prompt_tokens_details``,
DeepSeek ``prompt_cache_hit_tokens``, Anthropic
``cache_read/cache_creation_input_tokens``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

log = logging.getLogger("ai_gateway")

_REPORT_TIMEOUT_SECONDS = 3.0


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(round(value)))
    return 0


def _cache_read_tokens(usage: dict[str, Any]) -> int:
    for key in ("cache_read_input_tokens", "prompt_cache_hit_tokens"):
        if usage.get(key):
            return _int(usage.get(key))
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens"):
        return _int(details.get("cached_tokens"))
    return 0


def build_payload(
    *,
    provider: str,
    model: str,
    agent: str = "opencode",
    usage: dict[str, Any] | None = None,
    request_id: str | None = None,
    status: str = "success",
    error_code: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    """Метрики хода → тело POST /api/v1/ai/usage Борта."""
    usage = usage if isinstance(usage, dict) else {}
    cost = usage.get("cost")
    return {
        "provider": str(provider)[:255],
        "agent": str(agent)[:255] or "opencode",
        "model": str(model)[:255],
        "input_tokens": _int(usage.get("prompt_tokens")),
        "output_tokens": _int(usage.get("completion_tokens")),
        "cache_read_tokens": _cache_read_tokens(usage),
        "cache_write_tokens": _int(usage.get("cache_creation_input_tokens")),
        "cost_microusd": _int(round(float(cost) * 1_000_000)) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0,
        "duration_ms": int(duration_ms) if duration_ms is not None else None,
        "status": status if status in ("success", "error") else "success",
        "error_code": str(error_code)[:255] if error_code else None,
        "external_request_id": str(request_id)[:255] if request_id else None,
    }


async def report(client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> None:
    """POST события в Борт. Ошибка никогда не ломает ответ клиенту."""
    try:
        response = await client.post(url, json=payload, timeout=_REPORT_TIMEOUT_SECONDS)
        if response.status_code == 409:
            return  # дубликат external_request_id — событие уже записано
        if response.status_code not in (200, 201):
            log.warning(
                "bort usage report rejected status=%s provider=%s",
                response.status_code,
                payload.get("provider"),
            )
    except httpx.HTTPError as exc:
        log.warning(
            "bort usage report failed (%s) provider=%s",
            type(exc).__name__,
            payload.get("provider"),
        )


class OpenAIUsageSniffer:
    """Пассивный сборщик usage/id из OpenAI-SSE без изменения потока.

    Байты проходят насквозь нетронутыми; класс только разбирает кадры
    ``data: {...}`` и запоминает первый ``id`` и последний непустой
    ``usage`` (финальный чанк при ``stream_options.include_usage``).
    """

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] | None = None
        self.request_id: str | None = None

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buffer += chunk
        while True:
            index = self._buffer.find(b"\n\n")
            if index < 0:
                break
            frame, self._buffer = self._buffer[:index], self._buffer[index + 2 :]
            self._handle_frame(frame)

    def _handle_frame(self, frame: bytes) -> None:
        for line in frame.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                parsed = json.loads(data)
            except ValueError:
                continue
            if not isinstance(parsed, dict):
                continue
            if self.request_id is None and isinstance(parsed.get("id"), str):
                self.request_id = parsed["id"]
            usage = parsed.get("usage")
            if isinstance(usage, dict) and usage:
                self.usage = usage
