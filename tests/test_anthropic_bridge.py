"""Тесты моста OpenAI ↔ Anthropic Messages. Сети нет: HTTP подменён."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import anthropic_bridge
import gateway

TOKEN = "test-gateway-token"
KRAUBE_KEY = "fake-kraube-serve-key"

ANTHROPIC_CONFIG: dict[str, Any] = {
    "listen": {"host": "127.0.0.1", "port": 8130},
    "auth": {"token": TOKEN, "token_env": "AI_GATEWAY_TOKEN"},
    "request_timeout_seconds": 30,
    "upstreams": {
        "kraube": {
            "base_url": "http://127.0.0.1:8787",
            "api_key_env": "KRAUBE_SERVE_KEY",
            "api_format": "anthropic",
            "headers": {},
        },
    },
    "models": {
        "claude-opus": {
            "upstream": "kraube",
            "model": "claude-opus-4-6",
            "name": "Claude Opus 4.6 (subscription)",
        },
    },
}

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def anthropic_message(
    *,
    text: str = "Hi!",
    stop_reason: str = "end_turn",
    tool_use: dict[str, Any] | None = None,
    thinking: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig"})
    if text:
        content.append({"type": "text", "text": text})
    if tool_use is not None:
        content.append({"type": "tool_use", **tool_use})
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage
        or {"input_tokens": 10, "output_tokens": 7},
    }


# --------------------------------------------------------------------------- #
# openai_to_anthropic
# --------------------------------------------------------------------------- #
def test_basic_request_translation() -> None:
    body = {
        "model": "claude-opus",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "stop": ["\n\nHuman:"],
    }
    request = anthropic_bridge.openai_to_anthropic(body, "claude-opus-4-6")
    assert request == {
        "model": "claude-opus-4-6",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "system": [
            {
                "type": "text",
                "text": "Be terse.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "temperature": 0.2,
        "stop_sequences": ["\n\nHuman:"],
    }


def test_prompt_cache_breakpoints_like_claude_code() -> None:
    """system + последний инструмент + предпоследнее сообщение; всего ≤3."""
    body = {
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "t1", "description": "d", "parameters": {"type": "object", "properties": {}}},
            },
            {
                "type": "function",
                "function": {"name": "t2", "description": "d", "parameters": {"type": "object", "properties": {}}},
            },
        ],
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    # system-блок помечен
    assert request["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # последний инструмент помечен, первый — нет
    assert "cache_control" not in request["tools"][0]
    assert request["tools"][1]["cache_control"] == {"type": "ephemeral"}
    messages = request["messages"]
    # предпоследнее сообщение (a1) помечено, новое (q2) и q1 — нет
    assert messages[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in messages[0]["content"][-1]
    assert "cache_control" not in messages[2]["content"][-1]
    assert json.dumps(request).count('"ephemeral"') == 3


def test_prompt_cache_skips_short_conversations() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    assert json.dumps(request).count('"ephemeral"') == 0


def test_prompt_cache_marks_tool_result_block() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "res"},
            {"role": "assistant", "content": "done"},
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    # messages[-2] — user-сообщение с tool_result; его последний блок помечен
    blocks = request["messages"][-2]["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    # последнее сообщение не помечено
    assert "cache_control" not in request["messages"][-1]["content"][-1]


# --------------------------------------------------------------------------- #
# move_env_to_user: env-секция OpenCode уходит из system в user-сообщение
# --------------------------------------------------------------------------- #
OPENCODE_SYSTEM = (
    "You are an AI agent running in OpenCode, a coding agent harness.\n\n"
    "Today's date: Mon Oct 05 2026\n\n"
    "Here is some useful information about the environment you are running in:\n"
    "<env>\n"
    "Working directory: /Users/demo/project\n"
    "Workspace root folder: /Users/demo/project\n"
    "Is directory a git repo: yes\n"
    "Platform: darwin\n"
    "</env>\n\n"
    "Instructions from: /Users/demo/project/AGENTS.md\n"
    "# Борт — правила работы с интерфейсом\n"
    "Не меняйте стек."
)
OPENCODE_ENV_SECTION = (
    "Today's date: Mon Oct 05 2026\n\n"
    "Here is some useful information about the environment you are running in:\n"
    "<env>\n"
    "Working directory: /Users/demo/project\n"
    "Workspace root folder: /Users/demo/project\n"
    "Is directory a git repo: yes\n"
    "Platform: darwin\n"
    "</env>"
)


def _opencode_body() -> dict[str, Any]:
    return {
        "model": "claude-opus",
        "messages": [
            {"role": "system", "content": OPENCODE_SYSTEM},
            {"role": "user", "content": "Сделай задачу"},
        ],
        "max_tokens": 512,
    }


def test_move_env_to_user_moves_env_section() -> None:
    request = anthropic_bridge.openai_to_anthropic(
        _opencode_body(), "claude-opus-4-6", move_env_to_user=True
    )
    assert request["system"] == [
        {
            "type": "text",
            "text": (
                "You are an AI agent running in OpenCode, a coding agent "
                "harness.\n\nInstructions from: "
                "/Users/demo/project/AGENTS.md\n"
                "# Борт — правила работы с интерфейсом\n"
                "Не меняйте стек."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    first = request["messages"][0]
    assert first["role"] == "user"
    assert first["content"][0] == {"type": "text", "text": OPENCODE_ENV_SECTION}
    assert first["content"][1] == {"type": "text", "text": "Сделай задачу"}


def test_move_env_to_user_is_off_by_default() -> None:
    request = anthropic_bridge.openai_to_anthropic(_opencode_body(), "claude-opus-4-6")
    assert "<env>" in request["system"][0]["text"]
    assert request["messages"][0]["content"] == [
        {"type": "text", "text": "Сделай задачу"}
    ]


def test_move_env_to_user_drops_empty_system() -> None:
    body = {
        "messages": [
            {"role": "system", "content": OPENCODE_ENV_SECTION},
            {"role": "user", "content": "hi"},
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(
        body, "m", move_env_to_user=True
    )
    assert "system" not in request
    assert request["messages"][0]["content"] == [
        {"type": "text", "text": OPENCODE_ENV_SECTION},
        {"type": "text", "text": "hi"},
    ]


def test_move_env_to_user_block_without_intro() -> None:
    """Блок без интро/даты всё равно переносится — секция едина."""
    body = {
        "messages": [
            {"role": "system", "content": "Rules.\n\n<env>\nWorking directory: /x\n</env>"},
            {"role": "user", "content": "hi"},
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(
        body, "m", move_env_to_user=True
    )
    assert request["system"][0]["text"] == "Rules."
    assert request["messages"][0]["content"][0]["text"] == (
        "<env>\nWorking directory: /x\n</env>"
    )


def test_move_env_to_user_unclosed_block_left_in_place() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "Rules.\n<env>\nno closing tag"},
            {"role": "user", "content": "hi"},
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(
        body, "m", move_env_to_user=True
    )
    assert request["system"][0]["text"] == "Rules.\n<env>\nno closing tag"


def test_move_env_to_user_after_leading_tool_results() -> None:
    """tool_result-блоки остаются первыми; текст идёт сразу за ними."""
    body = {
        "messages": [
            {"role": "system", "content": OPENCODE_ENV_SECTION},
            {"role": "tool", "tool_call_id": "c1", "content": "res"},
            {"role": "user", "content": "next"},
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(
        body, "m", move_env_to_user=True
    )
    first = request["messages"][0]
    assert [b["type"] for b in first["content"]] == ["tool_result", "text", "text"]
    assert first["content"][1]["text"] == OPENCODE_ENV_SECTION



def test_max_tokens_defaults_and_aliases() -> None:
    request = anthropic_bridge.openai_to_anthropic(
        {"messages": [{"role": "user", "content": "hi"}]}, "claude-opus-4-6"
    )
    assert request["max_tokens"] == anthropic_bridge.DEFAULT_MAX_TOKENS
    request = anthropic_bridge.openai_to_anthropic(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 1000,
        },
        "claude-opus-4-6",
    )
    assert request["max_tokens"] == 1000


def test_tools_and_tool_choice_translation() -> None:
    body = {
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "required",
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    assert request["tools"] == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert request["tool_choice"] == {"type": "any"}

    request = anthropic_bridge.openai_to_anthropic(
        {**body, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
        "m",
    )
    assert request["tool_choice"] == {"type": "tool", "name": "get_weather"}

    request = anthropic_bridge.openai_to_anthropic(
        {**body, "tool_choice": "auto", "parallel_tool_calls": False}, "m"
    )
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


# --------------------------------------------------------------------------- #
# Санитайзер input_schema: upstream «anthropic» 400-ит на anyOf/oneOf/allOf и
# требует единственный строковый "type" в каждом узле схемы
# --------------------------------------------------------------------------- #
_COMBINATOR_KEYS = ("anyOf", "oneOf", "allOf")


def _combinator_paths(node: Any, path: str = "") -> list[str]:
    """Пути всех anyOf/oneOf/allOf в дереве (рекурсивно)."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _COMBINATOR_KEYS:
                found.append(f"{path}/{key}")
            found.extend(_combinator_paths(value, f"{path}/{key}"))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_combinator_paths(item, f"{path}[{index}]"))
    return found


def _input_schema(parameters: dict[str, Any], name: str = "tool") -> dict[str, Any]:
    """Прогнать одну OpenAI-function-схему через _convert_tools."""
    tools = [
        {
            "type": "function",
            "function": {"name": name, "description": "", "parameters": parameters},
        }
    ]
    converted = anthropic_bridge._convert_tools(tools)
    return converted[0]["input_schema"]


def _schema_nodes(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Узлы-схемы: корень, значения properties и items (рекурсивно)."""
    nodes = [schema]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for value in properties.values():
            if isinstance(value, dict):
                nodes.extend(_schema_nodes(value))
    items = schema.get("items")
    if isinstance(items, dict):
        nodes.extend(_schema_nodes(items))
    return nodes


def _assert_every_node_typed(schema: dict[str, Any]) -> None:
    for node in _schema_nodes(schema):
        assert isinstance(node.get("type"), str) and node["type"], node


def test_tool_schema_without_combinators_is_unchanged() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City", "default": "Tokyo"},
            "days": {"type": "integer", "minimum": 1, "maximum": 14},
            "nested": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"zip": {"type": "string", "pattern": "^[0-9]+$"}},
                    "additionalProperties": True,
                },
            },
        },
        "required": ["city"],
    }
    snapshot = json.loads(json.dumps(parameters))
    schema = _input_schema(parameters)
    assert schema == snapshot
    # Чистая схема проходит без копирования — тот же объект.
    assert schema is parameters
    assert parameters == snapshot


def test_tool_schema_strips_anyof_at_property_level() -> None:
    """Форма terminal: notify с anyOf из bool и array → типизированный bool."""
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command"},
            "notify": {
                "description": "true: one notification on exit",
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}},
                ],
            },
        },
        "required": ["command"],
    }
    schema = _input_schema(parameters)
    assert schema["properties"]["notify"] == {
        "type": "boolean",
        "description": "true: one notification on exit",
    }
    # Соседние чистые поля не тронуты.
    assert schema["properties"]["command"] == {
        "type": "string",
        "description": "Shell command",
    }
    assert schema["required"] == ["command"]
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_tool_schema_strips_anyof_inside_array_items() -> None:
    """Форма manage_connections: array.items.anyOf из string и object."""
    parameters = {
        "type": "object",
        "properties": {
            "connectors": {
                "type": "array",
                "description": "Targets",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    ]
                },
            }
        },
    }
    schema = _input_schema(parameters)
    connectors = schema["properties"]["connectors"]
    assert connectors["type"] == "array"
    assert connectors["description"] == "Targets"
    # Первый member с валидным типом — строка (union сведён к одному типу).
    assert connectors["items"] == {"type": "string"}
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_tool_schema_strips_oneof_and_allof() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "a": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "b": {
                "allOf": [
                    {"type": "object", "properties": {"x": {"type": "string"}}},
                    {"required": ["x"]},
                ]
            },
            "c": {
                "type": "array",
                "items": {"oneOf": [{"type": "boolean"}]},
            },
        },
    }
    schema = _input_schema(parameters)
    assert schema["properties"]["a"] == {"type": "string"}
    # Полная схема выбранного member'а, включая вложенные properties.
    assert schema["properties"]["b"] == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    assert schema["properties"]["c"]["items"] == {"type": "boolean"}
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_combinator_members_without_string_type_are_skipped() -> None:
    parameters = {
        "type": "object",
        "properties": {
            # Первый member без type — пропускается, второй несёт "+".
            "with_type": {
                "oneOf": [
                    {"description": "no type here"},
                    {"enum": ["a", "b"]},
                    {"type": "integer", "minimum": 1},
                ]
            },
            # Ни один member не даёт строкового type → object-заглушка.
            "without_type": {
                "description": "Union без типов",
                "default": None,
                "anyOf": [{"enum": [1, 2]}, {"pattern": "^x$"}],
            },
        },
    }
    schema = _input_schema(parameters)
    assert schema["properties"]["with_type"] == {"type": "integer", "minimum": 1}
    assert schema["properties"]["without_type"] == {
        "type": "object",
        "description": "Union без типов",
        "default": None,
    }
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_tool_schema_strips_combinator_nested_in_combinator_member() -> None:
    parameters = {
        "type": "object",
        "properties": {
            # Единственный member сам без type → object-заглушка.
            "x": {"anyOf": [{"oneOf": [{"type": "string"}]}]},
            # Member с типом выбран, вложенный комбинатор вычищен рекурсивно.
            "z": {
                "anyOf": [
                    {"oneOf": [{"type": "string"}]},
                    {
                        "type": "object",
                        "properties": {"y": {"allOf": [{"type": "integer"}]}},
                    },
                ]
            },
        },
    }
    schema = _input_schema(parameters)
    assert schema["properties"]["x"] == {"type": "object"}
    assert schema["properties"]["z"] == {
        "type": "object",
        "properties": {"y": {"type": "integer"}},
    }
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_combinator_node_keeps_description_and_default() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": "Mode selector",
                "default": "auto",
                "enum": ["auto", "manual"],
                "anyOf": [
                    {"type": "string", "description": "member desc"},
                    {"type": "null"},
                ],
            }
        },
    }
    schema = _input_schema(parameters)
    # Описание/дефолт исходного узла побеждают поля member'а; type берётся
    # из member'а, а enum исходного узла отброшен (иначе он бы конфликтовал).
    assert schema["properties"]["mode"] == {
        "type": "string",
        "description": "Mode selector",
        "default": "auto",
    }
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_type_array_without_combinator_is_normalized() -> None:
    parameters = {
        "type": ["object", "null"],
        "properties": {
            "a": {"type": ["string", "null"], "description": "A"},
            "b": {"type": [], "description": "B"},
            "c": {"type": [1, "integer"]},
            "nested": {
                "type": "array",
                "items": {"type": ["object", "string"]},
            },
        },
    }
    schema = _input_schema(parameters)
    assert schema["type"] == "object"
    assert schema["properties"]["a"] == {"type": "string", "description": "A"}
    assert schema["properties"]["b"] == {"type": "object", "description": "B"}
    assert schema["properties"]["c"] == {"type": "integer"}
    assert schema["properties"]["nested"]["items"] == {"type": "object"}
    _assert_every_node_typed(schema)


def test_type_array_inside_chosen_combinator_member_is_normalized() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "a": {
                "anyOf": [
                    {"type": ["string", "null"], "minLength": 1},
                    {"type": "object"},
                ]
            }
        },
    }
    schema = _input_schema(parameters)
    # Первый member годится после сведения type-массива к строке.
    assert schema["properties"]["a"] == {"type": "string", "minLength": 1}
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_combinator_replacements_are_never_permissive() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "garbage": {"anyOf": ["x", None, {"type": ""}, {"type": [1, 2]}]},
            "allof_not_list": {"allOf": {"type": "string"}},
            "empty_members": {"oneOf": []},
        },
    }
    schema = _input_schema(parameters)
    for name in ("garbage", "allof_not_list", "empty_members"):
        node = schema["properties"][name]
        # Не {} и не description-only: upstream требует строковый type.
        assert node == {"type": "object"}, name
    assert _combinator_paths(schema) == []
    _assert_every_node_typed(schema)


def test_tool_schema_sanitizing_does_not_mutate_caller_dicts() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "manage_connections",
                "description": "Connect",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "connectors": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {"name": {"type": "string"}},
                                    },
                                ]
                            },
                        }
                    },
                },
            },
        }
    ]
    snapshot = json.loads(json.dumps(tools))
    anthropic_bridge._convert_tools(tools)
    assert tools == snapshot


# Урезанная выжимка из реального payload'а Hermes Agent (112 КБ, 35
# инструментов): три схемы с комбинаторами (tools[18], [25], [34]) и одна
# чистая. Полный файл в тесты не тащим.
HERMES_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "manage_connections",
            "description": "Connect the user to apps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["status", "connect"]},
                    "connectors": {
                        "type": "array",
                        "description": "Targets",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "mcp": {"type": "boolean"},
                                        "alias": {"type": "string"},
                                    },
                                    "required": ["name"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Execute shell commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "background": {"type": "boolean", "default": False},
                    "timeout": {"type": "integer", "minimum": 1},
                    "notify": {
                        "description": "With background=true.",
                        "anyOf": [
                            {"type": "boolean"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_manage",
            "description": "Create, update, or delete skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "description": "Ordered ops",
                        "items": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "action": {"type": "string", "enum": ["create"]},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["name", "action", "content"],
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "action": {"type": "string", "enum": ["delete"]},
                                    },
                                    "required": ["name", "action"],
                                },
                            ]
                        },
                    }
                },
                "required": ["operations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clarify",
            "description": "Ask the user a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question"],
            },
        },
    },
]


def test_hermes_fixture_tools_have_no_combinators() -> None:
    converted = anthropic_bridge._convert_tools(HERMES_TOOLS)
    by_name = {tool["name"]: tool for tool in converted}
    for name in ("manage_connections", "terminal", "skill_manage", "clarify"):
        assert _combinator_paths(by_name[name]["input_schema"]) == [], name
        _assert_every_node_typed(by_name[name]["input_schema"])
    assert by_name["terminal"]["input_schema"]["properties"]["notify"] == {
        "type": "boolean",
        "description": "With background=true.",
    }
    assert by_name["manage_connections"]["input_schema"]["properties"]["connectors"][
        "items"
    ] == {"type": "string"}
    # Первый member operations.items сохраняется целиком (тип object и поля).
    operations_items = by_name["skill_manage"]["input_schema"]["properties"][
        "operations"
    ]["items"]
    assert operations_items["type"] == "object"
    assert set(operations_items["properties"]) == {"name", "action", "content"}
    assert operations_items["required"] == ["name", "action", "content"]
    # Чистый инструмент дошёл без искажений.
    assert by_name["clarify"]["input_schema"] == HERMES_TOOLS[3]["function"]["parameters"]


def test_non_function_tools_keep_passthrough_schema() -> None:
    tools = [
        {"type": "web_search_20250305", "name": "web_search", "input_schema": {"anyOf": []}},
        {"type": "function", "function": {"name": "f", "parameters": {"anyOf": [{"type": "string"}]}}},
    ]
    snapshot = json.loads(json.dumps(tools))
    converted = anthropic_bridge._convert_tools(tools)
    assert converted[0] == snapshot[0]
    assert converted[1]["input_schema"] == {"type": "string"}
    assert tools == snapshot


# --------------------------------------------------------------------------- #
# alias_tool_names: переименование function-tools и de-alias ответа
# --------------------------------------------------------------------------- #
def _function_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        },
    }


def test_alias_tool_names_off_keeps_request_identical() -> None:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_function_tool("get_weather"), _function_tool("terminal")],
        "tool_choice": {"type": "function", "function": {"name": "terminal"}},
    }
    base = anthropic_bridge.openai_to_anthropic(body, "m")
    explicit = anthropic_bridge.openai_to_anthropic(body, "m", alias_tool_names=False)
    assert explicit == base
    assert [tool["name"] for tool in base["tools"]] == ["get_weather", "terminal"]
    assert base["tool_choice"] == {"type": "tool", "name": "terminal"}


def test_alias_tool_names_deterministic_zero_padded_order() -> None:
    tools = [_function_tool(f"tool_{index}") for index in range(12)]
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": tools}
    request, mapping = anthropic_bridge.openai_to_anthropic_with_tool_aliases(
        body, "m"
    )
    names = [tool["name"] for tool in request["tools"]]
    assert names[0] == "fn00_tool_0"
    assert names[9] == "fn09_tool_9"
    assert names[10] == "fn10_tool_10"
    # Двузначный префикс сохраняет лексикографический порядок.
    assert names == sorted(names)
    assert mapping == {f"fn{index:02d}_tool_{index}": f"tool_{index}" for index in range(12)}

    again, again_map = anthropic_bridge.openai_to_anthropic_with_tool_aliases(
        body, "m"
    )
    assert again["tools"] == request["tools"]
    assert again_map == mapping
    # Флаг на одиночной обёртке тоже переименовывает запрос.
    plain = anthropic_bridge.openai_to_anthropic(body, "m", alias_tool_names=True)
    assert [tool["name"] for tool in plain["tools"]] == names


def test_alias_tool_name_truncated_to_64_and_unique() -> None:
    long_a = "a" * 80
    long_b = "a" * 80 + "b"
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_function_tool(long_a), _function_tool(long_b)],
    }
    request, mapping = anthropic_bridge.openai_to_anthropic_with_tool_aliases(
        body, "m"
    )
    first, second = (tool["name"] for tool in request["tools"])
    assert len(first) == 64
    assert first == "fn00_" + "a" * 59
    assert len(second) == 64
    # Уникальность держится на индексе, хотя имена усечены одинаково.
    assert first != second
    assert mapping == {first: long_a, second: long_b}


def test_alias_tool_names_applies_to_tool_choice() -> None:
    tools = [_function_tool("get_weather"), _function_tool("terminal")]
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "terminal"}},
    }
    request, _ = anthropic_bridge.openai_to_anthropic_with_tool_aliases(body, "m")
    assert request["tool_choice"] == {"type": "tool", "name": "fn01_terminal"}

    # Anthropic-форма tool_choice тоже переводится на alias.
    anthropic_choice = {
        **body,
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    request, _ = anthropic_bridge.openai_to_anthropic_with_tool_aliases(
        anthropic_choice, "m"
    )
    assert request["tool_choice"] == {"type": "tool", "name": "fn00_get_weather"}


def test_alias_tool_names_skips_non_function_tools() -> None:
    tools = [
        {"type": "web_search_20250305", "name": "web_search", "input_schema": {}},
        _function_tool("terminal"),
    ]
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": tools}
    request, mapping = anthropic_bridge.openai_to_anthropic_with_tool_aliases(
        body, "m"
    )
    assert request["tools"][0] == tools[0]
    assert request["tools"][1]["name"] == "fn01_terminal"
    assert mapping == {"fn01_terminal": "terminal"}


def test_response_tool_names_dealiased_non_stream() -> None:
    payload = {
        "id": "msg_multi",
        "content": [
            {"type": "text", "text": "Working"},
            {"type": "tool_use", "id": "toolu_1", "name": "fn00_get_weather",
             "input": {"city": "Tokyo"}},
            {"type": "tool_use", "id": "toolu_2", "name": "fn01_terminal",
             "input": {"command": "ls"}},
            {"type": "tool_use", "id": "toolu_3", "name": "not_in_map",
             "input": {}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    result = anthropic_bridge.anthropic_to_openai(
        payload,
        "claude-opus",
        tool_name_map={"fn00_get_weather": "get_weather", "fn01_terminal": "terminal"},
    )
    calls = result["choices"][0]["message"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == [
        "get_weather",
        "terminal",
        "not_in_map",
    ]
    assert calls[0]["function"]["arguments"] == '{"city": "Tokyo"}'


def test_stream_tool_names_dealiased_multi_and_unknown() -> None:
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_a"}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "tool_use",
                                                   "id": "toolu_0",
                                                   "name": "fn00_get_weather"}}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "tool_use",
                                                   "id": "toolu_1",
                                                   "name": "fn01_terminal"}}),
        ("content_block_start", {"type": "content_block_start", "index": 2,
                                 "content_block": {"type": "tool_use",
                                                   "id": "toolu_2",
                                                   "name": "unknown_alias"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        ("message_stop", {"type": "message_stop"}),
    )
    translator = anthropic_bridge.AnthropicStreamTranslator(
        "claude-opus",
        tool_name_map={"fn00_get_weather": "get_weather", "fn01_terminal": "terminal"},
    )
    parsed = parse_frames(b"".join(translator.feed(raw) + translator.finish()))
    names = [
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        for chunk in parsed[1:]
        if isinstance(chunk, dict)
        and chunk["choices"][0]["delta"].get("tool_calls")
        and "name" in chunk["choices"][0]["delta"]["tool_calls"][0]["function"]
    ]
    assert names == ["get_weather", "terminal", "unknown_alias"]


def test_alias_and_sanitizer_together() -> None:
    tools = [
        HERMES_TOOLS[0],
        HERMES_TOOLS[1],
        {"type": "web_search_20250305", "name": "web_search", "input_schema": {"anyOf": []}},
        HERMES_TOOLS[3],
    ]
    snapshot = json.loads(json.dumps(tools))
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": tools}
    request, mapping = anthropic_bridge.openai_to_anthropic_with_tool_aliases(
        body, "m"
    )
    names = [tool["name"] for tool in request["tools"]]
    # Индекс — позиция в исходном массиве, pass-through не переименован.
    assert names == [
        "fn00_manage_connections",
        "fn01_terminal",
        "web_search",
        "fn03_clarify",
    ]
    assert mapping == {
        "fn00_manage_connections": "manage_connections",
        "fn01_terminal": "terminal",
        "fn03_clarify": "clarify",
    }
    for tool in request["tools"][:2] + request["tools"][3:]:
        assert _combinator_paths(tool["input_schema"]) == [], tool["name"]
        _assert_every_node_typed(tool["input_schema"])
    # Санитайзер и алиасинг не мутируют исходные tools.
    assert tools == snapshot




def test_tool_roundtrip_messages() -> None:
    """assistant tool_calls → tool_use; tool-результаты → user с tool_result."""
    body = {
        "messages": [
            {"role": "user", "content": "weather in Tokyo?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "21C, rain",
            },
            {"role": "user", "content": "and in Paris?"},
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    assert request["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "weather in Tokyo?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "21C, rain"},
                {"type": "text", "text": "and in Paris?"},
            ],
        },
    ]
    # брейкпоинт — на предпоследнем сообщении (assistant tool_use), не на новом ходе
    assert json.dumps(request).count('"ephemeral"') == 1


def test_unparsable_tool_arguments_are_not_lost() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "f", "arguments": "{oops"}}
            ]},
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    assert request["messages"][1]["content"][0]["input"] == {"_unparsed": "{oops"}


def test_image_parts_translation() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is it?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ]
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    content = request["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is it?"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    assert content[2] == {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}


def test_reasoning_effort_maps_to_thinking_and_drops_sampling() -> None:
    body = {
        "messages": [{"role": "user", "content": "q"}],
        "max_tokens": 8192,
        "temperature": 0.3,
        "reasoning_effort": "high",
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    # budget < max_tokens ⇒ max_tokens поднят, sampling убран
    assert request["max_tokens"] == 17408
    assert "temperature" not in request

    request = anthropic_bridge.openai_to_anthropic(
        {**body, "reasoning_effort": "none"}, "m"
    )
    assert "thinking" not in request
    assert request["temperature"] == 0.3

    # reasoning.effort (OpenRouter-форма) работает без reasoning_effort
    request = anthropic_bridge.openai_to_anthropic(
        {
            "messages": body["messages"],
            "reasoning": {"effort": "low"},
        },
        "m",
    )
    assert request["thinking"]["budget_tokens"] == 1024

    with pytest.raises(anthropic_bridge.TranslationError):
        anthropic_bridge.openai_to_anthropic(
            {**body, "reasoning_effort": "ultra"}, "m"
        )


def test_explicit_anthropic_thinking_passthrough() -> None:
    body = {
        "messages": [{"role": "user", "content": "q"}],
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }
    request = anthropic_bridge.openai_to_anthropic(body, "m")
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert request["max_tokens"] == 8192


# --------------------------------------------------------------------------- #
# Adaptive thinking (Claude 5+): output_config.effort
# --------------------------------------------------------------------------- #
def test_effort_capable_detection_by_provider_model() -> None:
    for model in (
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-opus-5",
        "claude-fable-5-1",
        "claude-haiku-5-5",
        "CLAUDE-OPUS-5-5",
    ):
        assert anthropic_bridge._is_effort_capable(model) is True, model
    for model in (
        "claude-haiku-4-5",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "m",
        "",
        "gpt-5",
    ):
        assert anthropic_bridge._is_effort_capable(model) is False, model


def test_adaptive_thinking_effort_maps_to_output_config() -> None:
    base = {"messages": [{"role": "user", "content": "q"}]}
    for effort, expected in (
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
    ):
        request = anthropic_bridge.openai_to_anthropic(
            {**base, "reasoning_effort": effort}, "claude-opus-5-5"
        )
        assert request["output_config"] == {"effort": expected}, effort
        assert "thinking" not in request

    # reasoning.effort (OpenRouter-форма) тоже работает
    request = anthropic_bridge.openai_to_anthropic(
        {"messages": base["messages"], "reasoning": {"effort": "high"}},
        "claude-opus-5-5",
    )
    assert request["output_config"] == {"effort": "high"}

    with pytest.raises(anthropic_bridge.TranslationError):
        anthropic_bridge.openai_to_anthropic(
            {**base, "reasoning_effort": "ultra"}, "claude-opus-5-5"
        )


def test_adaptive_thinking_disabled_or_absent_sends_nothing() -> None:
    base = {"messages": [{"role": "user", "content": "q"}]}
    for effort in (None, "none", "off", "disabled", "false", "", "default"):
        body = dict(base) if effort is None else {**base, "reasoning_effort": effort}
        request = anthropic_bridge.openai_to_anthropic(body, "claude-opus-5-5")
        assert "output_config" not in request, effort
        assert "thinking" not in request, effort


def test_adaptive_thinking_drops_explicit_thinking_and_honours_output_config() -> None:
    base = {"messages": [{"role": "user", "content": "q"}]}
    for thinking in (
        {"type": "enabled", "budget_tokens": 2048},
        {"type": "disabled"},
    ):
        request = anthropic_bridge.openai_to_anthropic(
            {**base, "thinking": thinking}, "claude-opus-5-5"
        )
        assert "thinking" not in request
        assert "output_config" not in request

    # явный output_config форвардится как есть и важнее reasoning_effort
    request = anthropic_bridge.openai_to_anthropic(
        {
            **base,
            "output_config": {"effort": "max"},
            "reasoning_effort": "ultra",
        },
        "claude-opus-5-5",
    )
    assert request["output_config"] == {"effort": "max"}
    assert "thinking" not in request


def test_adaptive_thinking_sampling_and_max_tokens() -> None:
    base = {
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0.3,
        "top_p": 0.9,
        "reasoning_effort": "high",
    }
    # без max_tokens — дефолт 32768, sampling убран
    request = anthropic_bridge.openai_to_anthropic(dict(base), "claude-opus-5-5")
    assert request["max_tokens"] == anthropic_bridge.EFFORT_DEFAULT_MAX_TOKENS == 32768
    assert "temperature" not in request
    assert "top_p" not in request

    # клиентский max_tokens не трогаем (без budget bump)
    request = anthropic_bridge.openai_to_anthropic(
        {**base, "max_tokens": 1000}, "claude-opus-5-5"
    )
    assert request["max_tokens"] == 1000
    assert request["output_config"] == {"effort": "high"}

    request = anthropic_bridge.openai_to_anthropic(
        {**base, "max_completion_tokens": 2048}, "claude-opus-5-5"
    )
    assert request["max_tokens"] == 2048


def test_haiku_4_5_keeps_legacy_thinking_budget() -> None:
    body = {
        "messages": [{"role": "user", "content": "q"}],
        "reasoning_effort": "high",
    }
    request = anthropic_bridge.openai_to_anthropic(body, "claude-haiku-4-5")
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "output_config" not in request
    assert request["max_tokens"] == 17408

    request = anthropic_bridge.openai_to_anthropic(
        {"messages": [{"role": "user", "content": "q"}]}, "claude-haiku-4-5"
    )
    assert request["max_tokens"] == anthropic_bridge.DEFAULT_MAX_TOKENS


def test_unsupported_things_raise_translation_error() -> None:
    base = {"messages": [{"role": "user", "content": "q"}]}
    with pytest.raises(anthropic_bridge.TranslationError):
        anthropic_bridge.openai_to_anthropic(
            {**base, "messages": [{"role": "wizard", "content": "q"}]}, "m"
        )
    with pytest.raises(anthropic_bridge.TranslationError):
        anthropic_bridge.openai_to_anthropic(
            {**base, "messages": [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}]},
            "m",
        )
    with pytest.raises(anthropic_bridge.TranslationError):
        anthropic_bridge.openai_to_anthropic(
            {**base, "tool_choice": "auto"}, "m"
        )


def test_conversation_must_start_with_user() -> None:
    with pytest.raises(anthropic_bridge.TranslationError, match="user message"):
        anthropic_bridge.openai_to_anthropic(
            {"messages": [{"role": "assistant", "content": "hi"}]}, "m"
        )


# --------------------------------------------------------------------------- #
# anthropic_to_openai
# --------------------------------------------------------------------------- #
def test_response_translation_text_tools_thinking_usage() -> None:
    payload = anthropic_message(
        text="Calling tool",
        thinking="let me think",
        stop_reason="tool_use",
        tool_use={"id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo"}},
        usage={
            "input_tokens": 100,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25,
            "output_tokens": 9,
        },
    )
    result = anthropic_bridge.anthropic_to_openai(payload, "claude-opus", created=123)
    assert result["id"] == "chatcmpl-msg_test"
    assert result["object"] == "chat.completion"
    assert result["created"] == 123
    assert result["model"] == "claude-opus"
    message = result["choices"][0]["message"]
    assert message["content"] == "Calling tool"
    assert message["reasoning_content"] == "let me think"
    assert message["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
        }
    ]
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result["usage"] == {
        "prompt_tokens": 175,
        "completion_tokens": 9,
        "total_tokens": 184,
        "prompt_tokens_details": {"cached_tokens": 50},
        "cache_read_input_tokens": 50,
        "cache_creation_input_tokens": 25,
    }


def test_response_text_only_and_stop_reasons() -> None:
    result = anthropic_bridge.anthropic_to_openai(
        anthropic_message(stop_reason="max_tokens"), "alias"
    )
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}
    assert result["choices"][0]["message"]["content"] == "Hi!"
    assert "tool_calls" not in result["choices"][0]["message"]
    assert "reasoning_content" not in result["choices"][0]["message"]


def test_error_normalization() -> None:
    assert anthropic_bridge.anthropic_error_to_openai(
        {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    ) == {"error": {"message": "slow down", "type": "rate_limit_error", "code": None}}
    assert anthropic_bridge.anthropic_error_to_openai(None) == {
        "error": {"message": "upstream error", "type": "api_error"}
    }


# --------------------------------------------------------------------------- #
# Стриминг
# --------------------------------------------------------------------------- #
def sse(*events: tuple[str, dict[str, Any]]) -> bytes:
    parts = []
    for name, data in events:
        parts.append(f"event: {name}")
        parts.append(f"data: {json.dumps(data)}")
        parts.append("")
        parts.append("")
    return "\n".join(parts).encode("utf-8")


def text_stream() -> bytes:
    return sse(
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_s", "role": "assistant", "model": "m",
            "usage": {"input_tokens": 4, "output_tokens": 1},
        }}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "Hel"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "lo"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    )


def parse_frames(payload: bytes) -> list[Any]:
    frames = []
    for raw in payload.split(b"\n\n"):
        if not raw.strip():
            continue
        assert raw.startswith(b"data: ")
        line = raw[len(b"data: "):].decode("utf-8")
        frames.append(line if line == "[DONE]" else json.loads(line))
    return frames


def test_stream_translation_text() -> None:
    translator = anthropic_bridge.AnthropicStreamTranslator("claude-opus", include_usage=True)
    frames: list[bytes] = []
    # произвольные границы байтовых чанков
    raw = text_stream()
    for i in range(0, len(raw), 7):
        frames.extend(translator.feed(raw[i : i + 7]))
    frames.extend(translator.finish())

    parsed = parse_frames(b"".join(frames))
    assert parsed[-1] == "[DONE]"
    chunks = parsed[:-1]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[0]["id"] == "chatcmpl-msg_s"
    assert chunks[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert chunks[2]["choices"][0]["delta"] == {"content": "lo"}
    final = chunks[3]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["choices"][0]["delta"] == {}
    usage_chunk = chunks[4]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {
        "prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9
    }
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_stream_translation_tool_use() -> None:
    raw = sse(
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_t", "usage": {"input_tokens": 4, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "tool_use", "id": "toolu_9",
                                                   "name": "get_weather"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "input_json_delta", "partial_json": '{"ci'}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "input_json_delta", "partial_json": 'ty": "Tokyo"}'}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        ("message_stop", {"type": "message_stop"}),
    )
    translator = anthropic_bridge.AnthropicStreamTranslator("claude-opus")
    frames = translator.feed(raw) + translator.finish()

    parsed = parse_frames(b"".join(frames))
    assert parsed[-1] == "[DONE]"
    chunks = parsed[:-1]
    start = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert start == {
        "index": 0, "id": "toolu_9", "type": "function",
        "function": {"name": "get_weather", "arguments": ""},
    }
    tool_chunk2 = chunks[2]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_chunk2 == {"index": 0, "function": {"arguments": '{"ci'}}
    tool_chunk3 = chunks[3]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_chunk3["function"]["arguments"] == 'ty": "Tokyo"}'
    assert chunks[4]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_thinking_delta_becomes_reasoning_content() -> None:
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_r"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
        ("message_stop", {"type": "message_stop"}),
    )
    translator = anthropic_bridge.AnthropicStreamTranslator("alias")
    parsed = parse_frames(b"".join(translator.feed(raw) + translator.finish()))
    assert parsed[1]["choices"][0]["delta"] == {"reasoning_content": "hmm"}


def test_stream_without_message_stop_finishes_gracefully() -> None:
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_x"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "partial"}}),
    )
    translator = anthropic_bridge.AnthropicStreamTranslator("alias")
    parsed = parse_frames(b"".join(translator.feed(raw) + translator.finish()))
    assert parsed[-1] == "[DONE]"
    assert parsed[-2]["choices"][0]["finish_reason"] == "stop"


def test_stream_error_event_becomes_openai_error_frame() -> None:
    raw = sse(
        ("message_start", {"type": "message_start", "message": {"id": "msg_e"}}),
        ("error", {"type": "error", "error": {"type": "overloaded_error",
                                              "message": "Overloaded"}}),
    )
    translator = anthropic_bridge.AnthropicStreamTranslator("alias")
    parsed = parse_frames(b"".join(translator.feed(raw)))
    assert parsed[-1] == "[DONE]"
    assert parsed[-2] == {"error": {"message": "Overloaded", "type": "overloaded_error"}}
    # после ошибки поток считается завершённым
    assert translator.feed(b"event: ping\ndata: {}\n\n") == []


def test_stream_ping_ignored() -> None:
    raw = sse(("ping", {"type": "ping"}))
    translator = anthropic_bridge.AnthropicStreamTranslator("alias")
    assert translator.feed(raw) == []


# --------------------------------------------------------------------------- #
# Интеграция с шлюзом (MockTransport)
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _fake_kraube_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRAUBE_SERVE_KEY", KRAUBE_KEY)


def make_client(handler, config: dict | None = None) -> TestClient:
    app = gateway.create_app(config or ANTHROPIC_CONFIG, transport=httpx.MockTransport(handler))
    return TestClient(app)


def test_gateway_routes_to_anthropic_upstream() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=anthropic_message())

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "claude-opus",
                "messages": [
                    {"role": "system", "content": "Be nice."},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 256,
            },
        )

    assert response.status_code == 200
    upstream = captured["request"]
    assert str(upstream.url) == "http://127.0.0.1:8787/v1/messages"
    assert upstream.headers["authorization"] == f"Bearer {KRAUBE_KEY}"
    sent = json.loads(upstream.content)
    assert sent["model"] == "claude-opus-4-6"
    assert sent["max_tokens"] == 256
    assert sent["system"] == [
        {
            "type": "text",
            "text": "Be nice.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]

    body = response.json()
    assert body["model"] == "claude-opus"
    assert body["choices"][0]["message"]["content"] == "Hi!"
    assert body["usage"]["total_tokens"] == 17


def test_gateway_moves_env_section_when_configured() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=anthropic_message())

    config = json.loads(json.dumps(ANTHROPIC_CONFIG))
    config["upstreams"]["kraube"]["move_env_to_user"] = True

    with make_client(handler, config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "claude-opus",
                "messages": [
                    {"role": "system", "content": OPENCODE_SYSTEM},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 64,
            },
        )

    assert response.status_code == 200
    sent = json.loads(captured["request"].content)
    assert "<env>" not in sent["system"][0]["text"]
    assert sent["messages"][0]["content"][0] == {
        "type": "text",
        "text": OPENCODE_ENV_SECTION,
    }


def test_gateway_env_flag_requires_real_bool() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=anthropic_message())

    config = json.loads(json.dumps(ANTHROPIC_CONFIG))
    config["upstreams"]["kraube"]["move_env_to_user"] = "yes"

    with make_client(handler, config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "claude-opus",
                "messages": [
                    {"role": "system", "content": OPENCODE_SYSTEM},
                    {"role": "user", "content": "hi"},
                ],
            },
        )

    assert response.status_code == 200
    sent = json.loads(captured["request"].content)
    assert "<env>" in sent["system"][0]["text"]


GATEWAY_ALIAS_BODY: dict[str, Any] = {
    "model": "claude-opus",
    "messages": [{"role": "user", "content": "weather?"}],
    "tools": [_function_tool("get_weather"), _function_tool("terminal")],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
}


def test_gateway_alias_tool_names_round_trip() -> None:
    """Флаг upstream'а переименовывает tools и de-alias'ит ответ."""
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json=anthropic_message(
                text="",
                stop_reason="tool_use",
                tool_use={
                    "id": "toolu_1",
                    "name": "fn00_get_weather",
                    "input": {"city": "Tokyo"},
                },
            ),
        )

    config = json.loads(json.dumps(ANTHROPIC_CONFIG))
    config["upstreams"]["kraube"]["alias_tool_names"] = True

    with make_client(handler, config) as client:
        response = client.post(
            "/v1/chat/completions", headers=AUTH, json=GATEWAY_ALIAS_BODY
        )

    assert response.status_code == 200
    sent = json.loads(captured["request"].content)
    assert [tool["name"] for tool in sent["tools"]] == [
        "fn00_get_weather",
        "fn01_terminal",
    ]
    assert sent["tool_choice"] == {"type": "tool", "name": "fn00_get_weather"}
    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "get_weather"


def test_gateway_alias_flag_requires_real_bool() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=anthropic_message())

    config = json.loads(json.dumps(ANTHROPIC_CONFIG))
    config["upstreams"]["kraube"]["alias_tool_names"] = "yes"

    with make_client(handler, config) as client:
        response = client.post(
            "/v1/chat/completions", headers=AUTH, json=GATEWAY_ALIAS_BODY
        )

    assert response.status_code == 200
    sent = json.loads(captured["request"].content)
    assert [tool["name"] for tool in sent["tools"]] == ["get_weather", "terminal"]
    assert sent["tool_choice"] == {"type": "tool", "name": "get_weather"}


def test_gateway_alias_tool_names_stream_round_trip() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        raw = sse(
            ("message_start", {"type": "message_start", "message": {"id": "msg_s"}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "tool_use",
                                                       "id": "toolu_1",
                                                       "name": "fn00_get_weather"}}),
            ("message_stop", {"type": "message_stop"}),
        )
        return httpx.Response(
            200, content=raw, headers={"content-type": "text/event-stream"}
        )

    config = json.loads(json.dumps(ANTHROPIC_CONFIG))
    config["upstreams"]["kraube"]["alias_tool_names"] = True

    with make_client(handler, config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={**GATEWAY_ALIAS_BODY, "stream": True},
        )

    assert response.status_code == 200
    sent = json.loads(captured["request"].content)
    assert sent["tools"][0]["name"] == "fn00_get_weather"
    parsed = parse_frames(response.content)
    starts = [
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        for chunk in parsed
        if isinstance(chunk, dict)
        and chunk.get("choices")
        and chunk["choices"][0]["delta"].get("tool_calls")
        and "name" in chunk["choices"][0]["delta"]["tool_calls"][0]["function"]
    ]
    assert starts == ["get_weather"]


def test_gateway_passthrough_anthropic_model() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=anthropic_message())

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "kraube/claude-haiku-4-5",
                  "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert json.loads(captured["request"].content)["model"] == "claude-haiku-4-5"


def test_gateway_streaming_translation() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=text_stream(), headers={"content-type": "text/event-stream"}
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "claude-opus",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = parse_frames(response.content)
    assert parsed[-1] == "[DONE]"
    deltas = [c for c in parsed[:-1] if c.get("choices")]
    texts = [d["choices"][0]["delta"].get("content") for d in deltas]
    assert "Hel" in texts and "lo" in texts
    usage_chunks = [c for c in parsed[:-1] if not c.get("choices")]
    assert usage_chunks and usage_chunks[0]["usage"]["completion_tokens"] == 5


def test_gateway_translation_error_is_400() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "claude-opus",
                "messages": [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}],
            },
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "audio" in error["message"]


def test_gateway_upstream_error_normalized() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error",
                                             "message": "slow down"}},
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 429
    assert response.json() == {
        "error": {"message": "slow down", "type": "rate_limit_error", "code": None}
    }


def test_gateway_missing_kraube_key_is_502(monkeypatch) -> None:
    monkeypatch.delenv("KRAUBE_SERVE_KEY", raising=False)

    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 502
    assert "KRAUBE_SERVE_KEY" in response.json()["error"]["message"]


def test_gateway_invalid_api_format_rejected() -> None:
    config = json.loads(json.dumps(ANTHROPIC_CONFIG))
    config["upstreams"]["kraube"]["api_format"] = "graphql"

    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    with make_client(handler, config=config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 500
    assert "api_format" in response.json()["error"]["message"]


def test_gateway_reasoning_effort_reaches_anthropic_as_thinking() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=anthropic_message())

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "claude-opus",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
            },
        )
    assert response.status_code == 200
    sent = json.loads(captured["request"].content)
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert sent["max_tokens"] > 1024


def test_gateway_effort_model_sends_output_config_not_thinking() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=anthropic_message())

    config = json.loads(json.dumps(ANTHROPIC_CONFIG))
    config["models"]["claude-opus-5"] = {
        "upstream": "kraube",
        "model": "claude-opus-5-5",
        "name": "Claude Opus 5.5 (subscription)",
    }

    with make_client(handler, config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
    assert response.status_code == 200
    sent = json.loads(captured["request"].content)
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": "high"}
    assert "thinking" not in sent
    assert sent["max_tokens"] == anthropic_bridge.EFFORT_DEFAULT_MAX_TOKENS


def test_catalog_allowlist_skips_anthropic_upstreams() -> None:
    import metadata_refresh

    config = {
        "upstreams": {
            "deepseek": {"base_url": "https://api.deepseek.com/v1"},
            "kraube": {"base_url": "http://127.0.0.1:8787", "api_format": "anthropic"},
        }
    }
    assert metadata_refresh.source_allowlist(config) == ["deepseek"]


# --------------------------------------------------------------------------- #
# Ретрай спорадического «extra usage» 400
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _fast_flake_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Реальная лестница ретраев — десятки секунд; тестам нужны миллисекунды."""
    monkeypatch.setattr(gateway, "_ANTHROPIC_FLAKE_BACKOFF", (0.01, 0.01))


def _extra_usage_error() -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "Third-party apps now draw from your extra usage, "
                    "not your plan limits. Add more at claude.ai/settings/usage "
                    "and keep going."
                ),
            },
        },
    )


def test_extra_usage_flake_is_retried_and_hidden() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _extra_usage_error()
        return httpx.Response(200, json=anthropic_message())

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hi!"
    assert calls["n"] == 2  # один флейк + один успешный повтор


def test_extra_usage_flake_stream_is_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _extra_usage_error()
        return httpx.Response(
            200, content=text_stream(), headers={"content-type": "text/event-stream"}
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert b"[DONE]" in response.content
    assert calls["n"] == 2


def test_extra_usage_flake_surfaces_after_retries() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _extra_usage_error()

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert "extra usage" in error["message"]
    assert calls["n"] == 3  # исходный + 2 коротких ретрая


def test_other_400_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error",
                                                  "message": "max_tokens is required"}}
        )

    with make_client(handler) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={"model": "claude-opus", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    assert "max_tokens" in response.json()["error"]["message"]
    assert calls["n"] == 1
