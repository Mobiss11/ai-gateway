# Архитектура метаданных AI Gateway

Документ описывает цепочку «upstream → каталог → `/v1/models` → OpenCode»
и принятые ограничения. Он дополняет `README.md`, не заменяя его.

## 1. Поток данных

```
DeepSeek /v1/models ─┐
OpenRouter /api/v1/models ─┤   metadata_refresh (явно, metadata-only)
OpenCode native catalog ───┘            │
                                       ▼
                              catalog.json  (source + timestamp)
                                   │            │
                    gateway /v1/models│            │catalog_cli sync
                                   ▼            ▼
                            OpenCode/клиент   providers.homelab.models
```

Каталог — единственный источник обогащённых метаданных. Шлюз **читает** его;
сеть на каждый чат не ходит. Обновление каталога — только явная CLI-команда.

`/v1/models` перечитывает каталог при смене `mtime`/размера файла (bounded,
без сети) и отдаёт по каждой записи
`metadata.source/source_type/fetched_at/processed_at/quality/stale`.
Метаданные применяются только при совпадении маршрута alias; при расхождении
возвращается `metadata.route_mismatch: true` без чужих лимитов.

## 2. Схема записи каталога

Поля alias-записи (`catalog.json → models.<alias>`):

| Поле | Смысл |
| --- | --- |
| `upstream`, `upstream_model` | куда и какую модель роутит шлюз |
| `limit.context`, `limit.output` | проверенный контекст/максимум вывода |
| `limit.context_declared` | заявленный контекст, если обслуживаемый неизвестен (router) |
| `capabilities` | `{tools, input[], output[]}` |
| `reasoning` | `{supported_efforts[], default_effort, mandatory, enabled_by_default}` |
| `cost` | массив per-million `{input, output, cache{read,write}, tier?}` |
| `compatibility` | `{reasoningField}` для native-пакетов (DeepSeek) |
| `package` | нативный пакет OpenCode |
| `variants` | форма variants, совпадающая с генератором пакета |
| `metadata_quality` | `exact` / `partial` / `unknown` (exact требует limit И capabilities) |
| `recommended_disabled` | маршрутизатор без фиксированных границ/цен |
| `source`, `fetched_at` | откуда и когда получены метаданные (для фикстур — исходный timestamp) |
| `source_type`, `processed_at` | `live`/`fixture` и время обработки каталогом |
| `stale`, `retained_fields`, `stale_fields_from` | часть полей удержана из LKG |

Инварианты валидации:

* `positive_int` принимает только целые `> 0`; `bool`/строки/NaN/inf → `None`;
* цены `per_million`/`nonneg_number`: отрицательные и нечисловые → `None`
  (Auto Router `-1` — это unknown, **не** `0` и не «2M бесплатно»);
* native-цены OpenCode уже за 1M токенов, upstream-цены OpenRouter — за токен;
* отсутствующие поля не подменяются «точными» дефолтами: `None` остаётся
  `None`, `metadata_quality` понижается;
* **живые метаданные важнее native**: supported_efforts/pricing из API
  задают variants/цены; native используется только как fallback при
  неизвестном live-значении;
* `default_effort` отбрасывается, если не входит в `supported_efforts`.

## 3. Обновление (refresh)

* Allowlist источников = настроенные `upstreams` конфига шлюза. Сторонние
  хосты недоступны by design.
* Только `GET {base_url}/models` с ключом upstream, ограниченный
  `timeout` (по умолчанию 10 c). Никаких chat/LLM-запросов.
* Live-режим включается **явно** (`--live`). Иначе используется
  `--from-fixture` (sanitized-фикстуры).
* `--live` сам подгружает env-файл шлюза (`--env-file`, по умолчанию
  `~/.config/ai-gateway/env`) через парсер `gateway.load_env_file`
  (не перезаписывает уже установленные переменные); fixture-режим секреты
  не читает.
* Сбой или пустой/битый ответ upstream → записи этого upstream остаются
  прежними (last-known-good); если ни один источник не обновился, файл
  каталога не перезаписывается вовсе.
* Неполные, но валидные данные не затирают точные: поля сливаются
  по-отдельности (limit.context/output, capabilities.*, reasoning,
  variants, cost, name/package/compatibility). Удержанные поля помечаются
  `stale: true` и `retained_fields`; `metadata_quality` пересчитывается.
* Старые поля НЕ удерживаются, если alias теперь роутится в другой
  upstream/модель (`route mismatch`), а alias, удалённые из конфига,
  вычищаются (`models_pruned`). То же проверяют `/v1/models` и sync.
* Провенанс честный: fixture-обновление пишет `kind/type="fixture"` и
  исходный `fetched_at` из фикстуры, а `processed_at` — отдельно; live-запрос
  пишет `kind="live"` и `fetched_at` момента опроса. Свежие точные данные
  не помечаются `stale`.

## 4. Синхронизация OpenCode

V2-конфиг (подтверждено OpenAPI и нормализатором CLI 2.0.21):

* корень — `providers` (не `provider`), модели — `providers.<id>.models`;
* поля модели: `modelID`, `name`, `package`, `settings`, `headers`, `body`,
  `compatibility`, `capabilities`, `variants[]`, `cost[]`, `limit`, `disabled`;
* модель мержит `provider → model → variant`; **явный массив `variants`
  не удаляет унаследованные** — OpenCode объединяет по `id`, а авто-генерацию
  из пакета выполняет только при отсутствии `variants`.

Поэтому для homelab:

* DeepSeek-модели получают `package=@opencode/ai/providers/deepseek`,
  `compatibility.reasoningField=reasoning_content` и variants
  `none/low/high/max` с `body.thinking` + `settings.reasoningEffort`
  (форма генератора `hj`);
* OpenRouter-модели получают `package=@opencode/ai/providers/openrouter` и
  variants `settings.reasoning.effort` (форма генератора `J6`);
* `id`/`modelID` **не меняются** — шлюз должен получать alias. Если у модели
  уже стоит `modelID`, отличный от alias, синк модели **отказывается**
  (`refused`): иначе OpenCode отправит upstream-id и alias не сроутится;
* управляемые поля: `name`, `package`, `compatibility`, `capabilities`,
  `limit`, `cost`, `variants` (и опционально `disabled: true`);
* поля **сливаются, а не заменяются**: variants — по `id` (пользовательские
  variants и лишние ключи settings/body сохраняются), compatibility/limit/
  capabilities — рекурсивным merge; пользовательские options не теряются;
* маршрут проверяется по gateway-алиасам (`sync --config gateway.json`): если
  каталог и gateway расходятся по upstream/модели, модель пропускается
  (`refused`, `route mismatch`); без `--config` план помечается
  `route_verified: false`;
* unknown/incomplete limit/capabilities дают warning даже когда оба значения
  `None`; цены с NaN/Inf отклоняются; JSON пишется строго (`allow_nan=False`).

### Native-пакеты и credential

Per-model нативные пакеты (`@opencode/ai/providers/deepseek`,
`.../openrouter`) требуют `provider.settings.apiKey`; без него OpenCode
убирает модель из `/api/model` (подтверждено runtime-проверкой на моке).
Поэтому при переключении модели на нативный пакет sync:

* если `provider.settings.apiKey` (или объявленный `provider.env`) уже есть —
  ничего не делает;
* иначе берёт токен **только** из case-insensitive `Authorization: Bearer …`
  в `provider.headers` и кладёт его в `provider.settings.apiKey` (тот же
  токен шлюза, чужой upstream-ключ не используется);
* если credential нет — нативный switch **отказывается** (`refused` + warning),
  модель не отключается молча;
* scope строго ограничен `providers.<homelab>.settings.apiKey`; headers,
  baseURL и чужие поля не трогаются; значение не попадает в summary
  (`<redacted>`), backup — `0600`.

Прочее: `sync --request-output-budget N` — явный opt-in, ограничивающий
verified-поле `limit.output` до `min(catalog_hard, N)`; по умолчанию не задан.
Это не `compute_budgets` и не «включение» advisory-эвристики.

Запись: dry-run по умолчанию; `--apply` создаёт timestamped-backup `0600`
рядом с файлом и выполняет атомарную замену (`os.replace`), целевой файл
`0600`. Повторный запуск идемпотентен и не создаёт новых копий.

## 5. Бюджеты вывода и компакция

`limit.output` — **жёсткий максимум upstream**, а не рекомендованный размер
одного запроса. `catalog.compute_budgets()` — **исключительно advisory**:
шлюз и sync его не применяют по умолчанию и не выставляют output-cap в конфиг
OpenCode. Ориентиры:

* `hard_output` — предел upstream (у DeepSeek фактический дефолт `max_tokens`
  около 256000, но это не «включённый» helper-бюджет);
* `recommended_output` — запас (по умолчанию ≤ 131072);
* `usable_input` — `context − recommended_output`.

Единственный способ реально ограничить вывод — явный
`sync --request-output-budget N` (verified `limit.output`). Не используем
полные 393216 токенов вывода на шаг и не считаем весь заявленный контекст
доступным для ввода.

## 6. Прозрачность и кэш

* Тело запроса проксируется как есть; alias → upstream-имя подменяется только
  в `model`. История, `tools` и `reasoning_content` не переупорядочиваются,
  не усекаются и не обрезаются.
* SSE forwarding байт-в-байт через `aiter_bytes()`, без ручного разбора
  фреймов. Usage в streaming не парсится и не досчитывается.
* В non-stream режиме логируются только числовые поля `usage` (включая
  `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens`), без содержимого.
  Cache write никогда не синтезируется.
* httpx сам распаковывает gzip/br/deflate, поэтому `content-encoding`
  upstream не пробрасывается клиенту (иначе тело было бы помечено как сжатое
  дважды).

## 7. Известные ограничения

* Native runtime E2E выполнен на моке: distinct-провайдер даёт ровно DS
  1048576/393216 `none/low/high/max`, Claude 1M/128k
  `max/xhigh/high/medium/low`, Luna 1.05M/128k `+none`; wire — DS
  `reasoning_effort=max` + `thinking.enabled`, OpenRouter
  `reasoning.effort=high/none`, `stream_options.include_usage=true`; mock usage
  in200/cache800/out15/reasoning5 без двойного счёта. Это MOCK, артефакт
  аудита в репозиторий не входит.
* Не выполнялись: полный 1M-контекст/нагрузка/конкурентность, платные
  Pro-completions и реальные новые full-tool тесты OpenRouter. Живой
  prefix-cache — best effort, без гарантии коэффициента, output-cache и
  прогрева нет; fallback/биллинг провайдеров не менялись.

Отдельно (не снимается):

* Маршрут проверяется лениво: `/v1/models` и `sync --config` сравнивают
  upstream/модель каталога с конфигом; для sync без `--config`
  `route_verified: false`.
* `auto` (OpenRouter Router) не имеет фиксированных границ/цен; 2M/0$ не
  выдаются за гарантию, Auto отключён только в OpenCode.
* `compute_budgets()` — advisory: реальный per-step output-cap OpenCode не
  задаётся (у DeepSeek фактический wire-дефолт `max_tokens` ≈ 256000);
  `limit.output` = hard metadata, а не бюджет шага и не влияние на компакцию.
* Расчётная цена каталога — ориентир, не биллинг; общие бюджеты/экономия
  неизвестны и не заявляются.

## 8. Anthropic-мост (api_format: "anthropic")

Чат-путь шлюза — не только OpenAI-проксирование. Upstream с
`api_format: "anthropic"` (например демон `kraube serve`, Messages API через
OAuth-подписку Claude) получает переведённый запрос:

```
клиент (OpenAI chat/completions)
  → gateway.py: resolve alias/passthrough, auth, api_format
  → anthropic_bridge.openai_to_anthropic()
      system/developer → system; tool_calls → tool_use;
      tool-сообщения → tool_result в user; картинки → image-блоки;
      reasoning_effort/reasoning.effort → thinking.budget_tokens
      (none — выключено; бюджет всегда < max_tokens, max_tokens при
      необходимости поднимается; при thinking sampling не отправляется)
  → POST {base_url}/v1/messages  (Bearer = api_key_env upstream'а)
  → ответ/SSE ← anthropic_bridge: anthropic_to_openai() или
      AnthropicStreamTranslator (SSE → chat.completion.chunk,
      input_json_delta → потоковые tool_calls, thinking → reasoning_content)
  → клиент получает OpenAI-формат; ошибки upstream'а нормализуются в
      {"error": {...}} с исходным статусом
```

Свойства моста:

* **перевод без сети**: `anthropic_bridge.py` — чистые функции над
  dict/bytes; HTTP, таймауты, логи вызовов и usage — общие с OpenAI-путём;
* **usage**: `prompt_tokens = input + cache_read + cache_creation`;
  cache-поля пробрасываются и попадают в usage-лог (числовые поля);
  в стриме — из `message_start` + `message_delta`, с пересчётом total;
* **надёжность стрима**: произвольные границы байтовых чанков, `\n\n`/`\r\n\r\n`,
  битые JSON-кадры пропускаются; пропавший `message_stop` не оставляет
  поток без финального кадра и `[DONE]`; SSE-`error` → OpenAI-кадр ошибки;
* **честность 400**: молча отбрасываются только поля без аналога
  (`n`, `logprobs`, `response_format`, `seed`, penalties, `logit_bias`,
  `user`); непереводимые роли/контент — явная ошибка перевода;
* **env-секция OpenCode** (флаг upstream'а `move_env_to_user`, по умолчанию
  выключен): секция `Today's date: …` + «Here is some useful information
  about the environment you are running in:» + `<env>…</env>` вырезается из
  system и вставляется текст-блоком в начало первого user-сообщения (после
  ведущих `tool_result`-блоков). Мотив: в system Anthropic тарифицирует её
  как third-party app — extra-usage кредиты, без них детерминированный 400
  (05.10.2026: триггер — секция целиком; идентичность агента, дата и поля
  по отдельности проходят; в user-сообщении секция проходит). Содержимое и
  теги сохраняются дословно, незакрытый `<env>` не трогается; выключенный
  флаг — system уходит как есть;
* **каталог не опрашивает anthropic-upstream'ы** (`metadata_refresh.source_allowlist`):
  у Messages API нет `GET /models`; модели такого upstream'а описываются
  только в `models` конфига и в `/v1/models` отдаются без метаданных
  каталога (метаданные для OpenCode при необходимости вносятся руками или
  сид-фикстурой);
* **поля без аналога в стриме** (citations, signature_delta, ping) —
  игнорируются; `redacted_thinking` не переводится.
