# План доработки виртуального оператора

Дата: 2026-04-28

## Цель

Повысить стабильность и производительность голосового пайплайна, обеспечить достижение SLO по задержке (<= 1.5 c), и подготовить систему к production-нагрузке.

## Текущее состояние (кратко)

- Архитектура в целом зрелая: Strategy-провайдеры, Voice Engine, семантический роутер, CaseManager, RAG с гибридным ранжированием, runtime-настройки из БД.
- Основной риск по latency: TTS-провайдеры буферизуют весь аудиофайл перед отправкой в WS.
- Есть точки роста по отказоустойчивости, кэшам и мультиворкерной консистентности настроек.

## Приоритизация

### P0 - Критично (влияет на SLO напрямую)

1. Настоящий streaming TTS (OpenAI/ElevenLabs)

- Файлы:
  - `apps/backend/services/openai_tts.py`
  - `apps/backend/services/elevenlabs_tts.py`
- Проблема:
  - Сейчас чанки копятся в `bytearray`, а в пайплайн отдается уже собранный `bytes`.
  - Потеря "first-byte latency", задержка старта озвучки.
- Доработка:
  - Отдавать аудио чанки сразу внутри `async for ... iter_bytes()/aiter_bytes()`.
  - Сохранить текущую сегментацию текста, но убрать финальную буферизацию.
- Ожидаемый эффект:
  - Снижение `first_audio_sent_ms` примерно на 150-400 мс на сегмент.

1. Улучшение VAD-гейта (trailing hangover)

- Файл:
  - `apps/backend/core/vad_audio_gate.py`
- Проблема:
  - Поведение на границе фраз может быть неустойчивым; нужен управляемый "хвост" речи.
- Доработка:
  - Ввести hangover 200-300 мс после последних speech-фреймов.
  - Явно сбрасывать состояние сегмента после hangover.
- Ожидаемый эффект:
  - Более стабильная детекция конца фразы, меньше обрывов/ложных пауз.

## P1 - Высокий приоритет (эффективность и стабильность hot-path)

1. Кэш query-эмбеддингов для RAG

- Файл:
  - `apps/backend/rag/service.py`
- Доработка:
  - LRU/TTL кэш для embedding запроса.
- Эффект:
  - Меньше внешних вызовов, ниже latency и стоимость.

1. Убрать `ensure_collection()` из hot-path поиска

- Файл:
  - `apps/backend/rag/service.py`
- Доработка:
  - Локальный флаг `_ensured`, инициализация коллекции только на startup.

1. Budget watchdog для tool-loop в оркестраторе

- Файл:
  - `apps/backend/core/orchestrator.py`
- Доработка:
  - `asyncio.wait_for` с остатком бюджета на итерации tool-loop.
- Эффект:
  - Контроль p95/p99 и предсказуемый fallback.

1. Кэш динамических фраз CaseManager

- Файл:
  - `apps/backend/core/case_manager.py`
- Доработка:
  - TTL-кэш по ключам `(intent, slot, locale)` для динамических шаблонов.

1. Консистентность runtime-настроек между воркерами

- Файл:
  - `apps/backend/core/bot_runtime.py`
- Доработка:
  - Invalidation через Postgres `LISTEN/NOTIFY` или периодический refresh.
- Эффект:
  - Все воркеры видят одинаковые пороги VAD/router без рестарта.

1. Оптимизация записи turn-ов в Postgres

- Файл:
  - `apps/backend/core/conversation_recorder.py`
- Доработка:
  - Уйти от `SELECT max(seq)` на каждый turn, использовать локальный счетчик/батчинг.

## P2 - Production hardening

1. Экспорт метрик в Prometheus

- Файлы:
  - `apps/backend/utils/slo_metrics.py`
  - API/инициализация приложения
- Доработка:
  - Histograms/counters + `/metrics`.

1. Circuit breaker + retry policy для внешних провайдеров

- Файлы:
  - `apps/backend/services/*` (OpenAI/Deepgram/ElevenLabs/local_http)
- Доработка:
  - Ограниченные retry, backoff, break-open при всплесках ошибок.

1. Улучшить cold start

- Файлы:
  - `docker/backend.Dockerfile`
  - `docker/docker-compose.yml`
- Доработка:
  - Pre-bake моделей/артефактов или надежный prewarm и `healthcheck.start_period`.

1. Rate limiting на WS-эндпоинты

- Файлы:
  - `apps/backend/api/voice_ws.py`
  - `apps/backend/api/agent_assist_ws.py`
- Доработка:
  - Пер-IP/per-token лимиты параллельных сессий.

## P3 - Улучшения качества сопровождения

1. Детеминизм smalltalk в тестах

- Файл:
  - `apps/backend/core/orchestrator.py`

1. Устранить дублирование порогов/правил роутера

- Файлы:
  - `apps/backend/core/orchestrator.py`
  - `apps/backend/core/semantic_local.py`

1. Вынести дополнительные fallback-политики в runtime-настройки

- Файлы:
  - `apps/backend/core/voice_engine.py`
  - `apps/backend/core/prompts.py`
  - `apps/backend/core/bot_runtime.py`

## План внедрения (итерации)

Итерация 1 (SLO-first):

- P0.1 streaming TTS
- P0.2 VAD hangover
- Замеры: `first_audio_sent_ms`, `turn_total_ms` до/после

Итерация 2 (hot-path):

- P1.3, P1.4, P1.5
- Регресс-тесты маршрутизации и voice interrupt

Итерация 3 (runtime + data-path):

- P1.6, P1.7, P1.8

Итерация 4 (production hardening):

- P2.9, P2.10, P2.11, P2.12

## Критерии приемки

- p95 `first_audio_sent_ms` и `turn_total_ms` укладываются в целевые пороги.
- Нет деградации сценариев `interrupt`, `escalation`, `fallback`.
- Настройки бота применяются консистентно во всех воркерах.
- При отказах внешних API система деградирует предсказуемо, без зависаний.

