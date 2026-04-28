# VoiceAgent — голосовой ИИ-агент для контакт-центра

Прототип комплексной системы автоматизации КЦ согласно `Instruction.md`:

- **Voice Bot** — входящая линия с пайплайном `Audio → VAD → STT → Router/Intent → LLM/Tools → TTS → Audio` через WebSocket.
- **Agent Assist (суфлёр)** — отдельный **пул Qdrant** (`knowledge_agent_assist`) и свой RAG; не смешивается с базой голосового бота.
- **Речевая аналитика** — загрузка записи: транскрипция (Whisper), диаризация через `DiarizationService` (`DIARIZATION_PROVIDER=llm|deepgram|pyannote|mock`): deepgram — batch `POST /v1/listen` с `diarize=true` и `utterances`; pyannote — внешний HTTP-воркер (см. ниже); llm — эвристика по сегментам ASR. Оценка по настраиваемым критериям с весами.
- **Запись диалога бот–клиент** — таблицы `conversations` / `conversation_turns`: роли `customer` / `assistant`, эскалация с **саммари** и пакетом истории (`escalation_packet` по WebSocket + `GET /api/v1/conversations/{id}`).
- **Control Plane** — Next.js админка: плейграунд, база знаний, промпты, голоса, аналитика, **«Поведение бота»** (`/bot`) — VAD-пороги, семантический роутинг, intent policy, окно истории, дефолты TTS.

## Стек


| Слой       | Технология                                                                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Backend    | Python 3.11, FastAPI, asyncio, structlog                                                                                                         |
| Frontend   | Next.js 15, React 19, Tailwind CSS, shadcn-style UI                                                                                              |
| БД         | PostgreSQL 16 (метаданные), Qdrant 1.12 (векторы)                                                                                                |
| STT        | Deepgram (WebSocket), OpenAI Whisper, `local_http`, моки + server-side VAD (Silero/WebRTC)                                                       |
| LLM        | OpenAI `gpt-4o-mini` + `text-embedding-3-small`, `local_http` (Ollama / vLLM / LM Studio), локальный semantic-intent/router (`all-MiniLM-L6-v2`) |
| TTS        | ElevenLabs, OpenAI `tts-1`, `local_http`, моки                                                                                                   |
| Контейнеры | Docker, Docker Compose                                                                                                                           |


Все провайдеры спрятаны за **Strategy-интерфейсами** (`apps/backend/services/interfaces.py`).
Замена API на локальные модели: встроенные варианты `local_http` (см. раздел ниже) или свой класс в `factory.py` — без правок call-flow.

## Структура

```
/project-root
  /apps
    /backend           FastAPI приложение
      /api             REST + WebSocket эндпоинты
      /core            Voice Engine, Orchestrator, intent
      /services        Провайдеры STT/LLM/TTS (Deepgram, OpenAI, ElevenLabs, local_http, моки)
      /rag             Чанкинг, парсинг, Qdrant (два пула: voice / agent_assist)
      /analytics       Речевая аналитика (транскрипт, диаризация-LLM, QA-скоринг)
      /models          SQLAlchemy ORM + Pydantic DTO
      /utils           Логирование (structured)
      /tests           pytest
    /frontend          Next.js (App Router): админка + плейграунд
  /docker              Dockerfile-ы и docker-compose.yml
  /shared              Общие константы (TS)
  Instruction.md       Исходное ТЗ
```

## Быстрый старт

### 1. Через Docker Compose (рекомендуется)

```bash
cp .env.example .env
# заполните DEEPGRAM_API_KEY / OPENAI_API_KEY / ELEVENLABS_API_KEY
# или поставьте *_PROVIDER=mock — всё запустится в полностью мок-режиме

docker compose -f docker/docker-compose.yml up --build
```

После старта:

- Админка: [http://localhost:3000](http://localhost:3000)
- API + Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Qdrant Web UI: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### 2. Локально (без Docker)

```bash
# Backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r apps/backend/requirements.txt
export PYTHONPATH=.
uvicorn apps.backend.main:app --reload

# Frontend
cd apps/frontend
npm install
npm run dev
```

Нужны живые PostgreSQL и Qdrant (см. `docker/docker-compose.yml` — можно поднять только инфра-сервисы: `docker compose up postgres qdrant`).

## Ключевые сценарии

### Текстовый плейграунд

`POST /api/v1/playground/chat`

```json
{ "text": "Какие у вас тарифы по вкладам?", "locale": "ru" }
```

Ответ содержит intent, источники из RAG и латентность каждого этапа.

### Голосовой канал (WebSocket)

```
ws://localhost:8000/api/v1/ws/voice?locale=ru&sample_rate=16000&voice_id=<UUID-профиля-из-таблицы-voices>
```

Параметр `voice_id` (UUID строки `voices.id`) необязателен: если не указан, берётся голос с `is_default=true`, иначе — fallback из `.env` (`ELEVENLABS_VOICE_ID` / `OPENAI_TTS_VOICE`). Для выбранного профиля подставляются `provider_voice_id` и JSON `tts_params` (скорость OpenAI, stability ElevenLabs и т.д.).

VAD перед STT на сервере переключается через `.env`:

```dotenv
VOICE_VAD_BACKEND=silero_onnx   # или webrtc
```

- `silero_onnx` — ONNX-модель Silero (кэшируется локально на первом использовании).
- `webrtc` — легковесный WebRTC VAD (режим агрессивности задаётся в runtime).

Протокол:


| Направление     | Тип       | Содержимое                                                 |
| --------------- | --------- | ---------------------------------------------------------- |
| client → server | binary    | PCM16 mono 16 kHz (потоково, непрерывный разговор)         |
| client → server | text/JSON | `{"type":"end"}` — завершение сессии                       |
| client → server | text/JSON | `{"type":"interrupt"}` — **перебить** текущий ответ бота   |
| client → server | text/JSON | `{"type":"escalate","reason":"..."}` — передать оператору  |
| server → client | binary    | MP3 (OpenAI TTS или ElevenLabs)                            |
| server → client | text/JSON | `{"type":"session","conversation_id":"..."}` — id для REST |
| server → client | text/JSON | `{"type":"escalation_packet",...}` — саммари + `turns[]`   |
| server → client | text/JSON | `{"type":"interrupted"}` — сервер отменил генерацию        |


Просмотр сохранённого транскрипта: `GET /api/v1/conversations/{conversation_id}`.

### Два пула баз знаний (Qdrant)


| Пул            | Переменная `.env`                | Кто читает                                |
| -------------- | -------------------------------- | ----------------------------------------- |
| `voice`        | `QDRANT_COLLECTION_VOICE`        | Orchestrator, голосовой WebSocket         |
| `agent_assist` | `QDRANT_COLLECTION_AGENT_ASSIST` | Agent Assist WebSocket `/ws/agent-assist` |


Загрузка через форму `knowledge_pool=voice|agent_assist` или UI «База знаний» (вкладки).

### Загрузка документа в базу знаний

```bash
curl -F "file=@knowledge.pdf" -F "title=Тарифы" -F "locale=ru" \
  -F "knowledge_pool=voice" \
  http://localhost:8000/api/v1/documents
```

### Речевая аналитика

- UI: `/analytics` — критерии (код, вес, max_score, rubric), загрузка аудио, отчёт.
- API: `GET/POST /api/v1/analytics/criteria-sets`, `POST .../criteria`, `POST /api/v1/analytics/analyze`.
- Seed создаёт набор `default-qa` с четырьмя демо-критериями.

## Метрики и требования (из ФТ)

- **Latency ≤ 1.5 c** — `timed_stage` логирует duration_ms каждого этапа (`stt`, `llm`, `tts`, `turn`).
- **Intent accuracy ≥ 80%** — многоступенчато: semantic router (локальные эмбеддинги) + semantic intent + keyword/LLM-фолбэк.
- **Локализация ru / uz** — системные промпты и фолбэки для обеих локалей.
- **Безопасность** — никаких хардкодов; провайдеры заменяемы; данные при необходимости остаются в контуре.
- **Webitel / Creatio** — точка интеграции = WebSocket `/ws/voice` (audio in/out) + REST CRUD; адаптер пишется поверх.

## Тесты

```bash
cd apps/backend
PYTHONPATH=../.. pytest
```

## Миграции и seed

Схема создаётся миграциями Alembic (`apps/backend/alembic/`):

```bash
docker compose -f docker/docker-compose.yml exec backend \
  alembic -c apps/backend/alembic.ini upgrade head
```

Загрузить демо-данные (промпты, голоса, **два пула** KB: полный voice + короткий agent_assist, критерии аналитики):

```bash
docker compose -f docker/docker-compose.yml exec backend \
  python -m apps.backend.scripts.seed
```

После этого зайдите в Плейграунд и спросите, например, *«какая ставка по
вкладу „Накопительный“?»* — должен сработать RAG и ответить из контекста.

## Запуск без Deepgram / ElevenLabs

В `.env` достаточно одного `OPENAI_API_KEY`:

```dotenv
STT_PROVIDER=openai      # Whisper batch с буферизацией ~1.2 c
LLM_PROVIDER=openai      # gpt-4o-mini
TTS_PROVIDER=openai      # tts-1, голос alloy / nova / ...
```

Когда придут ключи — переключаете `STT_PROVIDER=deepgram` / `TTS_PROVIDER=elevenlabs`,
никакой код менять не нужно (Strategy в `apps/backend/services/factory.py`).

## Локальные провайдеры (`local_http`)

Подключение **on-prem** или своих сервисов без смены оркестратора и Voice Engine:


| Переменная                | Назначение                                                                                                                                                                                                                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `STT_PROVIDER=local_http` | `LOCAL_STT_URL` — **POST** multipart, поле `file`: WAV mono 16 kHz; ответ JSON с полем `text` / `transcript` / `result`.                                                                                                                                  |
| `LLM_PROVIDER=local_http` | `LOCAL_LLM_BASE_URL` — база **без** хвоста `/v1` (например `http://ollama:11434`); вызываются `POST .../v1/chat/completions` и `POST .../v1/embeddings`. Модели: `LOCAL_LLM_CHAT_MODEL`, `LOCAL_LLM_EMBED_MODEL`. При необходимости: `LOCAL_LLM_API_KEY`. |
| `TTS_PROVIDER=local_http` | `LOCAL_TTS_URL` — **POST** JSON `{"text","voice_id","locale"}`; тело ответа — бинарное аудио (целиком режется на чанки для WebSocket). Опционально в теле передаётся `tts_params` из профиля голоса.                                                      |


Подробнее переменные перечислены в `.env.example`.

## Локальные модели и кэш

При первом использовании backend может автоматически скачать локальные артефакты:

- **Silero VAD ONNX** (для `VOICE_VAD_BACKEND=silero_onnx`)
- **MiniLM embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) для semantic router/intent

Это normal behavior: на холодном старте возможна небольшая задержка первого запроса. Для стабильного production обычно монтируют volume под кэш модели.

## Поведение бота и голос (админка + API)

- **UI:** страница [Поведение бота](http://localhost:3000/bot) (`/bot`) — редактирование JSON настроек.
- **API:** `GET /api/v1/bot-settings`, `PUT /api/v1/bot-settings` — частичный merge в БД (таблица `bot_runtime_settings`, одна строка `id=1`).

Типичные ключи в payload:


| Ключ                                                       | Смысл                                                                                                                      |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `llm_temperature`, `llm_max_tokens`                        | Параметры генерации в оркестраторе и SSE-плейграунде.                                                                      |
| `history_max_messages`                                     | Сколько последних сообщений истории уходит в LLM (и обрезка буфера в Voice Engine).                                        |
| `system_prompt_suffix`                                     | Дополнительный блок в системном промпте (политика компании и т.п.).                                                        |
| `suppress_repeated_greeting`                               | Если `true` и в истории уже есть реплики ассистента — в промпт добавляется правило не начинать ответ с приветствия заново. |
| `openai_tts_speed`                                         | Скорость синтеза OpenAI TTS (0.25–4.0), пока не переопределено в профиле голоса.                                           |
| `elevenlabs_stability`, `elevenlabs_similarity_boost`      | Аналогично для ElevenLabs.                                                                                                 |
| `vad_silero_speech_threshold`, `vad_webrtc_aggressiveness` | Порог/чувствительность server-side VAD перед STT (Silero/WebRTC).                                                          |
| `semantic_router_embed_enabled`, `router_embed_`*          | Локальный семантический роутер (noise/simple/complex/escalation) по косинусному сходству.                                  |
| `semantic_intent_embed_enabled`, `semantic_intent_`*       | Локальная semantic intent-классификация до keyword/LLM-фолбэка.                                                            |


**Профиль голоса:** в таблице `voices` есть колонка `tts_params` (JSONB). В админке «Голоса» — поле под каждым профилем; API: `PATCH /api/v1/voices/{id}` с телом `{"tts_params":{...}}`. Переопределения: например `{"speed":1.1}` для OpenAI, `{"stability":0.4,"similarity_boost":0.8}` для ElevenLabs.

После миграции `0005_bot_runtime_voice_tts` выполните `alembic upgrade head`.

## Диаризация в аналитике: Deepgram и pyannote

- `DIARIZATION_PROVIDER=deepgram` — в сервис аналитики передаётся сырой файл записи; бэкенд вызывает `POST https://api.deepgram.com/v1/listen` с `diarize=true`, `utterances=true`, парсит `results.utterances` и мапит в `DiarizedTurn`. Нужен `DEEPGRAM_API_KEY`. Эвристика спикеров: `0 → agent`, `1 → customer` (типичный порядок в моно-звонке); при одном спикере — роль `unknown`.
- `DIARIZATION_PROVIDER=pyannote` — тяжёлая модель не встроена в образ API. Ожидается отдельный сервис по адресу `PYANNOTE_WORKER_URL`: POST `multipart/form-data`, поле `file` — то же аудио, что загрузил пользователь. Ответ JSON:

```json
{
  "turns": [
    {
      "speaker": "customer",
      "text": "…",
      "start_sec": 0.0,
      "end_sec": 2.5
    }
  ]
}
```

`speaker` — одно из: `customer`, `agent`, `unknown`. Если URL пуст или воркер недоступен, пайплайн не падает: возвращается один блок с полным текстом транскрипта и `unknown`. Такой воркер обычно поднимают в контейнере **с GPU** и очередью задач (Celery, Redis, отдельный микросервис).

## Что важно перед production

- Закрепить кэш моделей (Silero/MiniLM) через volume или prewarm на этапе деплоя.
- Добавить инфраструктурные метрики (Prometheus/Grafana) поверх текущих SLO-метрик API.
- Настроить адаптеры внешних систем (телефония/CRM) поверх `/ws/voice` и REST API.

