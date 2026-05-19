# llama-cpp-gpt-api

OpenAI-совместимый HTTP API на [Gin](https://github.com/gin-gonic/gin) поверх [go-llama-new.cpp](https://github.com/redirsvr/go-llama-new.cpp) (llama.cpp). Поддерживает chat completions, embeddings, реестр моделей, **Hybrid RAG** (PostgreSQL + pgvector) и веб-интерфейс загрузки документов.

Конфигурация — только из YAML-файла (переменные окружения не используются).

## Возможности

- **Chat Completions** — OpenAI-формат, ChatML/Qwen, streaming, `use_rag` для подмешивания контекста из базы знаний
- **Embeddings** — отдельные embedding-модели из каталога
- **Модели** — `GET /v1/models`, алиасы по имени файла `.gguf`
- **Hybrid RAG** — семантический чанкинг, pgvector + полнотекстовый поиск (RRF), авто-ingest каталогов
- **Веб-UI** — `/ui/` для загрузки файлов и текста
- **Prometheus** — `/metrics` (инференс, HTTP, Go runtime)
- **Grafana** — готовый дашборд в `grafana/dashboards/gpt-api.json`

## Требования

- Go 1.25+
- GCC/G++, CMake
- CUDA (опционально, для GPU)
- Локальный клон [go-llama-new.cpp](../go-llama-new.cpp) рядом с проектом (см. `replace` в `go.mod`)
- Для RAG: PostgreSQL 16+ с расширением [pgvector](https://github.com/pgvector/pgvector)

## Быстрый старт

Полная справка по целям Makefile:

```bash
make help
```

**Одна строка** — binding + Swagger + бинарник `bin/gpt-api`:

```bash
make all
```

Конфиг и запуск:

```bash
make config    # etc/gpt-api.yaml из примера (если файла ещё нет)
make run       # запуск (сборка только если нет bin/gpt-api)
```

Сборка с GPU:

```bash
make cuda      # NVIDIA (см. ../go-llama-new.cpp/build.conf)
make rocm      # AMD ROCm
```

RAG PostgreSQL (опционально):

```bash
make rag-db    # docker compose в deploy/rag-postgres/
```

### Другие цели

| Команда | Описание |
|---------|----------|
| `make check` | Проверка Go, cmake, g++, каталога `../go-llama-new.cpp` |
| `make binding` | Только `libbinding.a` (llama.cpp + CGO) |
| `make app` | Только Go-сборка (binding уже собран) |
| `make swagger` | Перегенерация `docs/` (OpenAPI) |
| `make test` | `go test` пакетов без полного CGO main |
| `make clean` | Удалить `bin/gpt-api` |

Обёртка `./build.sh [all\|cpu\|cuda\|rocm]` вызывает Makefile в `go-llama-new.cpp`.

В `go.mod`:

```go
replace github.com/redirsvr/go-llama-new.cpp => ../go-llama-new.cpp
```

Требуется каталог `../go-llama-new.cpp` с подкаталогом `llama.cpp`.

## Запуск (вручную)

```bash
./bin/gpt-api -f etc/gpt-api.yaml
```

Сервер по умолчанию: `http://0.0.0.0:8080`  
Swagger UI: `http://localhost:8080/docs/`

## Конфигурация

Основной файл: `etc/gpt-api.yaml`. Пример с GPU и RAG: `etc/gpt-api-gpu.example.yaml`.

| Параметр | Описание |
|----------|----------|
| `ModelsDir` | Каталог с `.gguf` / `.bin` |
| `DefaultModel` | Алиас модели для chat (имя без расширения) |
| `EmbeddingModels` | Список алиасов для `/v1/embeddings` |
| `ModelOption` | Параметры llama.cpp: `NGPULayers`, `TensorSplit`, `ContextSize` и др. |
| `SystemPrompt` | Первое system-сообщение (как в OpenAI) |
| `DisableThinking` | Для Qwen3: `/no_think` в промпте |
| `PreloadDefaultModel` | `false` — не грузить модель при старте |
| `RAG` | PostgreSQL, чанкинг, hybrid search, auto-ingest |

### GPU: важно

- **`NGPULayers: -1`** — `fit_params` подбирает слои под VRAM (без ручного `TensorSplit`).
- **`TensorSplit` + `NGPULayers: -1`** — несовместимы: укажите явное число слоёв, например `80`, и `TensorSplit: "1,1"` для 2 GPU.
- **`NGPULayers: 0` в логе** — проверьте, что в YAML числа парсятся (после исправления `ReflectVal` значения из YAML применяются корректно).

## API

### Chat

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B-Q4_K_M",
    "messages": [{"role": "user", "content": "Привет!"}],
    "max_tokens": 256
  }'
```

С RAG:

```json
{
  "use_rag": true,
  "rag_top_k": 8,
  "messages": [{"role": "user", "content": "Вопрос по документам"}]
}
```

### Embeddings

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "nomic-embed-text-v1.5", "input": "текст"}'
```

### Модели

```bash
curl -s http://localhost:8080/v1/models
curl -s http://localhost:8080/v1/models/Qwen3.6-35B-A3B-Q4_K_M
```

## Hybrid RAG

### PostgreSQL

```bash
docker compose -f deploy/rag-postgres/docker-compose.yaml up -d
```

В конфиге:

```yaml
RAG:
  Enabled: true
  Postgres:
    Host: 127.0.0.1
    Port: 5432
    Database: gpt_rag
    User: gpt
    Password: gpt
  EmbeddingModel: "nomic-embed-text-v1.5"
  EmbeddingDimensions: 768
```

### Веб-интерфейс

Откройте в браузере: **http://localhost:8080/ui/**

- загрузка файлов (drag & drop)
- вставка текста
- список и удаление документов
- проверка гибридного поиска

Отключить UI: `RAG.UIDisabled: true`

Статика `/ui/` (HTML, `app.js`, `app.css`) вшита в бинарник через `go:embed` — отдельно копировать файлы на сервер не нужно.

### REST API RAG

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/v1/rag/documents` | JSON: `title`, `content` или `path` |
| POST | `/v1/rag/documents/upload` | multipart `file` |
| POST | `/v1/rag/documents/scan` | Скан каталогов из `AutoIngest.Directories` |
| GET | `/v1/rag/documents` | Список документов |
| DELETE | `/v1/rag/documents/:id` | Удаление |
| POST | `/v1/rag/query` | Гибридный поиск: `{"query":"...", "top_k":8}` |

Семантический чанкинг: эмбеддинги предложений → границы по падению cosine similarity (`BreakpointPercentile`), затем объединение с лимитом `MaxChunkChars`.

## Мониторинг

### Prometheus

```bash
curl -s http://localhost:8080/metrics | grep gpt_api_
```

Метрики: HTTP, chat (токены/с, длительность predict), embeddings, загрузка моделей, `go_*`, `process_*`.

### Grafana

```bash
cd grafana && docker compose up -d
```

- Grafana: http://localhost:3000 (`admin` / `admin`)
- Импорт дашборда: `grafana/dashboards/gpt-api.json` (uid: `gpt-api-llama`)

## Структура проекта

```
.
├── main.go                 # точка входа
├── Makefile                # make help | make all | make run
├── build.sh                # обёртка: сборка go-llama-new.cpp
├── docs/                   # OpenAPI (swag), UI: /docs/
├── etc/                    # примеры конфигов YAML
├── internal/
│   ├── api/                # Gin: маршруты, handlers
│   ├── embedassets/        # go:embed — HTML/CSS/JS для /ui/ (в бинарнике)
│   ├── config/             # загрузка YAML
│   ├── logic/gpt/          # chat, embeddings, models
│   └── types/              # DTO запросов/ответов
├── pkg/
│   ├── model/              # реестр моделей, промпты, GPU-опции
│   ├── rag/                # RAG: чанкинг, PostgreSQL, hybrid search
│   └── metrics/            # Prometheus
├── deploy/rag-postgres/    # Docker Compose для pgvector
└── grafana/                # Prometheus + дашборд
```

## Разработка

```bash
make test
make app          # быстрая пересборка API после изменений Go
make swagger      # после правок аннотаций в internal/api/
```

Файлы `restapi/*.api` — legacy-описание для goctl; актуальные маршруты — `internal/api/router.go`.

## Лицензия

См. репозиторий и зависимости (llama.cpp, go-llama-new.cpp).
