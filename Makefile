# llama-cpp-gpt-api — сборка и запуск
# Справка:  make help
# Одна строка до готового бинарника:  make all

SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LLAMA_DIR    ?= $(abspath $(PROJECT_ROOT)/../go-llama-new.cpp)
BIN_DIR      := $(PROJECT_ROOT)/bin
BINARY       := $(BIN_DIR)/gpt-api
CONFIG       ?= $(PROJECT_ROOT)/etc/gpt-api.yaml
CONFIG_EXAMPLE := $(PROJECT_ROOT)/etc/gpt-api-gpu.example.yaml
GO           ?= go
SWAG         := $(shell $(GO) env GOPATH)/bin/swag
NPROC        := $(shell nproc 2>/dev/null || echo 4)

# Бэкенд llama.cpp: cpu | cuda | rocm (см. ../go-llama-new.cpp/build.conf)
BACKEND ?=

# Цель llama-make: all, cpu, cuda, rocm
LLAMA_GOAL := all
ifneq ($(filter cuda rocm cpu,$(MAKECMDGOALS)),)
  LLAMA_GOAL := $(firstword $(filter cuda rocm cpu,$(MAKECMDGOALS)))
  BACKEND := $(LLAMA_GOAL)
endif
ifneq ($(strip $(BACKEND)),)
  LLAMA_GOAL := $(BACKEND)
endif

.PHONY: help all build app binding swagger run config deps test check clean clean-binding clean-all \
        rag-db rag-db-down install-swag cpu cuda rocm llama-binding

# --- Справка -----------------------------------------------------------------

help: ## Показать эту справку
	@echo "llama-cpp-gpt-api — OpenAI API + Hybrid RAG"
	@echo ""
	@echo "Быстрый старт (одна строка до бинарника):"
	@echo "  make all          binding (go-llama-new.cpp) + swagger + bin/gpt-api"
	@echo "  make config       создать etc/gpt-api.yaml из примера (если нет)"
	@echo "  make run          запуск (make all, если ещё нет bin/gpt-api)"
	@echo ""
	@echo "GPU (NVIDIA / AMD):"
	@echo "  make cuda         полная сборка с CUDA"
	@echo "  make rocm         полная сборка с ROCm"
	@echo "  BACKEND=cuda make all"
	@echo ""
	@echo "Цели:"
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Пути:"
	@echo "  Бинарник:     $(BINARY)"
	@echo "  Конфиг:       $(CONFIG)"
	@echo "  go-llama-new: $(LLAMA_DIR)"
	@echo "  Swagger UI:   http://localhost:8080/docs/"
	@echo "  RAG Web UI:   http://localhost:8080/ui/  (HTML/JS/CSS в бинарнике)"

# --- Полная сборка -----------------------------------------------------------

all: check deps binding swagger app ## Полная сборка → bin/gpt-api
	@echo ""
	@echo "Готово: $(BINARY)"
	@echo "Дальше: make config && make run"

build: all ## То же, что make all (алиас)

cpu cuda rocm: check deps binding swagger app ## Сборка с бэкендом cpu/cuda/rocm
	@echo ""
	@echo "Готово ($(LLAMA_GOAL)): $(BINARY)"

# --- Части сборки -----------------------------------------------------------

binding: ## Собрать libbinding.a в ../go-llama-new.cpp
	@$(MAKE) llama-binding

.PHONY: llama-binding
llama-binding:
	@test -d "$(LLAMA_DIR)" || { \
		echo "Ошибка: не найден $(LLAMA_DIR)"; \
		echo "Клонируйте go-llama-new.cpp рядом с репозиторием (см. go.mod replace)."; \
		exit 1; \
	}
	@test -d "$(LLAMA_DIR)/llama.cpp" || { \
		echo "Ошибка: нет $(LLAMA_DIR)/llama.cpp (submodule или клон llama.cpp)."; \
		exit 1; \
	}
	@echo "==> go-llama-new.cpp (цель: $(LLAMA_GOAL))"
	$(MAKE) -C "$(LLAMA_DIR)" $(LLAMA_GOAL)

swagger: install-swag ## Перегенерировать docs/ (OpenAPI)
	@test -x "$(SWAG)" || { echo "swag не установился в $(SWAG)"; exit 1; }
	@echo "==> swagger"
	"$(SWAG)" init -g main.go -o docs --parseDependency --parseInternal

install-swag:
	@command -v "$(SWAG)" >/dev/null 2>&1 || { \
		echo "==> go install swag"; \
		$(GO) install github.com/swaggo/swag/cmd/swag@latest; \
	}

app: ## Собрать только Go-приложение (нужен готовый libbinding.a)
	@test -f "$(LLAMA_DIR)/libbinding.a" || { \
		echo "Нет $(LLAMA_DIR)/libbinding.a — сначала: make binding"; \
		exit 1; \
	}
	@mkdir -p "$(BIN_DIR)"
	@echo "==> go build → $(BINARY)"
	CGO_ENABLED=1 $(GO) build -o "$(BINARY)" "$(PROJECT_ROOT)"

# --- Запуск и конфиг ---------------------------------------------------------

config: ## Создать etc/gpt-api.yaml из etc/gpt-api-gpu.example.yaml
	@if [[ -f "$(CONFIG)" ]]; then \
		echo "Конфиг уже есть: $(CONFIG)"; \
	else \
		cp "$(CONFIG_EXAMPLE)" "$(CONFIG)"; \
		echo "Создан $(CONFIG) — отредактируйте ModelsDir, RAG.Postgres и т.д."; \
	fi

run: config ## Запустить сервер (make all, если нет bin/gpt-api)
	@if [[ ! -x "$(BINARY)" ]]; then \
		echo "Бинарник не найден — полная сборка..."; \
		$(MAKE) all; \
	fi
	@echo "==> $(BINARY) -f $(CONFIG)"
	"$(BINARY)" -f "$(CONFIG)"

# --- Зависимости и проверки --------------------------------------------------

deps: ## go mod download + swag (для swagger)
	@echo "==> go mod download"
	$(GO) mod download
	@$(MAKE) install-swag

check: ## Проверить go, cmake, g++, каталог go-llama-new.cpp
	@echo "==> проверка окружения"
	@command -v $(GO) >/dev/null || { echo "Нужен Go (1.25+)"; exit 1; }
	@command -v cmake >/dev/null || { echo "Нужен cmake"; exit 1; }
	@command -v g++ >/dev/null 2>/dev/null || command -v c++ >/dev/null || { echo "Нужен g++"; exit 1; }
	@test -d "$(LLAMA_DIR)" || { echo "Нет $(LLAMA_DIR)"; exit 1; }
	@test -d "$(LLAMA_DIR)/llama.cpp" || { echo "Нет $(LLAMA_DIR)/llama.cpp"; exit 1; }
	@echo "OK: Go=$$($(GO) version | awk '{print $$3}')  LLAMA_DIR=$(LLAMA_DIR)"

test: ## go test (без CGO main)
	$(GO) test ./pkg/... ./internal/... -count=1

# --- RAG PostgreSQL ----------------------------------------------------------

rag-db: ## Поднять PostgreSQL+pgvector (docker compose)
	docker compose -f "$(PROJECT_ROOT)/deploy/rag-postgres/docker-compose.yaml" up -d

rag-db-down: ## Остановить контейнер RAG Postgres
	docker compose -f "$(PROJECT_ROOT)/deploy/rag-postgres/docker-compose.yaml" down

# --- Очистка -----------------------------------------------------------------

clean: ## Удалить bin/gpt-api
	rm -f "$(BINARY)"
	@rmdir "$(BIN_DIR)" 2>/dev/null || true

clean-binding: ## Очистить libbinding.a в go-llama-new.cpp
	$(MAKE) -C "$(LLAMA_DIR)" clean

clean-all: clean clean-binding ## clean + clean-binding
