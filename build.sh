#!/usr/bin/env bash
# Обёртка над Makefile go-llama-new.cpp (обратная совместимость).
# Предпочтительно: make binding   или   make cuda
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/../go-llama-new.cpp}"
GOAL="${1:-all}"

if [[ ! -d "$LLAMA_DIR" ]]; then
	echo "Ошибка: не найден $LLAMA_DIR" >&2
	echo "Клонируйте go-llama-new.cpp рядом с репозиторием." >&2
	exit 1
fi

case "$GOAL" in
	cpu|cuda|rocm|all|clean|help)
		exec make -C "$LLAMA_DIR" "$GOAL"
		;;
	*)
		echo "Использование: $0 [all|cpu|cuda|rocm|clean|help]" >&2
		exit 1
		;;
esac
