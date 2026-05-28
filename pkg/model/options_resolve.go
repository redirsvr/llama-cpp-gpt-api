package model

import (
	"llama-cpp-gpt-api/internal/config"
)

// ChatOptionInt возвращает числовую опцию chat-модели с учётом models-preset.
func ChatOptionInt(alias, key string) int {
	m := modelOptionsWithPreset(config.C.ModelOption, alias, "", false)
	if m == nil {
		return 0
	}
	v, ok := m[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}
