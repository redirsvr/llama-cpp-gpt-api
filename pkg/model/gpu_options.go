package model

import (
    "encoding/json"
    "fmt"
    "strings"

    "llama-cpp-gpt-api/internal/config"
)

// ValidateModelOptionGPU проверяет сочетание NGPULayers и TensorSplit.
func ValidateModelOptionGPU() error {
    if config.C.ModelOption == nil {
        return nil
    }
    ts := strings.TrimSpace(modelOptionString("TensorSplit"))
    ngpu, hasNGPU := modelOptionInt("NGPULayers")
    if ts == "" || !hasNGPU || ngpu >= 0 {
        return nil
    }
    return fmt.Errorf(
        "ModelOption: TensorSplit=%q несовместим с NGPULayers=%d. "+
            "Вариант 1 — авто VRAM (1 GPU или fit сам разложит веса): уберите TensorSplit, оставьте NGPULayers: -1. "+
            "Вариант 2 — 2+ GPU с ручным split: задайте NGPULayers явно (например 80), TensorSplit: \"1,1\"",
        ts, ngpu,
    )
}

func modelOptionString(key string) string {
    v, ok := config.C.ModelOption[key]
    if !ok {
        return ""
    }
    s, _ := v.(string)
    return s
}

func modelOptionInt(key string) (int, bool) {
    v, ok := config.C.ModelOption[key]
    if !ok {
        return 0, false
    }
    switch n := v.(type) {
    case int:
        return n, true
    case int64:
        return int(n), true
    case float64:
        return int(n), true
    case json.Number:
        i, err := n.Int64()
        if err != nil {
            return 0, true
        }
        return int(i), true
    default:
        return 0, true
    }
}
