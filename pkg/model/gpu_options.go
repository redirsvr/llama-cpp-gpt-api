package model

import (
    "encoding/json"
    "fmt"
    "log"
    "strings"

    "llama-cpp-gpt-api/internal/config"

    llama "github.com/redirsvr/go-llama-new.cpp"
)

// ValidateModelOptionGPU проверяет сочетание NGPULayers и TensorSplit.
// При авто-режиме (NGPULayers < 0 или AutoGPU) TensorSplit сбрасывается в ApplyAutoGPUFit.
func ValidateModelOptionGPU() error {
    if config.C.ModelOption == nil {
        return nil
    }
    ts := strings.TrimSpace(modelOptionString("TensorSplit"))
    ngpu, hasNGPU := modelOptionInt("NGPULayers")
    auto, hasAuto := modelOptionBool("AutoGPU")
    if ts == "" || !hasNGPU || ngpu >= 0 {
        return nil
    }
    if (hasAuto && auto) || ngpu < 0 {
        return nil
    }
    return fmt.Errorf(
        "ModelOption: TensorSplit=%q несовместим с NGPULayers=%d. "+
            "Авто по VRAM: NGPULayers: -1 (или AutoGPU: true) без TensorSplit — fit_params разложит слои и веса по GPU. "+
            "Ручной split: явное NGPULayers (например 80) и TensorSplit: \"1,1\"",
        ts, ngpu,
    )
}

// ApplyAutoGPUFit включает fit_params llama.cpp: число GPU-слоёв и tensor_split
// подбираются по свободной VRAM на всех устройствах. Ручной TensorSplit сбрасывается.
func ApplyAutoGPUFit(p *llama.ModelOptions, src map[string]interface{}) {
    if src == nil || !wantAutoGPUFit(src, p) {
        return
    }
    if ts := strings.TrimSpace(p.TensorSplit); ts != "" {
        log.Printf("models: auto GPU fit: сбрасываем TensorSplit=%q (fit_params распределит веса по GPU)", ts)
        p.TensorSplit = ""
    }
    if p.NGPULayers >= 0 {
        p.NGPULayers = -1
    }
}

func wantAutoGPUFit(src map[string]interface{}, p *llama.ModelOptions) bool {
    if auto, ok := optionBool(src, "AutoGPU"); ok && auto {
        return true
    }
    if p.NGPULayers < 0 {
        return true
    }
    if ngpu, ok := optionInt(src, "NGPULayers"); ok && ngpu < 0 {
        return true
    }
    return false
}

func modelOptionString(key string) string {
    if config.C.ModelOption == nil {
        return ""
    }
    return optionString(config.C.ModelOption, key)
}

func modelOptionInt(key string) (int, bool) {
    if config.C.ModelOption == nil {
        return 0, false
    }
    return optionInt(config.C.ModelOption, key)
}

func modelOptionBool(key string) (bool, bool) {
    if config.C.ModelOption == nil {
        return false, false
    }
    return optionBool(config.C.ModelOption, key)
}

func optionString(src map[string]interface{}, key string) string {
    v, ok := src[key]
    if !ok {
        return ""
    }
    s, _ := v.(string)
    return s
}

func optionInt(src map[string]interface{}, key string) (int, bool) {
    v, ok := src[key]
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

func optionBool(src map[string]interface{}, key string) (bool, bool) {
    v, ok := src[key]
    if !ok {
        return false, false
    }
    switch b := v.(type) {
    case bool:
        return b, true
    case string:
        s := strings.TrimSpace(strings.ToLower(b))
        return s == "true" || s == "1" || s == "yes" || s == "on", true
    default:
        return false, false
    }
}
