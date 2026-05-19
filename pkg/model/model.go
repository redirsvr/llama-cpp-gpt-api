package model

import (
    "encoding/json"
    "math"
    "strconv"

    "llama-cpp-gpt-api/internal/types"

    llama "github.com/redirsvr/go-llama-new.cpp"
)

func coerceInt(v interface{}) (int, bool) {
    switch n := v.(type) {
    case int:
        return n, true
    case int64:
        return int(n), true
    case int32:
        return int(n), true
    case float64:
        if math.IsNaN(n) || math.IsInf(n, 0) {
            return 0, false
        }
        return int(n), true
    case float32:
        return int(n), true
    case json.Number:
        i, err := n.Int64()
        if err != nil {
            return 0, false
        }
        return int(i), true
    case string:
        i, err := strconv.Atoi(stringsTrim(n))
        if err != nil {
            return 0, false
        }
        return i, true
    default:
        return 0, false
    }
}

func coerceFloat32(v interface{}) (float32, bool) {
    switch n := v.(type) {
    case float32:
        return n, true
    case float64:
        return float32(n), true
    case int:
        return float32(n), true
    case int64:
        return float32(n), true
    case json.Number:
        f, err := n.Float64()
        if err != nil {
            return 0, false
        }
        return float32(f), true
    case string:
        f, err := strconv.ParseFloat(stringsTrim(n), 32)
        if err != nil {
            return 0, false
        }
        return float32(f), true
    default:
        return 0, false
    }
}

func stringsTrim(s string) string {
    i, j := 0, len(s)
    for i < j && (s[i] == ' ' || s[i] == '\t') {
        i++
    }
    for j > i && (s[j-1] == ' ' || s[j-1] == '\t') {
        j--
    }
    return s[i:j]
}

func ReflectVal(k string, v interface{}, p *llama.ModelOptions) {
    intValue, hasInt := coerceInt(v)
    floatValue, hasFloat := coerceFloat32(v)

    switch k {
    case "ContextSize":
        if hasInt {
            p.ContextSize = intValue
        }
    case "Seed":
        if hasInt {
            p.Seed = intValue
        }
    case "NBatch":
        if hasInt {
            p.NBatch = intValue
        }
    case "F16Memory":
        if b, ok := v.(bool); ok {
            p.F16Memory = b
        }
    case "MLock":
        if b, ok := v.(bool); ok {
            p.MLock = b
        }
    case "MMap":
        if b, ok := v.(bool); ok {
            p.MMap = b
        }
    case "LowVRAM":
        if b, ok := v.(bool); ok {
            p.LowVRAM = b
        }
    case "Embeddings":
        // Устарело: режим embeddings задаётся через EmbeddingModels в конфиге.
    case "NUMA":
        if b, ok := v.(bool); ok {
            p.NUMA = b
        }
    case "NGPULayers":
        if hasInt {
            p.NGPULayers = intValue
        }
    case "MainGPU":
        if s, ok := v.(string); ok {
            p.MainGPU = s
        }
    case "TensorSplit":
        if s, ok := v.(string); ok {
            p.TensorSplit = s
        }
    case "FreqRopeBase":
        if hasFloat {
            p.FreqRopeBase = floatValue
        }
    case "FreqRopeScale":
        if hasFloat {
            p.FreqRopeScale = floatValue
        }
    case "LoraBase":
        if s, ok := v.(string); ok {
            p.LoraBase = s
        }
    case "LoraAdapter":
        if s, ok := v.(string); ok {
            p.LoraAdapter = s
        }
    case "Perplexity":
        if b, ok := v.(bool); ok {
            p.Perplexity = b
        }
    }
}

// ConvertMessages2Text — устаревшее имя, используйте FormatChatPrompt.
func ConvertMessages2Text(modelAlias string, messages []types.Message) string {
    return FormatChatPrompt(modelAlias, messages)
}
