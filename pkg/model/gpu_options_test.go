package model

import (
    "testing"

    "llama-cpp-gpt-api/internal/config"

    llama "github.com/redirsvr/go-llama-new.cpp"
)

func TestValidateModelOptionGPU(t *testing.T) {
    config.C = config.Config{
        ModelOption: map[string]interface{}{
            "NGPULayers":  -1,
            "TensorSplit": "1,1",
        },
    }
    if err := ValidateModelOptionGPU(); err != nil {
        t.Fatalf("auto-fit with TensorSplit in yaml should pass (cleared at load): %v", err)
    }

    config.C.ModelOption = map[string]interface{}{
        "NGPULayers": -1,
    }
    if err := ValidateModelOptionGPU(); err != nil {
        t.Fatalf("fit-only config should pass: %v", err)
    }

    config.C.ModelOption = map[string]interface{}{
        "NGPULayers":  80,
        "TensorSplit": "1,1",
    }
    if err := ValidateModelOptionGPU(); err != nil {
        t.Fatalf("explicit layers + split should pass: %v", err)
    }
}

func TestApplyAutoGPUFit(t *testing.T) {
    src := map[string]interface{}{
        "NGPULayers":  -1,
        "TensorSplit": "1,1",
    }
    p := &llama.ModelOptions{
        NGPULayers:  -1,
        TensorSplit: "1,1",
    }
    ApplyAutoGPUFit(p, src)
    if p.TensorSplit != "" {
        t.Fatalf("TensorSplit = %q, want empty for auto-fit", p.TensorSplit)
    }
    if p.NGPULayers != -1 {
        t.Fatalf("NGPULayers = %d, want -1", p.NGPULayers)
    }

    p = &llama.ModelOptions{NGPULayers: 24, TensorSplit: "1,1"}
    ApplyAutoGPUFit(p, map[string]interface{}{"AutoGPU": true})
    if p.TensorSplit != "" || p.NGPULayers != -1 {
        t.Fatalf("AutoGPU should force fit: TensorSplit=%q NGPULayers=%d", p.TensorSplit, p.NGPULayers)
    }

    p = &llama.ModelOptions{NGPULayers: 32, TensorSplit: "1,1"}
    ApplyAutoGPUFit(p, map[string]interface{}{"NGPULayers": 32})
    if p.TensorSplit != "1,1" || p.NGPULayers != 32 {
        t.Fatalf("manual offload must stay: TensorSplit=%q NGPULayers=%d", p.TensorSplit, p.NGPULayers)
    }
}
