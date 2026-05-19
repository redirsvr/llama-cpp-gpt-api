package model

import (
    "testing"

    "llama-cpp-gpt-api/internal/config"
)

func TestValidateModelOptionGPU(t *testing.T) {
    config.C = config.Config{
        ModelOption: map[string]interface{}{
            "NGPULayers":  -1,
            "TensorSplit": "1,1",
        },
    }
    if err := ValidateModelOptionGPU(); err == nil {
        t.Fatal("expected error for TensorSplit + NGPULayers -1")
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
