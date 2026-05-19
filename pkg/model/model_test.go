package model

import (
    "testing"

    llama "github.com/redirsvr/go-llama-new.cpp"
)

func TestReflectValYAMLInts(t *testing.T) {
    p := &llama.ModelOptions{}
    ReflectVal("NGPULayers", -1, p)
    if p.NGPULayers != -1 {
        t.Fatalf("NGPULayers = %d, want -1", p.NGPULayers)
    }
    ReflectVal("ContextSize", int64(8192), p)
    if p.ContextSize != 8192 {
        t.Fatalf("ContextSize = %d, want 8192", p.ContextSize)
    }
    ReflectVal("NBatch", float64(512), p)
    if p.NBatch != 512 {
        t.Fatalf("NBatch = %d, want 512", p.NBatch)
    }
}
