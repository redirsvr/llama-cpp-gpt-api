package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadUnloadModelsFromGPU(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "gpt-api.yaml")
	if err := os.WriteFile(cfgPath, []byte("UnloadModelsFromGPU: true\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := Load(cfgPath); err != nil {
		t.Fatal(err)
	}
	if !C.UnloadModelsFromGPU {
		t.Fatal("UnloadModelsFromGPU should be true")
	}
}

func TestLoadModelPresetsFromDefaultFile(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "gpt-api.yaml")
	if err := os.WriteFile(cfgPath, []byte("ModelsDir: /tmp/models\n"), 0644); err != nil {
		t.Fatal(err)
	}
	preset := []byte(`
Models:
  - File: chat-model.gguf
    Alias: chat
    ContextSize: 32768
    NBatch: 512
  - File: embed-model.gguf
    Alias: embed
    EmbeddingModelOption:
      ContextSize: 2048
      NBatch: 2048
`)
	if err := os.WriteFile(filepath.Join(dir, "models-preset.yaml"), preset, 0644); err != nil {
		t.Fatal(err)
	}

	if err := Load(cfgPath); err != nil {
		t.Fatal(err)
	}
	if got := ModelPresets["chat"].ModelOption["ContextSize"]; got != 32768 {
		t.Fatalf("chat ContextSize = %#v, want 32768", got)
	}
	if got := ModelPresets["embed"].EmbeddingModelOption["NBatch"]; got != 2048 {
		t.Fatalf("embed NBatch = %#v, want 2048", got)
	}
}
