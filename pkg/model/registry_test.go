package model

import (
    "errors"
    "testing"

    "llama-cpp-gpt-api/internal/config"
)

func TestNormalizeAlias(t *testing.T) {
    tests := []struct {
        in, want string
    }{
        {"Qwen3.6-35B-A3B-Q4_K_M.gguf", "Qwen3.6-35B-A3B-Q4_K_M"},
        {"Qwen3.6-35B-A3B-Q4_K_M", "Qwen3.6-35B-A3B-Q4_K_M"},
        {"models/foo/bar.gguf", "bar"},
        {"foo.bin", "foo"},
    }
    for _, tc := range tests {
        if got := NormalizeAlias(tc.in); got != tc.want {
            t.Errorf("NormalizeAlias(%q) = %q, want %q", tc.in, got, tc.want)
        }
    }
}

func TestResolveEmbeddingNotSupported(t *testing.T) {
    config.C = config.Config{}
    r := &Registry{
        aliases: map[string]string{
            "chat-only":   "/tmp/chat.gguf",
            "embed-model": "/tmp/embed.gguf",
        },
        embeddingAliases: map[string]struct{}{
            "embed-model": {},
        },
    }

    _, _, err := r.resolve("chat-only", loadModeEmbeddings)
    if err == nil {
        t.Fatal("expected error for chat-only model in embeddings mode")
    }
    if !errors.Is(err, ErrEmbeddingNotSupported) {
        t.Fatalf("expected ErrEmbeddingNotSupported, got %v", err)
    }

    _, _, err = r.resolve("embed-model", loadModeEmbeddings)
    if err != nil {
        t.Fatalf("embed-model should be allowed: %v", err)
    }
}

func TestUnloadOthersRemovesOtherModels(t *testing.T) {
	config.C = config.Config{UnloadModelsFromGPU: true}
	r := &Registry{
		loaded: map[string]*loadedEntry{
			"chat-model:chat":         {},
			"embed-model:embeddings": {},
		},
	}
	r.mu.Lock()
	r.unloadOthersLocked("embed-model:embeddings")
	r.mu.Unlock()

	if _, ok := r.loaded["chat-model:chat"]; ok {
		t.Fatal("chat model should be unloaded")
	}
	if _, ok := r.loaded["embed-model:embeddings"]; !ok {
		t.Fatal("embed model should remain")
	}
}

func TestUnloadEntryIfCurrent(t *testing.T) {
	config.C = config.Config{UnloadModelsFromGPU: true}
	entry := &loadedEntry{}
	r := &Registry{
		loaded: map[string]*loadedEntry{
			"m:chat": entry,
		},
	}
	r.unloadEntryIfCurrent("m:chat", entry)
	if _, ok := r.loaded["m:chat"]; ok {
		t.Fatal("entry should be removed")
	}
	if entry.ll != nil {
		t.Fatal("ll should be nil after unload")
	}
}

func TestUnloadOthersDisabled(t *testing.T) {
	config.C = config.Config{UnloadModelsFromGPU: false}
	r := &Registry{
		loaded: map[string]*loadedEntry{
			"a:chat":         {},
			"b:embeddings": {},
		},
	}
	r.unloadOthers("b:embeddings")
	if len(r.loaded) != 2 {
		t.Fatalf("expected 2 models, got %d", len(r.loaded))
	}
}

func TestSplitLoadModeKey(t *testing.T) {
    alias, mode := splitLoadModeKey("qwen-27b:chat")
    if alias != "qwen-27b" || mode != "chat" {
        t.Fatalf("got %q, %q", alias, mode)
    }
    alias, mode = splitLoadModeKey("nomic-embed:embeddings")
    if alias != "nomic-embed" || mode != "embeddings" {
        t.Fatalf("got %q, %q", alias, mode)
    }
}

func TestCapabilities(t *testing.T) {
    r := &Registry{
        embeddingAliases: map[string]struct{}{"e1": {}},
    }
    caps := r.capabilities("e1")
    if len(caps) != 2 {
        t.Fatalf("expected chat+embeddings, got %v", caps)
    }
    caps = r.capabilities("c1")
    if len(caps) != 1 || caps[0] != "chat" {
        t.Fatalf("expected chat only, got %v", caps)
    }
}
