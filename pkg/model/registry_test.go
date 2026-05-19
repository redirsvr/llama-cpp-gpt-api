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
