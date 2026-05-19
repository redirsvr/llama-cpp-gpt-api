package model

import (
    "fmt"

    llama "github.com/redirsvr/go-llama-new.cpp"

    "llama-cpp-gpt-api/internal/config"
)

// EmbeddingDimensions — размерность вектора для RAG/pgvector (nomic-embed: 768).
func EmbeddingDimensions() int {
    d := config.C.RAG.EmbeddingDimensions
    if d <= 0 {
        return 768
    }
    return d
}

// EmbedString считает эмбеддинг текста и нормализует размерность под RAG/pgvector.
func EmbedString(ll *llama.LLama, text string) ([]float32, error) {
    want := EmbeddingDimensions()
    // SetTokens задаёт размер буфера в binding (старые сборки без llama_binding_n_embd).
    v, err := ll.Embeddings(text, llama.SetTokens(want))
    if err != nil {
        return nil, err
    }
    return normalizeEmbedding(v, want)
}

func normalizeEmbedding(v []float32, want int) ([]float32, error) {
    switch {
    case len(v) == want:
        return v, nil
    case len(v) > want:
        return v[:want], nil
    default:
        return nil, fmt.Errorf(
            "размерность эмбеддинга %d, ожидалось %d (пересоберите binding: cd go-llama-new.cpp && make cuda)",
            len(v), want,
        )
    }
}
