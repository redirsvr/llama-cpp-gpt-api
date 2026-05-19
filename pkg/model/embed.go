package model

import (
    "fmt"

    llama "github.com/redirsvr/go-llama-new.cpp"

    "llama-cpp-gpt-api/internal/config"
)

// EmbedTexts считает эмбеддинги для списка текстов одной embedding-моделью.
func EmbedTexts(modelAlias string, texts []string) ([][]float32, error) {
    if len(texts) == 0 {
        return nil, nil
    }
    var out [][]float32
    err := UseEmbeddings(modelAlias, func(ll *llama.LLama, _ string) error {
        out = make([][]float32, 0, len(texts))
        for i, t := range texts {
            v, err := EmbedString(ll, t)
            if err != nil {
                return fmt.Errorf("текст %d: %w", i, err)
            }
            out = append(out, v)
        }
        return nil
    })
    return out, err
}

// DefaultEmbeddingModel возвращает алиас embedding-модели из конфига.
func DefaultEmbeddingModel() string {
    if config.C.RAG.EmbeddingModel != "" {
        return config.C.RAG.EmbeddingModel
    }
    return config.C.DefaultEmbeddingModel
}
