package rag

import (
    "strings"

    "llama-cpp-gpt-api/internal/config"
)

// PrefixQuery добавляет префикс embedding-модели к поисковому запросу (nomic: search_query:).
func PrefixQuery(q string) string {
    q = strings.TrimSpace(q)
    p := strings.TrimSpace(config.C.RAG.EmbedQueryPrefix)
    if p == "" || strings.HasPrefix(q, p) {
        return q
    }
    return p + q
}

// PrefixDocument добавляет префикс к тексту чанка при индексации (nomic: search_document:).
func PrefixDocument(text string) string {
    text = strings.TrimSpace(text)
    p := strings.TrimSpace(config.C.RAG.EmbedDocumentPrefix)
    if p == "" || strings.HasPrefix(text, p) {
        return text
    }
    return p + text
}
