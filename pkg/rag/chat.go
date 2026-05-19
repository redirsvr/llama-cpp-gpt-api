package rag

import (
    "llama-cpp-gpt-api/internal/config"
)

// UseInChatCompletions — нужен ли RAG для этого запроса chat/completions.
func UseInChatCompletions(requestUseRAG *bool) bool {
    if !Enabled() {
        return false
    }
    if requestUseRAG != nil {
        return *requestUseRAG
    }
    return config.C.RAG.ChatRAGEnabled()
}
