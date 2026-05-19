package rag

import (
    "context"
    "time"

    "llama-cpp-gpt-api/internal/config"
)

// IngestContext — таймаут индексации, не привязан к HTTP-запросу (чтобы не обрывать долгий ingest).
func IngestContext() (context.Context, context.CancelFunc) {
    sec := config.C.RAG.IngestTimeoutSec
    if sec <= 0 {
        sec = 900
    }
    return context.WithTimeout(context.Background(), time.Duration(sec)*time.Second)
}
