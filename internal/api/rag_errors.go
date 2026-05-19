package api

import (
    "context"
    "errors"
    "log"
    "net/http"

    "llama-cpp-gpt-api/pkg/rag"

    "github.com/gin-gonic/gin"
)

func respondRAGIngestError(c *gin.Context, err error) {
    if err == nil {
        return
    }
    log.Printf("RAG ingest: %v", err)

    switch {
    case errors.Is(err, rag.ErrPDFExtract), errors.Is(err, rag.ErrEmptyDocument):
        c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
    case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
        c.JSON(http.StatusGatewayTimeout, OpenAIError(err, "timeout"))
    default:
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "ingest_failed"))
    }
}
