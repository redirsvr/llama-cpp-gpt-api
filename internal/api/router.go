package api

import (
    "github.com/gin-gonic/gin"
    "github.com/prometheus/client_golang/prometheus/promhttp"

    "llama-cpp-gpt-api/pkg/metrics"
)

// NewRouter создаёт HTTP-маршрутизатор Gin с OpenAI-совместимыми эндпоинтами.
func NewRouter() *gin.Engine {
    gin.SetMode(gin.ReleaseMode)
    r := gin.New()
    r.Use(gin.Recovery())
    // Крупные upload в /v1/rag/documents/upload (иначе multipart >32MB падает).
    r.MaxMultipartMemory = 256 << 20

    r.GET("/metrics", gin.WrapH(promhttp.Handler()))
    registerDocs(r)
    registerUI(r)

    v1 := r.Group("/v1")
    {
        v1.POST("/chat/completions",
            metrics.GinMiddleware("/v1/chat/completions"),
            ChatCompletions,
        )
        v1.GET("/embeddings",
            metrics.GinMiddleware("/v1/embeddings"),
            Embeddings,
        )
        v1.POST("/embeddings",
            metrics.GinMiddleware("/v1/embeddings"),
            Embeddings,
        )
        v1.GET("/models",
            metrics.GinMiddleware("/v1/models"),
            ListModels,
        )
        v1.GET("/models/:model",
            metrics.GinMiddleware("/v1/models/:model"),
            RetrieveModel,
        )

        rag := v1.Group("/rag")
        {
            rag.POST("/documents",
                metrics.GinMiddleware("/v1/rag/documents"),
                RAGIngestDocument,
            )
            rag.POST("/documents/upload",
                metrics.GinMiddleware("/v1/rag/documents/upload"),
                RAGIngestUpload,
            )
            rag.POST("/documents/scan",
                metrics.GinMiddleware("/v1/rag/documents/scan"),
                RAGIngestScan,
            )
            rag.GET("/documents",
                metrics.GinMiddleware("/v1/rag/documents"),
                RAGListDocuments,
            )
            rag.GET("/documents/:id",
                metrics.GinMiddleware("/v1/rag/documents/:id"),
                RAGGetDocument,
            )
            rag.DELETE("/documents/:id",
                metrics.GinMiddleware("/v1/rag/documents/:id"),
                RAGDeleteDocument,
            )
            rag.POST("/query",
                metrics.GinMiddleware("/v1/rag/query"),
                RAGQuery,
            )
            rag.POST("/reindex-embeddings",
                metrics.GinMiddleware("/v1/rag/reindex-embeddings"),
                RAGReindexEmbeddings,
            )
        }
    }

    return r
}
