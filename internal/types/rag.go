package types

import "llama-cpp-gpt-api/pkg/rag/store"

type ReqRAGIngest struct {
    Title      string         `json:"title,omitempty"`
    Content    string         `json:"content,omitempty"`
    Path       string         `json:"path,omitempty"`
    Metadata   map[string]any `json:"metadata,omitempty" swaggertype:"object"`
}

type ResRAGIngest struct {
    Document store.Document `json:"document"`
}

type ReqRAGQuery struct {
    Query string `json:"query" binding:"required"`
    TopK  int    `json:"top_k,omitempty"`
}

type ResRAGQuery struct {
    Query   string              `json:"query"`
    Results []store.ChunkResult `json:"results"`
}

type ResRAGDocuments struct {
    Data []store.Document `json:"data"`
}
