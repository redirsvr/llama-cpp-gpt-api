package types

// OpenAIErrorBody — тело ошибки в формате OpenAI API.
type OpenAIErrorBody struct {
	Error OpenAIErrorDetail `json:"error"`
}

// OpenAIErrorDetail — поле error.
type OpenAIErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    string  `json:"code"`
}

// ResRAGScan — ответ POST /v1/rag/documents/scan.
type ResRAGScan struct {
	Ingested int `json:"ingested"`
}

// ResRAGReindex — ответ POST /v1/rag/reindex-embeddings.
type ResRAGReindex struct {
	ReindexedChunks int `json:"reindexed_chunks"`
}
