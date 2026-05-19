package types

type Choice struct {
    Index        int      `json:"index"`
    Message      Message  `json:"message,omitempty"`
    Delta        *Message `json:"delta,omitempty"`
    FinishReason string   `json:"finish_reason,omitempty"`
}

type Embedding struct {
    Embedding []float32 `json:"embedding" swaggertype:"array,number"`
    Index     int       `json:"index"`
}

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type ReqChatCompletion struct {
    Model       string    `json:"model,omitempty"`
    Messages    []Message `json:"messages"`
    UseRAG      *bool     `json:"use_rag,omitempty"` // nil → из конфига RAG.UseInChatCompletions (по умолчанию true)
    RAGTopK     int       `json:"rag_top_k,omitempty"`
    Stream      bool      `json:"stream,omitempty"`
    MaxTokens   int       `json:"max_tokens,omitempty"`
    Temperature float32   `json:"temperature,omitempty"`
    TopP        float32   `json:"top_p,omitempty"`
    User        string    `json:"user,omitempty"`
    Seed        int       `json:"seed,omitempty"`
    Prompt      string    `json:"prompt,omitempty"`
    RAGUsed      int           `json:"-"`
    RAGTitles    []string      `json:"-"`
    RAGFragments []RAGFragment `json:"-"`
}

type ReqEmbeddings struct {
    Model string `json:"model,omitempty" form:"model"`
    Input string `json:"input" form:"input" binding:"required"`
    User  string `json:"user,omitempty" form:"user"`
}

type ModelObject struct {
    ID           string   `json:"id"`
    Object       string   `json:"object"`
    Created      int64    `json:"created"`
    OwnedBy      string   `json:"owned_by"`
    Capabilities []string `json:"capabilities,omitempty"`
}

type ResModelsList struct {
    Object string        `json:"object"`
    Data   []ModelObject `json:"data"`
}

type ResChatCompletion struct {
    ID      string   `json:"id"`
    Choices []Choice `json:"choices"`
    Created int      `json:"created"`
    Model   string   `json:"model"`
    Usage   Usage    `json:"usage"`
    RAG     *RAGInfo `json:"rag,omitempty"`
}

// RAGFragment — один фрагмент из базы в ответе chat.
type RAGFragment struct {
    Title      string  `json:"title"`
    ChunkIndex int     `json:"chunk_index,omitempty"`
    SourcePath string  `json:"source_path,omitempty"`
    Page       int     `json:"page,omitempty"`
    Score      float64 `json:"score,omitempty"`
}

// RAGInfo — отладка: сколько фрагментов из базы ушло в промпт.
type RAGInfo struct {
    ChunksUsed int           `json:"chunks_used"`
    Titles     []string      `json:"titles,omitempty"`
    Fragments  []RAGFragment `json:"fragments,omitempty"`
}

type ResEmbeddings struct {
    Data  []Embedding `json:"data"`
    Model string      `json:"model"`
}

type Usage struct {
    CompletionToken int `json:"completion_tokens"`
    PromptTokens    int `json:"prompt_tokens"`
    TotalTokens     int `json:"total_tokens"`
}
