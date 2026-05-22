package config

// Config — см. etc/gpt-api.yaml и etc/gpt-api-gpu.example.yaml.
//
// ModelOption (GPU offload): NGPULayers, AutoGPU, MainGPU, TensorSplit.
// AutoGPU или NGPULayers: -1 — fit_params подбирает слои и распределение по GPU.
// Embeddings в ModelOption не используется — только EmbeddingModels.
type Config struct {
    Name    string    `yaml:"Name"`
    Host    string    `yaml:"Host"`
    Port    int       `yaml:"Port"`
    Timeout int64     `yaml:"Timeout"` // миллисекунды
    Log     LogConfig `yaml:"Log"`

    ModelOption           map[string]interface{} `yaml:"ModelOption"`
    EmbeddingModelOption  map[string]interface{} `yaml:"EmbeddingModelOption"` // иначе — ModelOption с ограничениями для embed
    ModelsDir             string                 `yaml:"ModelsDir"`
    DefaultModel          string                 `yaml:"DefaultModel"`
    EmbeddingModels       []string               `yaml:"EmbeddingModels"`
    DefaultEmbeddingModel string                 `yaml:"DefaultEmbeddingModel"`
    ModelPath             string                 `yaml:"ModelPath"`
    UserName              string                 `yaml:"UserName"`
    SystemPrompt          string                 `yaml:"SystemPrompt"`
    ChatTemplate          string                 `yaml:"ChatTemplate"`
    DefaultMaxTokens      int                    `yaml:"DefaultMaxTokens"`
    PreloadDefaultModel   bool                   `yaml:"PreloadDefaultModel"`
    DisableThinking       bool                   `yaml:"DisableThinking"`
    RAG                   RAGConfig              `yaml:"RAG"`
}

// RAGConfig — Hybrid RAG (pgvector + полнотекстовый поиск).
type RAGConfig struct {
    Enabled             bool              `yaml:"Enabled"`
    UIDisabled          bool              `yaml:"UIDisabled"` // true — не раздавать /ui/
    Postgres            PostgresConfig    `yaml:"Postgres"`
    EmbeddingModel      string            `yaml:"EmbeddingModel"`
    EmbeddingDimensions int               `yaml:"EmbeddingDimensions"`
    Chunking            ChunkingConfig    `yaml:"Chunking"`
    Hybrid              HybridConfig      `yaml:"Hybrid"`
    AutoIngest          AutoIngestConfig  `yaml:"AutoIngest"`
    IngestTimeoutSec           int    `yaml:"IngestTimeoutSec"`           // HTTP WriteTimeout для длинной индексации
    EmbeddingContextSize       int    `yaml:"EmbeddingContextSize"`       // n_ctx для embedding-модели (nomic: 2048)
    StagingDir                 string `yaml:"StagingDir"`                 // каталог для upload перед индексацией
    RemoveStagingAfterIngest   bool   `yaml:"RemoveStagingAfterIngest"`   // удалять файл после успешного ingest
    SemanticIngestMaxBytes     int    `yaml:"SemanticIngestMaxBytes"`     // выше — только чанкинг по размеру с диска
    PDFExtract                 string `yaml:"PDFExtract"`                 // auto | builtin | pdftotext — извлечение текста из PDF
    UseInChatCompletions       *bool  `yaml:"UseInChatCompletions"`       // nil → true при Enabled; подмешивать RAG в /v1/chat/completions
    EmbedQueryPrefix           string `yaml:"EmbedQueryPrefix"`           // nomic: search_query:
    EmbedDocumentPrefix        string `yaml:"EmbedDocumentPrefix"`        // nomic: search_document:
}

// ChatRAGEnabled — подмешивать ли контекст RAG в chat/completions по умолчанию.
func (r RAGConfig) ChatRAGEnabled() bool {
    if r.UseInChatCompletions != nil {
        return *r.UseInChatCompletions
    }
    return r.Enabled
}

type PostgresConfig struct {
    Host     string `yaml:"Host"`
    Port     int    `yaml:"Port"`
    Database string `yaml:"Database"`
    User     string `yaml:"User"`
    Password string `yaml:"Password"`
    SSLMode  string `yaml:"SSLMode"`
}

type ChunkingConfig struct {
    MaxChunkChars          int     `yaml:"MaxChunkChars"`
    MinChunkChars          int     `yaml:"MinChunkChars"`
    OverlapChars           int     `yaml:"OverlapChars"` // рун перекрытия соседних чанков
    BreakpointPercentile   float64 `yaml:"BreakpointPercentile"`
    MaxSentencesSemantic   int     `yaml:"MaxSentencesSemantic"` // выше — чанкинг по размеру без embed каждого предложения
    MaxEmbedRunes          int     `yaml:"MaxEmbedRunes"`          // лимит рун на один вызов Embeddings (токены ≤ n_ubatch)
}

type HybridConfig struct {
    TopK            int     `yaml:"TopK"`
    VectorWeight    float64 `yaml:"VectorWeight"`
    TextWeight      float64 `yaml:"TextWeight"`
    MinVectorScore  float64 `yaml:"MinVectorScore"` // отсечь слабые векторные совпадения (cosine similarity)
}

type AutoIngestConfig struct {
    Enabled     bool     `yaml:"Enabled"`
    Directories []string `yaml:"Directories"`
    Extensions  []string `yaml:"Extensions"`
    ScanOnStart bool     `yaml:"ScanOnStart"`
    IntervalSec int      `yaml:"IntervalSec"`
}

// LogConfig сохраняется для совместимости YAML; логирование — стандартный log.
type LogConfig struct {
    Mode     string `yaml:"Mode"`
    Level    string `yaml:"Level"`
    KeepDays int    `yaml:"KeepDays"`
}

var C Config
