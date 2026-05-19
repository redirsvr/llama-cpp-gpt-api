package config

import (
    "fmt"
    "os"
    "strings"

    "gopkg.in/yaml.v2"
)

// Load читает конфигурацию из YAML-файла.
func Load(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return fmt.Errorf("чтение конфига %q: %w", path, err)
    }
    var cfg Config
    if err := yaml.Unmarshal(data, &cfg); err != nil {
        return fmt.Errorf("разбор конфига %q: %w", path, err)
    }
    if cfg.Host == "" {
        cfg.Host = "0.0.0.0"
    }
    if cfg.Port == 0 {
        cfg.Port = 8080
    }
    if cfg.RAG.Postgres.Port == 0 {
        cfg.RAG.Postgres.Port = 5432
    }
    if cfg.RAG.Postgres.SSLMode == "" {
        cfg.RAG.Postgres.SSLMode = "disable"
    }
    if cfg.RAG.Hybrid.TopK == 0 {
        cfg.RAG.Hybrid.TopK = 8
    }
    if cfg.RAG.Chunking.MaxChunkChars == 0 {
        cfg.RAG.Chunking.MaxChunkChars = 1500
    }
    if cfg.RAG.Chunking.MinChunkChars == 0 {
        cfg.RAG.Chunking.MinChunkChars = 80
    }
    if cfg.RAG.Chunking.MaxSentencesSemantic == 0 {
        cfg.RAG.Chunking.MaxSentencesSemantic = 64
    }
    if cfg.RAG.Chunking.MaxEmbedRunes == 0 {
        cfg.RAG.Chunking.MaxEmbedRunes = 512
    }
    if cfg.RAG.IngestTimeoutSec == 0 {
        cfg.RAG.IngestTimeoutSec = 900
    }
    if cfg.RAG.EmbeddingContextSize == 0 {
        cfg.RAG.EmbeddingContextSize = 2048
    }
    if cfg.RAG.SemanticIngestMaxBytes == 0 {
        cfg.RAG.SemanticIngestMaxBytes = 512 * 1024
    }
    if strings.TrimSpace(cfg.RAG.PDFExtract) == "" {
        cfg.RAG.PDFExtract = "builtin"
    }
    if cfg.RAG.Enabled && cfg.RAG.UseInChatCompletions == nil {
        on := true
        cfg.RAG.UseInChatCompletions = &on
    }
    applyNomicEmbedPrefixes(&cfg)
    C = cfg
    return nil
}

func applyNomicEmbedPrefixes(cfg *Config) {
    if !cfg.RAG.Enabled {
        return
    }
    em := strings.ToLower(cfg.RAG.EmbeddingModel + " " + cfg.DefaultEmbeddingModel)
    if !strings.Contains(em, "nomic") {
        return
    }
    if strings.TrimSpace(cfg.RAG.EmbedQueryPrefix) == "" {
        cfg.RAG.EmbedQueryPrefix = "search_query: "
    }
    if strings.TrimSpace(cfg.RAG.EmbedDocumentPrefix) == "" {
        cfg.RAG.EmbedDocumentPrefix = "search_document: "
    }
}
