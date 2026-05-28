package config

import (
	"fmt"
	"os"
	"path/filepath"
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
	if err := loadModelPresets(path); err != nil {
		return err
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
	if cfg.RAG.Chunking.OverlapChars == 0 {
		cfg.RAG.Chunking.OverlapChars = cfg.RAG.Chunking.MaxChunkChars / 7
		if cfg.RAG.Chunking.OverlapChars < 80 {
			cfg.RAG.Chunking.OverlapChars = 80
		}
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
	if cfg.RAG.KeywordExtraction.MaxKeywords == 0 {
		cfg.RAG.KeywordExtraction.MaxKeywords = 10
	}
	if cfg.RAG.Enabled && cfg.RAG.UseInChatCompletions == nil {
		on := true
		cfg.RAG.UseInChatCompletions = &on
	}
	applyNomicEmbedPrefixes(&cfg)
	C = cfg
	return nil
}

func loadModelPresets(configPath string) error {
	presetPath := filepath.Join(filepath.Dir(configPath), "models-preset.yaml")
	data, err := os.ReadFile(presetPath)
	if err != nil {
		if os.IsNotExist(err) {
			ModelPresets = nil
			return nil
		}
		return fmt.Errorf("чтение пресетов моделей %q: %w", presetPath, err)
	}

	var root map[interface{}]interface{}
	if err := yaml.Unmarshal(data, &root); err != nil {
		return fmt.Errorf("разбор пресетов моделей %q: %w", presetPath, err)
	}
	rawModels := rootValue(root, "Models")
	if rawModels == nil {
		rawModels = rootValue(root, "ModelPresets")
	}
	presets := parseModelPresets(rawModels)
	if len(presets) == 0 {
		ModelPresets = nil
		return nil
	}
	ModelPresets = presets
	return nil
}

func rootValue(root map[interface{}]interface{}, key string) interface{} {
	for k, v := range root {
		if ks, ok := k.(string); ok && strings.EqualFold(strings.TrimSpace(ks), key) {
			return v
		}
	}
	return nil
}

func parseModelPresets(raw interface{}) map[string]ModelPreset {
	out := map[string]ModelPreset{}
	switch v := raw.(type) {
	case []interface{}:
		for _, item := range v {
			preset := normalizeRawModelPreset(toStringMap(item))
			key := presetKey(preset, "")
			if key != "" {
				out[key] = preset
			}
		}
	case map[interface{}]interface{}:
		for k, item := range v {
			keyHint, _ := k.(string)
			preset := normalizeRawModelPreset(toStringMap(item))
			if preset.File == "" && strings.TrimSpace(keyHint) != "" {
				preset.File = strings.TrimSpace(keyHint)
			}
			key := presetKey(preset, keyHint)
			if key != "" {
				out[key] = preset
			}
		}
	}
	return out
}

func presetKey(preset ModelPreset, fallback string) string {
	for _, s := range []string{preset.Alias, preset.File, fallback} {
		if key := strings.TrimSpace(s); key != "" {
			return key
		}
	}
	return ""
}

func normalizeRawModelPreset(raw map[string]interface{}) ModelPreset {
	preset := ModelPreset{}
	for k, v := range raw {
		switch strings.ToLower(strings.TrimSpace(k)) {
		case "file":
			if s, ok := v.(string); ok {
				preset.File = strings.TrimSpace(s)
			}
		case "alias":
			if s, ok := v.(string); ok {
				preset.Alias = strings.TrimSpace(s)
			}
		case "modeloption":
			preset.ModelOption = toStringMap(v)
		case "embeddingmodeloption":
			preset.EmbeddingModelOption = toStringMap(v)
		}
	}
	if len(preset.ModelOption) == 0 && len(preset.EmbeddingModelOption) == 0 {
		preset.ModelOption = mapWithoutPresetMeta(raw)
	}
	return preset
}

func mapWithoutPresetMeta(raw map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(raw))
	for k, v := range raw {
		switch strings.ToLower(strings.TrimSpace(k)) {
		case "file", "alias":
			continue
		default:
			out[k] = v
		}
	}
	return out
}

func toStringMap(v interface{}) map[string]interface{} {
	switch m := v.(type) {
	case map[string]interface{}:
		return m
	case map[interface{}]interface{}:
		out := make(map[string]interface{}, len(m))
		for k, v := range m {
			ks, ok := k.(string)
			if !ok {
				continue
			}
			out[ks] = v
		}
		return out
	default:
		return nil
	}
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
