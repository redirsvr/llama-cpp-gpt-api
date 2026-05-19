package chunking

import (
    "fmt"
    "math"
    "sort"
)

// BatchEmbedFunc возвращает векторы для списка фрагментов (один вызов — один проход модели).
type BatchEmbedFunc func(texts []string) ([][]float32, error)

// Config параметры семантического чанкинга.
type Config struct {
    MaxChunkChars        int
    MinChunkChars        int
    BreakpointPercentile float64 // 0..1 — границы там, где соседние предложения наименее похожи
    MaxSentencesSemantic int     // при большем числе предложений — только чанкинг по размеру
}

func (c Config) withDefaults() Config {
    if c.MaxChunkChars <= 0 {
        c.MaxChunkChars = 1500
    }
    if c.MinChunkChars <= 0 {
        c.MinChunkChars = 80
    }
    if c.BreakpointPercentile <= 0 || c.BreakpointPercentile > 1 {
        c.BreakpointPercentile = 0.25
    }
    if c.MaxSentencesSemantic <= 0 {
        c.MaxSentencesSemantic = 64
    }
    return c
}

// SemanticChunks режет текст по семантическим границам (падение похожести соседних предложений).
// Все предложения эмбеддятся одним батчем; при слишком длинном тексте — быстрый режим без семантики.
func SemanticChunks(text string, embed BatchEmbedFunc, cfg Config) ([]string, error) {
    cfg = cfg.withDefaults()
    sentences := SplitSentences(text)
    if len(sentences) <= 1 {
        return MergeByCharLimit(sentences, cfg.MaxChunkChars, cfg.MinChunkChars), nil
    }
    if len(sentences) > cfg.MaxSentencesSemantic {
        return MergeByCharLimit(sentences, cfg.MaxChunkChars, cfg.MinChunkChars), nil
    }

    vectors, err := embed(sentences)
    if err != nil {
        return nil, fmt.Errorf("embedding предложений: %w", err)
    }
    if len(vectors) != len(sentences) {
        return nil, fmt.Errorf("embedding: ожидалось %d векторов, получено %d", len(sentences), len(vectors))
    }

    sims := make([]float64, len(sentences)-1)
    for i := 0; i < len(sentences)-1; i++ {
        sims[i] = cosineSimilarity(vectors[i], vectors[i+1])
    }

    threshold := percentile(sims, cfg.BreakpointPercentile)

    var groups [][]string
    group := []string{sentences[0]}
    for i := 0; i < len(sims); i++ {
        if sims[i] < threshold {
            groups = append(groups, group)
            group = []string{sentences[i+1]}
        } else {
            group = append(group, sentences[i+1])
        }
    }
    groups = append(groups, group)

    var merged []string
    for _, g := range groups {
        merged = append(merged, MergeByCharLimit(g, cfg.MaxChunkChars, cfg.MinChunkChars)...)
    }
    if len(merged) == 0 {
        return MergeByCharLimit(sentences, cfg.MaxChunkChars, cfg.MinChunkChars), nil
    }
    return merged, nil
}

func cosineSimilarity(a, b []float32) float64 {
    if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
        return 0
    }
    var dot, na, nb float64
    for i := range a {
        fa := float64(a[i])
        fb := float64(b[i])
        dot += fa * fb
        na += fa * fa
        nb += fb * fb
    }
    if na == 0 || nb == 0 {
        return 0
    }
    return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func percentile(values []float64, p float64) float64 {
    if len(values) == 0 {
        return 0
    }
    cp := append([]float64(nil), values...)
    sort.Float64s(cp)
    idx := int(math.Floor(p * float64(len(cp)-1)))
    if idx < 0 {
        idx = 0
    }
    if idx >= len(cp) {
        idx = len(cp) - 1
    }
    return cp[idx]
}
