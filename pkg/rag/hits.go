package rag

import (
	"llama-cpp-gpt-api/internal/types"
	"llama-cpp-gpt-api/pkg/rag/store"
)

// HitTitles — заголовки документов из результатов поиска.
func HitTitles(hits []store.ChunkResult) []string {
	if len(hits) == 0 {
		return nil
	}
	out := make([]string, 0, len(hits))
	for _, h := range hits {
		out = append(out, h.Title)
	}
	return out
}

// HitFragments — краткие сведения о фрагментах для отладки в ответе chat.
func HitFragments(hits []store.ChunkResult) []types.RAGFragment {
	if len(hits) == 0 {
		return nil
	}
	out := make([]types.RAGFragment, 0, len(hits))
	for _, h := range hits {
		out = append(out, types.RAGFragment{
			Title:      h.Title,
			ChunkIndex: h.ChunkIndex,
			SourcePath: h.SourcePath,
			Score:      h.Score,
			Page:       metaPage(h.Metadata),
		})
	}
	return out
}
