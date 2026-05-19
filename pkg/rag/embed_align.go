package rag

import (
	"fmt"

	"llama-cpp-gpt-api/pkg/rag/store"
)

// alignChunksEmbeddings отбрасывает чанки без вектора (пустой текст после PrepareText).
func alignChunksEmbeddings(chunks []string, embeddings [][]float32) ([]string, [][]float32, error) {
	if len(chunks) != len(embeddings) {
		return nil, nil, fmt.Errorf("число чанков (%d) != числу эмбеддингов (%d)", len(chunks), len(embeddings))
	}
	outC := make([]string, 0, len(chunks))
	outE := make([][]float32, 0, len(chunks))
	for i := range chunks {
		if len(embeddings[i]) == 0 {
			continue
		}
		outC = append(outC, chunks[i])
		outE = append(outE, embeddings[i])
	}
	if len(outC) == 0 {
		return nil, nil, fmt.Errorf("нет чанков с эмбеддингами (весь текст пустой или модель не вернула вектор)")
	}
	return outC, outE, nil
}

// alignChunkInputs отбрасывает чанки без вектора.
func alignChunkInputs(chunks []store.ChunkInput, embeddings [][]float32) ([]store.ChunkInput, [][]float32, error) {
	if len(chunks) != len(embeddings) {
		return nil, nil, fmt.Errorf("число чанков (%d) != числу эмбеддингов (%d)", len(chunks), len(embeddings))
	}
	outC := make([]store.ChunkInput, 0, len(chunks))
	outE := make([][]float32, 0, len(chunks))
	for i := range chunks {
		if len(embeddings[i]) == 0 {
			continue
		}
		outC = append(outC, chunks[i])
		outE = append(outE, embeddings[i])
	}
	if len(outC) == 0 {
		return nil, nil, fmt.Errorf("нет чанков с эмбеддингами (весь текст пустой или модель не вернула вектор)")
	}
	return outC, outE, nil
}
