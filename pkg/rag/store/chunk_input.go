package store

// ChunkInput — чанк при записи в БД.
type ChunkInput struct {
	Content  string
	Metadata map[string]any
}

// MergeChunkMeta объединяет метаданные документа и чанка (индекс, путь, страница).
func MergeChunkMeta(chunkIndex int, sourcePath string, docMeta, chunkMeta map[string]any) map[string]any {
	out := map[string]any{"chunk_index": chunkIndex}
	if sourcePath != "" {
		out["source_path"] = sourcePath
	}
	if docMeta != nil {
		if v, ok := docMeta["source_pdf"]; ok {
			out["source_pdf"] = v
		}
	}
	for k, v := range chunkMeta {
		out[k] = v
	}
	out["chunk_index"] = chunkIndex
	return out
}
