package rag

import (
    "context"
    "fmt"
    "log"
    "os"
    "path/filepath"
    "strings"

    llama "github.com/redirsvr/go-llama-new.cpp"

    "llama-cpp-gpt-api/internal/config"
    "llama-cpp-gpt-api/pkg/model"
    "llama-cpp-gpt-api/pkg/rag/chunking"
    "llama-cpp-gpt-api/pkg/rag/store"
)

// ingestFromPath индексирует файл с диска: малые — семантика в памяти, большие — потоковые чанки.
func (s *Service) ingestFromPath(ctx context.Context, path, title string, meta map[string]any) (*store.Document, error) {
    path = filepath.Clean(path)
    info, err := os.Stat(path)
    if err != nil {
        return nil, err
    }
    if info.IsDir() {
        return nil, fmt.Errorf("%q — каталог, а не файл", path)
    }

    textPath, cleanup, err := PrepareTextPath(path)
    if err != nil {
        return nil, err
    }
    defer cleanup()
    if textPath != path {
        log.Printf("RAG: PDF → текст %q", textPath)
        meta = cloneMeta(meta)
        meta["source_pdf"] = path
    }

    title = strings.TrimSpace(title)
    if title == "" {
        title = filepath.Base(path)
    }
    if meta == nil {
        meta = map[string]any{}
    }
    meta["staging_path"] = path
    meta["file_size"] = info.Size()

    textInfo, err := os.Stat(textPath)
    if err != nil {
        return nil, err
    }

    maxSemantic := int64(config.C.RAG.SemanticIngestMaxBytes)
    if maxSemantic <= 0 {
        maxSemantic = 512 * 1024
    }

    if textInfo.Size() <= maxSemantic {
        data, err := os.ReadFile(textPath)
        if err != nil {
            return nil, err
        }
        return s.ingest(ctx, IngestInput{
            Title:      title,
            Content:    string(data),
            SourcePath: path,
            Metadata:   meta,
        })
    }

    log.Printf("RAG: индексация с диска %q (%d байт текста)", title, textInfo.Size())
    return s.ingestLargeFile(ctx, textPath, path, title, meta)
}

func cloneMeta(m map[string]any) map[string]any {
    if m == nil {
        return map[string]any{}
    }
    out := make(map[string]any, len(m))
    for k, v := range m {
        out[k] = v
    }
    return out
}

func (s *Service) ingestLargeFile(ctx context.Context, textPath, sourcePath, title string, meta map[string]any) (*store.Document, error) {
    maxChars := MaxEmbedRunes()
    minChars := config.C.RAG.Chunking.MinChunkChars
    hash, fileChunks, err := chunking.ChunksFromFile(textPath, maxChars, minChars)
    if err != nil {
        return nil, err
    }

    existing, err := s.db.FindByHashPath(ctx, hash, sourcePath)
    if err != nil {
        return nil, err
    }
    if existing != nil {
        log.Printf("RAG: документ уже проиндексирован %q (%s)", title, sourcePath)
        return existing, nil
    }

    log.Printf("RAG: %q — %d чанков (потоковая нарезка)", title, len(fileChunks))
    chunkTexts := make([]string, len(fileChunks))
    chunkInputs := make([]store.ChunkInput, len(fileChunks))
    for i, c := range fileChunks {
        chunkTexts[i] = c.Content
        chunkInputs[i] = store.ChunkInput{
            Content:  c.Content,
            Metadata: store.MergeChunkMeta(i, sourcePath, meta, c.Meta),
        }
    }
    embeddings, err := s.embedChunks(chunkTexts)
    if err != nil {
        return nil, WrapIngestError(err)
    }
    chunkInputs, embeddings, err = alignChunkInputs(chunkInputs, embeddings)
    if err != nil {
        return nil, WrapIngestError(err)
    }

    id, err := s.db.InsertDocument(ctx, title, sourcePath, hash, meta, chunkInputs, embeddings)
    if err != nil {
        return nil, err
    }
    log.Printf("RAG: готово %q — %d чанков", title, len(chunkInputs))
    return s.db.GetDocument(ctx, id)
}

func (s *Service) embedChunks(chunks []string) ([][]float32, error) {
    modelAlias := model.DefaultEmbeddingModel()
    if modelAlias == "" {
        return nil, fmt.Errorf("не задана embedding-модель")
    }
    maxEmbed := MaxEmbedRunes()

    var embeddings [][]float32
    err := model.UseEmbeddings(modelAlias, func(ll *llama.LLama, _ string) error {
        out := make([][]float32, 0, len(chunks))
        for i, t := range chunks {
            t = PrepareText(strings.TrimSpace(t))
            if t == "" {
                out = append(out, nil)
                continue
            }
            t = TruncateRunes(t, maxEmbed)
            v, err := model.EmbedString(ll, PrefixDocument(t))
            if err != nil {
                return fmt.Errorf("чанк %d: %w", i, err)
            }
            out = append(out, v)
            if (i+1)%50 == 0 {
                log.Printf("RAG: эмбеддинг %d/%d чанков", i+1, len(chunks))
            }
        }
        embeddings = out
        return nil
    })
    return embeddings, err
}
