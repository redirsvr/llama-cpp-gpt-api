package rag

import (
    "context"
    "log"
    "strings"

    llama "github.com/redirsvr/go-llama-new.cpp"

    "llama-cpp-gpt-api/pkg/model"
)

// ReindexAllEmbeddings пересчитывает embedding всех чанков (после включения search_document:).
func ReindexAllEmbeddings(ctx context.Context) (int, error) {
    s, err := svc()
    if err != nil {
        return 0, err
    }
    chunks, err := s.db.ListChunksForReembed(ctx)
    if err != nil {
        return 0, err
    }
    if len(chunks) == 0 {
        return 0, nil
    }
    modelAlias := model.DefaultEmbeddingModel()
    var done int
    err = model.UseEmbeddings(modelAlias, func(ll *llama.LLama, _ string) error {
        for _, c := range chunks {
            text := PrepareText(strings.TrimSpace(c.Content))
            if text == "" {
                continue
            }
            text = TruncateRunes(text, MaxEmbedRunes())
            vec, err := model.EmbedString(ll, PrefixDocument(text))
            if err != nil {
                return err
            }
            if err := s.db.UpdateChunkEmbedding(ctx, c.ID, vec); err != nil {
                return err
            }
            done++
            if done%100 == 0 {
                log.Printf("RAG: переиндексация embedding %d/%d", done, len(chunks))
            }
        }
        return nil
    })
    log.Printf("RAG: переиндексация embedding завершена — %d чанков", done)
    return done, err
}
