package store

import (
    "context"

    "github.com/jackc/pgx/v5"
    "github.com/pgvector/pgvector-go"
)

// ChunkForReembed — чанк для пересчёта embedding.
type ChunkForReembed struct {
    ID      string
    Content string
}

// ListChunksForReembed возвращает все чанки с текстом.
func (p *Postgres) ListChunksForReembed(ctx context.Context) ([]ChunkForReembed, error) {
    rows, err := p.pool.Query(ctx, `
        SELECT id, content FROM rag_chunks ORDER BY document_id, chunk_index
    `)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    var out []ChunkForReembed
    for rows.Next() {
        var c ChunkForReembed
        if err := rows.Scan(&c.ID, &c.Content); err != nil {
            return nil, err
        }
        out = append(out, c)
    }
    return out, rows.Err()
}

// UpdateChunkEmbedding обновляет вектор чанка.
func (p *Postgres) UpdateChunkEmbedding(ctx context.Context, chunkID string, vec []float32) error {
    _, err := p.pool.Exec(ctx, `
        UPDATE rag_chunks SET embedding = $2 WHERE id = $1
    `, chunkID, pgvector.NewVector(vec))
    return err
}

// BestVectorScore — лучший cosine similarity для диагностики (без порога).
func (p *Postgres) BestVectorScore(ctx context.Context, queryVec []float32) (float64, error) {
    vec := pgvector.NewVector(queryVec)
    var score float64
    err := p.pool.QueryRow(ctx, `
        SELECT 1 - (embedding <=> $1)
        FROM rag_chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1
        LIMIT 1
    `, vec).Scan(&score)
    if err == pgx.ErrNoRows {
        return 0, nil
    }
    return score, err
}
