package store

import (
	"context"
	"fmt"
	"log"
	"strings"
)

// migrateSearchSchema обновляет tsv для регистронезависимого full-text поиска.
func (p *Postgres) migrateSearchSchema(ctx context.Context) error {
	var expr string
	err := p.pool.QueryRow(ctx, `
		SELECT COALESCE(pg_get_expr(ad.adbin, ad.adrelid), '')
		FROM pg_attribute a
		JOIN pg_class c ON c.oid = a.attrelid
		LEFT JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
		WHERE c.relname = 'rag_chunks' AND a.attname = 'tsv'
	`).Scan(&expr)
	if err != nil {
		return fmt.Errorf("проверка tsv: %w", err)
	}
	if strings.Contains(strings.ToLower(expr), "lower(content)") {
		return nil
	}

	log.Println("RAG: миграция tsv → lower(content) для регистронезависимого поиска")
	if _, err := p.pool.Exec(ctx, `DROP INDEX IF EXISTS idx_rag_chunks_tsv`); err != nil {
		return fmt.Errorf("drop idx_rag_chunks_tsv: %w", err)
	}
	if _, err := p.pool.Exec(ctx, `ALTER TABLE rag_chunks DROP COLUMN IF EXISTS tsv`); err != nil {
		return fmt.Errorf("drop rag_chunks.tsv: %w", err)
	}
	if _, err := p.pool.Exec(ctx, `
		ALTER TABLE rag_chunks
		ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', lower(content))) STORED
	`); err != nil {
		return fmt.Errorf("add rag_chunks.tsv: %w", err)
	}
	if _, err := p.pool.Exec(ctx, `CREATE INDEX IF NOT EXISTS idx_rag_chunks_tsv ON rag_chunks USING gin(tsv)`); err != nil {
		return fmt.Errorf("create idx_rag_chunks_tsv: %w", err)
	}
	return nil
}

// migrateChunkMetadata добавляет колонку metadata на существующих БД.
func (p *Postgres) migrateChunkMetadata(ctx context.Context) error {
	var exists bool
	err := p.pool.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.columns
			WHERE table_name = 'rag_chunks' AND column_name = 'metadata'
		)
	`).Scan(&exists)
	if err != nil {
		return fmt.Errorf("проверка rag_chunks.metadata: %w", err)
	}
	if exists {
		return nil
	}
	log.Println("RAG: миграция rag_chunks.metadata (JSONB)")
	if _, err := p.pool.Exec(ctx, `
		ALTER TABLE rag_chunks ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'
	`); err != nil {
		return fmt.Errorf("add rag_chunks.metadata: %w", err)
	}
	return nil
}

// migrateEmbeddingSchema выравнивает vector(N) в пустой БД под RAG.EmbeddingDimensions.
func (p *Postgres) migrateEmbeddingSchema(ctx context.Context) error {
	var chunkCount int64
	if err := p.pool.QueryRow(ctx, `SELECT COUNT(*) FROM rag_chunks`).Scan(&chunkCount); err != nil {
		return fmt.Errorf("проверка rag_chunks: %w", err)
	}
	if chunkCount == 0 {
		if err := p.setEmbeddingColumnDim(ctx); err != nil {
			log.Printf("RAG: миграция embedding (пустая БД): %v", err)
		} else {
			log.Printf("RAG: колонка embedding → vector(%d) (таблица пуста)", p.dim)
		}
		return nil
	}

	var existingDim *int
	_ = p.pool.QueryRow(ctx, `
        SELECT vector_dims(embedding) FROM rag_chunks
        WHERE embedding IS NOT NULL LIMIT 1
    `).Scan(&existingDim)
	if existingDim != nil && *existingDim != p.dim {
		return fmt.Errorf(
			"в БД уже есть эмбеддинги размерности %d, в конфиге %d; очистите RAG-таблицы или смените EmbeddingDimensions",
			*existingDim, p.dim,
		)
	}
	return nil
}

func (p *Postgres) setEmbeddingColumnDim(ctx context.Context) error {
	_, _ = p.pool.Exec(ctx, `DROP INDEX IF EXISTS idx_rag_chunks_embedding_hnsw`)
	_, err := p.pool.Exec(ctx, fmt.Sprintf(
		`ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector(%d)`,
		p.dim,
	))
	return err
}
