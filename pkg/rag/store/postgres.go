package store

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"

	"llama-cpp-gpt-api/internal/config"
)

//go:embed schema.sql
var schemaFS embed.FS

// Postgres — хранилище чанков и документов RAG.
type Postgres struct {
	pool  *pgxpool.Pool
	dim   int
	topK  int
	vecW  float64
	textW float64
}

// Open подключается к PostgreSQL и применяет схему.
func Open(ctx context.Context) (*Postgres, error) {
	pg := config.C.RAG.Postgres
	if pg.Database == "" {
		return nil, fmt.Errorf("RAG.Postgres.Database не задан")
	}
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		pg.Host, pg.Port, pg.User, pg.Password, pg.Database, pg.SSLMode,
	)
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("postgres: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres ping: %w", err)
	}
	sql, err := schemaFS.ReadFile("schema.sql")
	if err != nil {
		pool.Close()
		return nil, err
	}
	if _, err := pool.Exec(ctx, string(sql)); err != nil {
		pool.Close()
		return nil, fmt.Errorf("миграция RAG: %w", err)
	}
	dim := config.C.RAG.EmbeddingDimensions
	if dim <= 0 {
		dim = 768
	}
	h := config.C.RAG.Hybrid
	topK := h.TopK
	if topK <= 0 {
		topK = 8
	}
	vecW, textW := h.VectorWeight, h.TextWeight
	if vecW <= 0 && textW <= 0 {
		vecW, textW = 0.6, 0.4
	}
	s := vecW + textW
	vecW, textW = vecW/s, textW/s

	p := &Postgres{pool: pool, dim: dim, topK: topK, vecW: vecW, textW: textW}
	if err := p.migrateSearchSchema(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	if err := p.migrateChunkMetadata(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	if err := p.migrateEmbeddingSchema(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return p, nil
}

func (p *Postgres) Close() {
	if p.pool != nil {
		p.pool.Close()
	}
}

// ChunkStats — всего чанков и с заполненным embedding.
func (p *Postgres) ChunkStats(ctx context.Context) (total, withEmbedding int64, err error) {
	err = p.pool.QueryRow(ctx, `SELECT COUNT(*) FROM rag_chunks`).Scan(&total)
	if err != nil {
		return 0, 0, err
	}
	err = p.pool.QueryRow(ctx, `SELECT COUNT(*) FROM rag_chunks WHERE embedding IS NOT NULL`).Scan(&withEmbedding)
	return total, withEmbedding, err
}

func (p *Postgres) ensureVectorIndex(ctx context.Context) error {
	// HNSW после первой вставки с известной размерностью
	_, err := p.pool.Exec(ctx, fmt.Sprintf(`
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
        ON rag_chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    `))
	return err
}

// Document метаданные документа в БД.
type Document struct {
	ID          string         `json:"id"`
	Title       string         `json:"title"`
	SourcePath  string         `json:"source_path,omitempty"`
	ContentHash string         `json:"content_hash"`
	Metadata    map[string]any `json:"metadata,omitempty" swaggertype:"object"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
	ChunkCount  int            `json:"chunk_count,omitempty"`
}

// ChunkResult — результат поиска.
type ChunkResult struct {
	ChunkID    string         `json:"chunk_id"`
	DocumentID string         `json:"document_id"`
	Title      string         `json:"title"`
	Content    string         `json:"content"`
	Score      float64        `json:"score"`
	ChunkIndex int            `json:"chunk_index"`
	SourcePath string         `json:"source_path,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty" swaggertype:"object"`
	VectorRank int            `json:"vector_rank,omitempty"`
	TextRank   int            `json:"text_rank,omitempty"`
}

// FindByHashPath ищет документ по хешу и пути.
func (p *Postgres) FindByHashPath(ctx context.Context, hash, path string) (*Document, error) {
	row := p.pool.QueryRow(ctx, `
        SELECT d.id, d.title, d.source_path, d.content_hash, d.metadata, d.created_at, d.updated_at,
               (SELECT count(*)::int FROM rag_chunks c WHERE c.document_id = d.id)
        FROM rag_documents d
        WHERE d.content_hash = $1 AND d.source_path = $2
    `, hash, path)
	d, err := scanDocument(row)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	return d, err
}

// InsertDocument создаёт документ и чанки с эмбеддингами.
func (p *Postgres) InsertDocument(ctx context.Context, title, sourcePath, hash string, docMeta map[string]any, chunks []ChunkInput, embeddings [][]float32) (string, error) {
	if len(chunks) != len(embeddings) {
		return "", fmt.Errorf("число чанков (%d) != числу эмбеддингов (%d)", len(chunks), len(embeddings))
	}
	if len(embeddings) > 0 && len(embeddings[0]) != p.dim {
		p.dim = len(embeddings[0])
	}
	docMetaJSON, _ := json.Marshal(docMeta)
	if docMetaJSON == nil {
		docMetaJSON = []byte("{}")
	}

	tx, err := p.pool.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)

	var docID string
	err = tx.QueryRow(ctx, `
        INSERT INTO rag_documents (title, source_path, content_hash, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING id
    `, title, sourcePath, hash, docMetaJSON).Scan(&docID)
	if err != nil {
		return "", err
	}

	for i, ch := range chunks {
		vec := pgvector.NewVector(embeddings[i])
		chunkMeta := MergeChunkMeta(i, sourcePath, docMeta, ch.Metadata)
		chunkMetaJSON, _ := json.Marshal(chunkMeta)
		if chunkMetaJSON == nil {
			chunkMetaJSON = []byte("{}")
		}
		_, err = tx.Exec(ctx, `
            INSERT INTO rag_chunks (document_id, chunk_index, content, metadata, embedding)
            VALUES ($1, $2, $3, $4::jsonb, $5)
        `, docID, i, ch.Content, chunkMetaJSON, vec)
		if err != nil {
			return "", fmt.Errorf("chunk %d: %w", i, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return "", err
	}
	if err := p.ensureVectorIndex(ctx); err != nil {
		// индекс может уже существовать с другими параметрами
		_ = err
	}
	return docID, nil
}

// DeleteDocument удаляет документ и чанки.
func (p *Postgres) DeleteDocument(ctx context.Context, id string) error {
	tag, err := p.pool.Exec(ctx, `DELETE FROM rag_documents WHERE id = $1`, id)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("документ %q не найден", id)
	}
	return nil
}

// ReplaceDocument перезаписывает чанки существующего документа.
func (p *Postgres) ReplaceDocument(ctx context.Context, docID string, sourcePath string, docMeta map[string]any, chunks []ChunkInput, embeddings [][]float32) error {
	tx, err := p.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `DELETE FROM rag_chunks WHERE document_id = $1`, docID); err != nil {
		return err
	}
	for i, ch := range chunks {
		vec := pgvector.NewVector(embeddings[i])
		chunkMeta := MergeChunkMeta(i, sourcePath, docMeta, ch.Metadata)
		chunkMetaJSON, _ := json.Marshal(chunkMeta)
		if chunkMetaJSON == nil {
			chunkMetaJSON = []byte("{}")
		}
		_, err = tx.Exec(ctx, `
            INSERT INTO rag_chunks (document_id, chunk_index, content, metadata, embedding)
            VALUES ($1, $2, $3, $4::jsonb, $5)
        `, docID, i, ch.Content, chunkMetaJSON, vec)
		if err != nil {
			return err
		}
	}
	_, err = tx.Exec(ctx, `UPDATE rag_documents SET updated_at = now() WHERE id = $1`, docID)
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// ListDocuments возвращает список документов.
func (p *Postgres) ListDocuments(ctx context.Context, limit int) ([]Document, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := p.pool.Query(ctx, `
        SELECT d.id, d.title, d.source_path, d.content_hash, d.metadata, d.created_at, d.updated_at,
               (SELECT count(*)::int FROM rag_chunks c WHERE c.document_id = d.id)
        FROM rag_documents d
        ORDER BY d.updated_at DESC
        LIMIT $1
    `, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Document, 0)
	for rows.Next() {
		d, err := scanDocumentRows(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *d)
	}
	return out, rows.Err()
}

// GetDocument по ID.
func (p *Postgres) GetDocument(ctx context.Context, id string) (*Document, error) {
	row := p.pool.QueryRow(ctx, `
        SELECT d.id, d.title, d.source_path, d.content_hash, d.metadata, d.created_at, d.updated_at,
               (SELECT count(*)::int FROM rag_chunks c WHERE c.document_id = d.id)
        FROM rag_documents d WHERE d.id = $1
    `, id)
	return scanDocument(row)
}

// HybridSearch — гибридный поиск: вектор + полнотекстовый (RRF).
func (p *Postgres) HybridSearch(ctx context.Context, queryVec []float32, queryText string, topK int) ([]ChunkResult, error) {
	if topK <= 0 {
		topK = p.topK
	}
	vec := pgvector.NewVector(queryVec)
	limit := topK * 3
	if limit < 20 {
		limit = 20
	}

	vectorRows, err := p.pool.Query(ctx, `
        SELECT c.id, c.document_id, d.title, c.content, c.chunk_index, d.source_path, c.metadata,
               1 - (c.embedding <=> $1) AS score,
               ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1) AS rk
        FROM rag_chunks c
        JOIN rag_documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1
        LIMIT $2
    `, vec, limit)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}
	vecHits := map[string]ChunkResult{}
	vecRank := map[string]int{}
	for vectorRows.Next() {
		var r ChunkResult
		var rk int
		var meta []byte
		if err := vectorRows.Scan(&r.ChunkID, &r.DocumentID, &r.Title, &r.Content, &r.ChunkIndex, &r.SourcePath, &meta, &r.Score, &rk); err != nil {
			vectorRows.Close()
			return nil, err
		}
		decodeChunkMeta(&r, meta)
		r.VectorRank = rk
		minScore := config.C.RAG.Hybrid.MinVectorScore
		if minScore > 0 && r.Score < minScore {
			continue
		}
		vecHits[r.ChunkID] = r
		vecRank[r.ChunkID] = rk
	}
	vectorRows.Close()

	q := strings.TrimSpace(queryText)
	textRows, err := p.pool.Query(ctx, `
        SELECT c.id, c.document_id, d.title, c.content, c.chunk_index, d.source_path, c.metadata,
               ts_rank(c.tsv, plainto_tsquery('simple', lower($1))) AS score,
               ROW_NUMBER() OVER (ORDER BY ts_rank(c.tsv, plainto_tsquery('simple', lower($1))) DESC) AS rk
        FROM rag_chunks c
        JOIN rag_documents d ON d.id = c.document_id
        WHERE c.tsv @@ plainto_tsquery('simple', lower($1))
        ORDER BY score DESC
        LIMIT $2
    `, q, limit)
	if err != nil {
		return nil, fmt.Errorf("text search: %w", err)
	}
	textHits := map[string]ChunkResult{}
	textRank := map[string]int{}
	for textRows.Next() {
		var r ChunkResult
		var rk int
		var meta []byte
		if err := textRows.Scan(&r.ChunkID, &r.DocumentID, &r.Title, &r.Content, &r.ChunkIndex, &r.SourcePath, &meta, &r.Score, &rk); err != nil {
			textRows.Close()
			return nil, err
		}
		decodeChunkMeta(&r, meta)
		r.TextRank = rk
		textHits[r.ChunkID] = r
		textRank[r.ChunkID] = rk
	}
	textRows.Close()

	const rrfK = 60.0
	merged := map[string]*ChunkResult{}
	addRRF := func(id string, r ChunkResult, rank int, weight float64) {
		if rank <= 0 {
			return
		}
		sc := weight * (1.0 / (rrfK + float64(rank)))
		if merged[id] == nil {
			cp := r
			cp.Score = sc
			merged[id] = &cp
		} else {
			merged[id].Score += sc
		}
	}
	for id, r := range vecHits {
		addRRF(id, r, vecRank[id], p.vecW)
	}
	for id, r := range textHits {
		addRRF(id, r, textRank[id], p.textW)
	}

	results := make([]ChunkResult, 0, len(merged))
	for _, r := range merged {
		results = append(results, *r)
	}
	sortResultsDesc(results)
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

// VectorSearch — только векторный поиск (запасной, если гибрид пуст).
func (p *Postgres) VectorSearch(ctx context.Context, queryVec []float32, topK int) ([]ChunkResult, error) {
	if topK <= 0 {
		topK = p.topK
	}
	vec := pgvector.NewVector(queryVec)
	rows, err := p.pool.Query(ctx, `
        SELECT c.id, c.document_id, d.title, c.content, c.chunk_index, d.source_path, c.metadata,
               1 - (c.embedding <=> $1) AS score
        FROM rag_chunks c
        JOIN rag_documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1
        LIMIT $2
    `, vec, topK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ChunkResult
	for rows.Next() {
		var r ChunkResult
		var meta []byte
		if err := rows.Scan(&r.ChunkID, &r.DocumentID, &r.Title, &r.Content, &r.ChunkIndex, &r.SourcePath, &meta, &r.Score); err != nil {
			return nil, err
		}
		decodeChunkMeta(&r, meta)
		minScore := config.C.RAG.Hybrid.MinVectorScore
		if minScore > 0 && r.Score < minScore {
			continue
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// KeywordSearch ищет по точным редким словам/кодам модели, когда embedding даёт слабый сигнал.
func (p *Postgres) KeywordSearch(ctx context.Context, queryText string, topK int) ([]ChunkResult, error) {
	if topK <= 0 {
		topK = p.topK
	}
	terms := keywordTerms(queryText, 6)
	if len(terms) == 0 {
		return nil, nil
	}

	args := make([]any, 0, len(terms)+1)
	scoreParts := make([]string, 0, len(terms))
	whereParts := make([]string, 0, len(terms))
	for i, term := range terms {
		args = append(args, "%"+term+"%")
		param := fmt.Sprintf("$%d", i+1)
		match := fmt.Sprintf(
			"(lower(c.content) LIKE lower(%s) OR lower(d.title) LIKE lower(%s) OR lower(d.source_path) LIKE lower(%s))",
			param, param, param,
		)
		scoreParts = append(scoreParts, fmt.Sprintf("CASE WHEN %s THEN 1 ELSE 0 END", match))
		whereParts = append(whereParts, match)
	}
	args = append(args, topK)
	limitParam := fmt.Sprintf("$%d", len(args))

	sql := fmt.Sprintf(`
        SELECT c.id, c.document_id, d.title, c.content, c.chunk_index, d.source_path, c.metadata,
               (%s)::float8 AS score
        FROM rag_chunks c
        JOIN rag_documents d ON d.id = c.document_id
        WHERE %s
        ORDER BY score DESC, c.chunk_index ASC
        LIMIT %s
    `, strings.Join(scoreParts, " + "), strings.Join(whereParts, " OR "), limitParam)

	rows, err := p.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]ChunkResult, 0, topK)
	for rows.Next() {
		var r ChunkResult
		var meta []byte
		if err := rows.Scan(&r.ChunkID, &r.DocumentID, &r.Title, &r.Content, &r.ChunkIndex, &r.SourcePath, &meta, &r.Score); err != nil {
			return nil, err
		}
		decodeChunkMeta(&r, meta)
		out = append(out, r)
	}
	return out, rows.Err()
}

func decodeChunkMeta(r *ChunkResult, meta []byte) {
	if len(meta) > 0 {
		_ = json.Unmarshal(meta, &r.Metadata)
	}
}

func keywordTerms(query string, max int) []string {
	query = strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' {
			return r
		}
		return ' '
	}, query)

	seen := map[string]struct{}{}
	out := make([]string, 0, max)
	for _, term := range strings.Fields(query) {
		term = strings.Trim(term, "-_")
		if len([]rune(term)) < 4 {
			continue
		}
		key := strings.ToLower(term)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, term)
		if len(out) >= max {
			break
		}
	}
	return out
}

func sortResultsDesc(r []ChunkResult) {
	sort.Slice(r, func(i, j int) bool { return r[i].Score > r[j].Score })
}

func scanDocument(row pgx.Row) (*Document, error) {
	var d Document
	var meta []byte
	err := row.Scan(&d.ID, &d.Title, &d.SourcePath, &d.ContentHash, &meta, &d.CreatedAt, &d.UpdatedAt, &d.ChunkCount)
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal(meta, &d.Metadata)
	return &d, nil
}

func scanDocumentRows(rows pgx.Rows) (*Document, error) {
	var d Document
	var meta []byte
	err := rows.Scan(&d.ID, &d.Title, &d.SourcePath, &d.ContentHash, &meta, &d.CreatedAt, &d.UpdatedAt, &d.ChunkCount)
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal(meta, &d.Metadata)
	return &d, nil
}
