package rag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	llama "github.com/redirsvr/go-llama-new.cpp"

	"llama-cpp-gpt-api/internal/config"
	"llama-cpp-gpt-api/pkg/model"
	"llama-cpp-gpt-api/pkg/rag/chunking"
	"llama-cpp-gpt-api/pkg/rag/store"
)

// Service — Hybrid RAG: семантический чанкинг, pgvector, гибридный поиск.
type Service struct {
	db *store.Postgres
}

var defaultSvc *Service

// Init подключает PostgreSQL и при необходимости сканирует каталоги.
func Init(ctx context.Context) error {
	if !config.C.RAG.Enabled {
		return nil
	}
	db, err := store.Open(ctx)
	if err != nil {
		return err
	}
	defaultSvc = &Service{db: db}
	log.Println("RAG: PostgreSQL + pgvector готов")
	if config.C.RAG.ChatRAGEnabled() {
		topK := config.C.RAG.Hybrid.TopK
		if topK <= 0 {
			topK = 8
		}
		log.Printf("RAG: /v1/chat/completions — контекст из базы включён (TopK=%d)", topK)
	} else {
		log.Println("RAG: /v1/chat/completions — контекст из базы выключен (UseInChatCompletions: false)")
	}
	if total, withEmb, err := db.ChunkStats(ctx); err == nil {
		log.Printf("RAG: в базе %d чанков (%d с эмбеддингами)", total, withEmb)
		if withEmb == 0 && total > 0 {
			log.Println("RAG: предупреждение — чанки без embedding, поиск не сработает; переиндексируйте документы")
		}
	}
	if p := strings.TrimSpace(config.C.RAG.EmbedDocumentPrefix); p != "" {
		log.Printf("RAG: префиксы embedding: query=%q document=%q — после смены нужна переиндексация",
			config.C.RAG.EmbedQueryPrefix, p)
	}

	if config.C.RAG.AutoIngest.ScanOnStart {
		n, err := defaultSvc.IngestConfiguredDirectories(ctx)
		if err != nil {
			log.Printf("RAG: авто-ingest при старте: %v", err)
		} else if n > 0 {
			log.Printf("RAG: проиндексировано файлов при старте: %d", n)
		}
	}
	return nil
}

// Close закрывает пул соединений.
func Close() {
	if defaultSvc != nil && defaultSvc.db != nil {
		defaultSvc.db.Close()
		defaultSvc = nil
	}
}

// Enabled возвращает true, если RAG инициализирован.
func Enabled() bool {
	return defaultSvc != nil
}

func svc() (*Service, error) {
	if defaultSvc == nil {
		return nil, fmt.Errorf("RAG отключён или не инициализирован")
	}
	return defaultSvc, nil
}

// IngestInput — параметры добавления документа.
type IngestInput struct {
	Title      string
	Content    string
	SourcePath string
	Metadata   map[string]any
}

// IngestDocument индексирует текст с семантическим чанкингом.
func IngestDocument(ctx context.Context, in IngestInput) (*store.Document, error) {
	s, err := svc()
	if err != nil {
		return nil, err
	}
	return s.ingest(ctx, in)
}

func (s *Service) ingest(ctx context.Context, in IngestInput) (*store.Document, error) {
	content := PrepareText(strings.TrimSpace(in.Content))
	if content == "" {
		return nil, fmt.Errorf("пустой контент документа")
	}
	title := strings.TrimSpace(in.Title)
	if title == "" {
		title = "document"
	}
	path := strings.TrimSpace(in.SourcePath)
	hash := hashContent(content)

	existing, err := s.db.FindByHashPath(ctx, hash, path)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		log.Printf("RAG: документ уже проиндексирован %q (%s)", title, path)
		return existing, nil
	}

	log.Printf("RAG: индексация %q (%d символов)", title, utf8.RuneCountInString(content))
	chunks, embeddings, err := s.chunkAndEmbed(content)
	if err != nil {
		return nil, WrapIngestError(err)
	}
	chunks, embeddings, err = alignChunksEmbeddings(chunks, embeddings)
	if err != nil {
		return nil, WrapIngestError(err)
	}
	log.Printf("RAG: готово %q — %d чанков", title, len(chunks))

	chunkInputs := chunking.ChunksWithMeta(content, chunks, path, in.Metadata)

	id, err := s.db.InsertDocument(ctx, title, path, hash, in.Metadata, chunkInputs, embeddings)
	if err != nil {
		return nil, err
	}
	return s.db.GetDocument(ctx, id)
}

func (s *Service) chunkAndEmbed(content string) ([]string, [][]float32, error) {
	cfg := chunking.Config{
		MaxChunkChars:        config.C.RAG.Chunking.MaxChunkChars,
		MinChunkChars:        config.C.RAG.Chunking.MinChunkChars,
		BreakpointPercentile: config.C.RAG.Chunking.BreakpointPercentile,
		MaxSentencesSemantic: config.C.RAG.Chunking.MaxSentencesSemantic,
	}
	modelAlias := model.DefaultEmbeddingModel()
	if modelAlias == "" {
		return nil, nil, fmt.Errorf("не задана embedding-модель (RAG.EmbeddingModel или DefaultEmbeddingModel)")
	}

	var chunks []string
	var embeddings [][]float32
	err := model.UseEmbeddings(modelAlias, func(ll *llama.LLama, _ string) error {
		maxEmbed := MaxEmbedRunes()
		embedBatch := func(texts []string) ([][]float32, error) {
			texts = CapSegments(texts, maxEmbed)
			out := make([][]float32, 0, len(texts))
			for i, t := range texts {
				t = PrepareText(strings.TrimSpace(t))
				if t == "" {
					out = append(out, nil)
					continue
				}
				t = TruncateRunes(t, maxEmbed)
				v, err := model.EmbedString(ll, PrefixDocument(t))
				if err != nil {
					return nil, fmt.Errorf("текст %d: %w", i, err)
				}
				out = append(out, v)
			}
			return out, nil
		}

		var err error
		chunks, err = chunking.SemanticChunks(content, embedBatch, cfg)
		if err != nil {
			return err
		}
		if len(chunks) == 0 {
			chunks = []string{content}
		}
		embeddings, err = embedBatch(chunks)
		return err
	})
	if err != nil {
		return nil, nil, err
	}
	return chunks, embeddings, nil
}

// IngestFile индексирует файл по пути на диске (без загрузки всего в память для больших файлов).
func IngestFile(ctx context.Context, path string) (*store.Document, error) {
	s, err := svc()
	if err != nil {
		return nil, err
	}
	title := filepath.Base(path)
	return s.ingestFromPath(ctx, path, title, map[string]any{"filename": title})
}

// ScanDirectories — публичная обёртка для сканирования каталогов из конфига.
func ScanDirectories(ctx context.Context) (int, error) {
	s, err := svc()
	if err != nil {
		return 0, err
	}
	return s.IngestConfiguredDirectories(ctx)
}

// IngestConfiguredDirectories сканирует каталоги из конфига.
func (s *Service) IngestConfiguredDirectories(ctx context.Context) (int, error) {
	ai := config.C.RAG.AutoIngest
	if !ai.Enabled {
		return 0, nil
	}
	exts := ai.Extensions
	if len(exts) == 0 {
		exts = []string{".txt", ".md", ".markdown"}
	}
	extSet := map[string]struct{}{}
	for _, e := range exts {
		extSet[strings.ToLower(e)] = struct{}{}
	}

	var count int
	for _, dir := range ai.Directories {
		dir = strings.TrimSpace(dir)
		if dir == "" {
			continue
		}
		err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			ext := strings.ToLower(filepath.Ext(path))
			if _, ok := extSet[ext]; !ok {
				return nil
			}
			if _, err := s.ingestFromPath(ctx, path, filepath.Base(path), map[string]any{"auto_ingest": true}); err != nil {
				log.Printf("RAG: пропуск %q: %v", path, err)
				return nil
			}
			count++
			return nil
		})
		if err != nil {
			return count, fmt.Errorf("каталог %q: %w", dir, err)
		}
	}
	return count, nil
}

// Query выполняет гибридный поиск.
func Query(ctx context.Context, query string, topK int) ([]store.ChunkResult, error) {
	s, err := svc()
	if err != nil {
		return nil, err
	}
	query = PrepareText(strings.TrimSpace(query))
	if query == "" {
		return nil, fmt.Errorf("пустой запрос")
	}
	maxR := MaxEmbedRunes()
	if utf8.RuneCountInString(query) > maxR {
		log.Printf("RAG: запрос обрезан до %d рун для embedding", maxR)
		query = TruncateRunes(query, maxR)
	}
	modelAlias := model.DefaultEmbeddingModel()
	hits, err := s.searchWithQueryVariants(ctx, modelAlias, query, topK)
	if err != nil {
		return nil, err
	}
	return hits, nil
}

func (s *Service) searchWithQueryVariants(ctx context.Context, modelAlias, query string, topK int) ([]store.ChunkResult, error) {
	variants := []struct {
		label string
		text  string
	}{
		{"search_query", PrefixQuery(query)},
		{"plain", query},
	}
	var lastVec []float32
	for _, v := range variants {
		vecs, err := model.EmbedTexts(modelAlias, []string{v.text})
		if err != nil || len(vecs) == 0 {
			continue
		}
		lastVec = vecs[0]
		hits, err := s.db.HybridSearch(ctx, vecs[0], query, topK)
		if err != nil {
			return nil, err
		}
		if len(hits) == 0 {
			hits, err = s.db.VectorSearch(ctx, vecs[0], topK)
			if err != nil {
				return nil, err
			}
		}
		if len(hits) > 0 {
			if v.label != "search_query" {
				log.Printf("RAG: совпадения по запросу (%s), префикс search_query не сработал — нужна POST /v1/rag/reindex-embeddings", v.label)
			}
			return hits, nil
		}
	}
	hits, err := s.db.KeywordSearch(ctx, query, topK)
	if err != nil {
		return nil, err
	}
	if len(hits) > 0 {
		log.Printf("RAG: vector/hybrid пуст — использован keyword fallback (%d чанков)", len(hits))
		return hits, nil
	}
	if len(lastVec) > 0 {
		if best, err := s.db.BestVectorScore(ctx, lastVec); err == nil {
			log.Printf("RAG: совпадений нет; лучший cosine≈%.3f. Выполните: curl -X POST http://127.0.0.1:8080/v1/rag/reindex-embeddings", best)
		}
	}
	return nil, nil
}

// ListDocuments список документов.
func ListDocuments(ctx context.Context, limit int) ([]store.Document, error) {
	s, err := svc()
	if err != nil {
		return nil, err
	}
	return s.db.ListDocuments(ctx, limit)
}

// GetDocument по ID.
func GetDocument(ctx context.Context, id string) (*store.Document, error) {
	s, err := svc()
	if err != nil {
		return nil, err
	}
	return s.db.GetDocument(ctx, id)
}

// DeleteDocument удаляет документ.
func DeleteDocument(ctx context.Context, id string) error {
	s, err := svc()
	if err != nil {
		return err
	}
	return s.db.DeleteDocument(ctx, id)
}

// FormatPromptForRAG дополняет сообщения пользователя контекстом RAG.
func FormatPromptForRAG(ctx context.Context, userQuery string, topK int) (string, error) {
	ctxText, _, err := FormatPromptForRAGWithHits(ctx, userQuery, topK)
	return ctxText, err
}

// FormatPromptForRAGWithHits как FormatPromptForRAG, плюс найденные чанки.
func FormatPromptForRAGWithHits(ctx context.Context, userQuery string, topK int) (ctxText string, hits []store.ChunkResult, err error) {
	if topK <= 0 {
		topK = config.C.RAG.Hybrid.TopK
	}
	hits, err = Query(ctx, userQuery, topK)
	if err != nil {
		return "", nil, err
	}
	logRAGHits(userQuery, hits)
	return BuildContext(hits), hits, nil
}

func logRAGHits(query string, hits []store.ChunkResult) {
	if len(hits) == 0 {
		return
	}
	var b strings.Builder
	for i, h := range hits {
		if i > 0 {
			b.WriteString("; ")
		}
		excerpt := strings.TrimSpace(h.Content)
		if len(excerpt) > 60 {
			excerpt = excerpt[:60] + "…"
		}
		page := metaPage(h.Metadata)
		if page > 0 {
			b.WriteString(fmt.Sprintf("#%d score=%.3f «%s» стр.%d фр.%d %q", i+1, h.Score, h.Title, page, h.ChunkIndex+1, excerpt))
		} else {
			b.WriteString(fmt.Sprintf("#%d score=%.3f «%s» фр.%d %q", i+1, h.Score, h.Title, h.ChunkIndex+1, excerpt))
		}
	}
	log.Printf("RAG: топ чанков для %q: %s", truncateQueryLog(query, 60), b.String())
}

func truncateQueryLog(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func hashContent(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// IngestReader сохраняет upload на диск и индексирует оттуда по чанкам.
func IngestReader(ctx context.Context, title string, r io.Reader, meta map[string]any) (doc *store.Document, err error) {
	s, err := svc()
	if err != nil {
		return nil, err
	}
	fname := title
	if meta != nil {
		if f, ok := meta["filename"].(string); ok && f != "" {
			fname = f
		}
	}
	path, err := SaveUpload(r, fname)
	if err != nil {
		return nil, WrapIngestError(err)
	}
	defer func() {
		if err != nil {
			_ = os.Remove(path)
		}
	}()
	log.Printf("RAG: файл сохранён на диск %q", path)
	doc, err = s.ingestFromPath(ctx, path, title, meta)
	if err != nil {
		return nil, WrapIngestError(err)
	}
	RemoveStaging(path)
	return doc, nil
}

// StartAutoIngestWatcher периодически переиндексирует каталоги.
func StartAutoIngestWatcher(ctx context.Context) {
	ai := config.C.RAG.AutoIngest
	if !ai.Enabled || ai.IntervalSec <= 0 {
		return
	}
	go func() {
		ticker := time.NewTicker(time.Duration(ai.IntervalSec) * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if defaultSvc == nil {
					continue
				}
				n, err := defaultSvc.IngestConfiguredDirectories(ctx)
				if err != nil {
					log.Printf("RAG: авто-ingest: %v", err)
				} else if n > 0 {
					log.Printf("RAG: авто-ingest: новых/обновлённых %d", n)
				}
			}
		}
	}()
}
