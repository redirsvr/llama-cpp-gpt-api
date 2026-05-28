package rag

import (
	"fmt"
	"strings"

	"llama-cpp-gpt-api/internal/config"
	"llama-cpp-gpt-api/internal/types"
	"llama-cpp-gpt-api/pkg/rag/store"
)

const ragMarker = "Фрагменты документов:"
const ragInstruction = `Отвечай на вопрос пользователя. Ниже — фрагменты из базы документов; используй их, если в них есть релевантные факты.`

// BuildContext собирает блок контекста для system-промпта.
func BuildContext(results []store.ChunkResult) string {
	if len(results) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString(ragInstruction)
	b.WriteString("\n\n")
	b.WriteString(ragMarker)
	b.WriteString("\n\n")
	for i, r := range results {
		b.WriteString(fmt.Sprintf("[%d] %s (score %.3f)", i+1, chunkCitation(r), r.Score))
		if kws := extractKeywordsFromMeta(r.Metadata); len(kws) > 0 {
			b.WriteString(fmt.Sprintf(" [теги: %s]", strings.Join(kws, ", ")))
		}
		b.WriteString(fmt.Sprintf("\n%s\n\n", r.Content))
	}
	return b.String()
}

func chunkCitation(r store.ChunkResult) string {
	parts := []string{fmt.Sprintf("«%s»", r.Title)}
	if page := metaPage(r.Metadata); page > 0 {
		parts = append(parts, fmt.Sprintf("стр. %d", page))
	}
	idx := r.ChunkIndex
	if idx == 0 {
		idx = metaChunkIndex(r.Metadata)
	}
	if idx >= 0 {
		parts = append(parts, fmt.Sprintf("фрагмент %d", idx+1))
	}
	if sp := strings.TrimSpace(r.SourcePath); sp != "" {
		parts = append(parts, sp)
	}
	return strings.Join(parts, ", ")
}

func metaPage(m map[string]any) int {
	return metaInt(m, "page")
}

func metaChunkIndex(m map[string]any) int {
	return metaInt(m, "chunk_index")
}

func metaInt(m map[string]any, key string) int {
	if m == nil {
		return 0
	}
	v, ok := m[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

// StripRAGFromMessages убирает RAG-инструкции из истории, когда контекст базы не подмешивается.
func StripRAGFromMessages(messages []types.Message) []types.Message {
	if len(messages) == 0 {
		return messages
	}
	out := make([]types.Message, 0, len(messages))
	for _, m := range messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		content := m.Content
		if role == "system" && containsRAGBlock(content) {
			continue
		}
		if role == "user" {
			content = stripInjectedRAGBlock(content)
		}
		if strings.TrimSpace(content) == "" && role == "user" {
			continue
		}
		out = append(out, types.Message{Role: m.Role, Content: content})
	}
	return out
}

func containsRAGBlock(s string) bool {
	return strings.Contains(s, ragMarker) ||
		strings.Contains(s, "Используй ТОЛЬКО факты из блока") ||
		strings.Contains(s, "В загруженных документах этого нет")
}

// ApplyRAGToMessages кладёт RAG в одно system-сообщение; в user остаётся только чистый вопрос.
func ApplyRAGToMessages(messages []types.Message, ragBlock string) []types.Message {
	if ragBlock == "" {
		return messages
	}
	sys := ragBlock
	if base := strings.TrimSpace(config.C.SystemPrompt); base != "" && !strings.Contains(ragBlock, base) {
		sys = base + "\n\n---\n\n" + ragBlock
	}

	var rest []types.Message
	for _, m := range messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		if role == "system" {
			continue
		}
		cp := m
		if role == "user" {
			cp.Content = stripInjectedRAGBlock(m.Content)
		}
		if strings.TrimSpace(cp.Content) != "" || role != "user" {
			rest = append(rest, cp)
		}
	}
	return append([]types.Message{{Role: "system", Content: sys}}, rest...)
}

// HasRAGInPrompt проверяет, что контекст попал в итоговый промпт.
func HasRAGInPrompt(prompt string) bool {
	return strings.Contains(prompt, ragMarker)
}
