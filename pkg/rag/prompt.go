package rag

import (
	"fmt"
	"strings"

	"llama-cpp-gpt-api/internal/config"
	"llama-cpp-gpt-api/internal/types"
	"llama-cpp-gpt-api/pkg/rag/store"
)

const ragMarker = "Фрагменты документов:"
const ragInstruction = `Ты отвечаешь на вопрос пользователя. Используй ТОЛЬКО факты из блока «Фрагменты документов» ниже.
Если ответа нет во фрагментах — скажи: «В загруженных документах этого нет». Не используй общие знания модели.`

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
		b.WriteString(fmt.Sprintf("[%d] %s (score %.3f)\n%s\n\n", i+1, chunkCitation(r), r.Score, r.Content))
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
