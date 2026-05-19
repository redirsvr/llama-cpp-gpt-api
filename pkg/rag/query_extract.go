package rag

import (
	"strings"

	"llama-cpp-gpt-api/internal/types"
)

// OpenWebUIServiceTask определяет служебные запросы Open WebUI (follow-up, title, tags).
func OpenWebUIServiceTask(query string) string {
	q := strings.TrimSpace(query)
	if q == "" {
		return ""
	}
	if !strings.HasPrefix(q, "### Task:") && !strings.Contains(q, "<chat_history>") {
		return ""
	}
	lower := strings.ToLower(q)
	switch {
	case strings.Contains(lower, "follow_ups") || strings.Contains(lower, "follow-up"):
		return "follow_ups"
	case strings.Contains(lower, "\"title\"") || strings.Contains(lower, "generate a concise") && strings.Contains(lower, "title"):
		return "title"
	case strings.Contains(lower, "\"tags\"") || strings.Contains(lower, "generate 1-3 broad tags"):
		return "tags"
	default:
		return "task"
	}
}

// ShouldSkipRAGSearch — true, если это не пользовательский вопрос для базы знаний.
func ShouldSkipRAGSearch(query string) bool {
	q := strings.TrimSpace(query)
	return q == "" || OpenWebUIServiceTask(q) != ""
}

// ExtractSearchQuery — короткий текст для embedding-поиска (без прошлого RAG-контекста в истории).
func ExtractSearchQuery(messages []types.Message, prompt string) string {
	q := lastUserText(messages)
	if q == "" {
		q = strings.TrimSpace(prompt)
	}
	q = stripInjectedRAGBlock(q)
	q = PrepareText(strings.TrimSpace(q))
	if q == "" || ShouldSkipRAGSearch(q) {
		return ""
	}
	return TruncateRunes(q, MaxEmbedRunes())
}

func lastUserText(messages []types.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if strings.EqualFold(strings.TrimSpace(messages[i].Role), "user") {
			return messages[i].Content
		}
	}
	return ""
}

func stripInjectedRAGBlock(s string) string {
	s = strings.TrimSpace(s)
	if idx := strings.Index(s, "\n---\nВопрос:"); idx >= 0 {
		return strings.TrimSpace(s[idx+len("\n---\nВопрос:"):])
	}
	if idx := strings.Index(s, "Фрагменты документов:"); idx >= 0 {
		// повторный запрос с уже вставленным RAG — берём только хвост после последнего «Вопрос:»
		if j := strings.LastIndex(s, "Вопрос:"); j >= 0 {
			return strings.TrimSpace(s[j+len("Вопрос:"):])
		}
		return ""
	}
	return s
}
