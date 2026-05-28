package rag

import (
	"fmt"
	"log"
	"strings"

	llama "github.com/redirsvr/go-llama-new.cpp"

	"llama-cpp-gpt-api/internal/config"
	"llama-cpp-gpt-api/internal/types"
	"llama-cpp-gpt-api/pkg/model"
)

var _ = llama.SetTokens // keep import

const defaultRewritePrompt = `Rewrite the following question as a focused, concise search query for document retrieval. Extract the key entities and concepts. Return only the rewritten query in the same language as the original, no explanations.

Original: %s
Rewritten:`

const shortRewriteResponse = 200

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
		if j := strings.LastIndex(s, "Вопрос:"); j >= 0 {
			return strings.TrimSpace(s[j+len("Вопрос:"):])
		}
		return ""
	}
	return s
}

// RewriteSearchQuery улучшает поисковый запрос через LLM (как multi-modal analysis в RAG-Anything).
// Возвращает переписанный запрос, более подходящий для векторного поиска.
// Требует RAG.QueryRewriting.Enabled = true.
func RewriteSearchQuery(original string) string {
	if original == "" {
		return original
	}
	rc := config.C.RAG.QueryRewriting
	if !rc.Enabled {
		return original
	}
	modelAlias := rc.Model
	if modelAlias == "" {
		modelAlias = config.C.DefaultModel
	}
	if modelAlias == "" {
		return original
	}

	log.Printf("RAG: переписывание запроса через %q: %q", modelAlias, truncateQueryLog(original, 60))

	prompt := rc.Prompt
	if prompt == "" {
		prompt = defaultRewritePrompt
	}
	prompt = fmt.Sprintf(prompt, original)

	var rewritten string
	err := model.UseChat(modelAlias, func(ll *llama.LLama, _ string) error {
		result, err := ll.Predict(prompt, llama.SetTokens(shortRewriteResponse))
		if err != nil {
			return fmt.Errorf("rewrite prediction: %w", err)
		}
		rewritten = strings.TrimSpace(result)
		return nil
	})
	if err != nil {
		log.Printf("RAG: переписывание запроса не удалось (%v) — использую оригинал", err)
		return original
	}
	if rewritten == "" {
		return original
	}

	rewritten = strings.Trim(rewritten, `"'«»`)
	rewritten = PrepareText(strings.TrimSpace(rewritten))

	if rewritten == "" {
		return original
	}

	log.Printf("RAG: запрос переписан: %q → %q", truncateQueryLog(original, 40), truncateQueryLog(rewritten, 60))
	return rewritten
}
