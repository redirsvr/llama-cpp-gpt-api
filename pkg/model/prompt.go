package model

import (
    "regexp"
    "strings"

    "llama-cpp-gpt-api/internal/config"
    "llama-cpp-gpt-api/internal/types"
)

const ChatTemplateQwen = "qwen"
const ChatTemplatePlain = "plain"

var (
    tokenIMStart = "<|" + "im_start" + "|>"
    tokenIMEnd   = "<|" + "im_end" + "|>"
    tokenThinkO  = "<" + "think" + ">"
    tokenThinkC  = "<" + "/" + "think" + ">"
)

var (
    reThinkBlockClosed  = regexp.MustCompile("(?s)" + regexp.QuoteMeta(tokenThinkO) + "[\\s\\S]*?" + regexp.QuoteMeta(tokenThinkC))
    reRedactedThinkClosed = regexp.MustCompile(`(?is)<think>[\s\S]*?</think>`)
    reRedactedThinkTags   = regexp.MustCompile(`(?is)</?redacted_thinking>`)
)

// FormatChatPrompt собирает промпт под chat-модель.
func FormatChatPrompt(modelAlias string, messages []types.Message) string {
    tmpl := strings.ToLower(strings.TrimSpace(config.C.ChatTemplate))
    if tmpl == "" && strings.Contains(strings.ToLower(modelAlias), "qwen") {
        tmpl = ChatTemplateQwen
    }
    if tmpl == "" {
        tmpl = ChatTemplateQwen
    }
    switch tmpl {
    case ChatTemplatePlain, "legacy":
        return formatPlainPrompt(messages)
    default:
        return formatQwenPrompt(modelAlias, messages)
    }
}

// prepareChatMessages формирует список сообщений как в OpenAI Chat Completions.
// Если RAG уже вставил system с «Фрагменты документов» — не дублируем SystemPrompt.
func prepareChatMessages(messages []types.Message) []types.Message {
    if len(messages) == 0 {
        return messages
    }
    if strings.EqualFold(strings.TrimSpace(messages[0].Role), "system") &&
        strings.Contains(messages[0].Content, "Фрагменты документов") {
        return messages
    }
    cfg := strings.TrimSpace(config.C.SystemPrompt)
    if cfg == "" {
        out := make([]types.Message, len(messages))
        copy(out, messages)
        return out
    }
    if strings.EqualFold(strings.TrimSpace(messages[0].Role), "system") {
        out := make([]types.Message, len(messages))
        copy(out, messages)
        if !strings.Contains(out[0].Content, cfg) {
            out[0].Content = cfg + "\n\n" + out[0].Content
        }
        return out
    }
    out := make([]types.Message, 0, len(messages)+1)
    out = append(out, types.Message{Role: "system", Content: cfg})
    out = append(out, messages...)
    return out
}

func formatQwenPrompt(modelAlias string, messages []types.Message) string {
    msgs := prepareChatMessages(messages)
    if shouldDisableThinking(modelAlias) {
        appendNoThinkHint(&msgs)
    }

    var b strings.Builder
    for _, m := range msgs {
        writeIMTurn(&b, normalizeRole(m.Role), m.Content)
    }
    b.WriteString(tokenIMStart)
    b.WriteString("assistant\n")
    return b.String()
}

func shouldDisableThinking(modelAlias string) bool {
    if config.C.DisableThinking {
        return true
    }
    lower := strings.ToLower(modelAlias)
    return strings.Contains(lower, "qwen3")
}

func appendNoThinkHint(msgs *[]types.Message) {
    for i := len(*msgs) - 1; i >= 0; i-- {
        if normalizeRole((*msgs)[i].Role) == "user" {
            c := strings.TrimSpace((*msgs)[i].Content)
            if !strings.Contains(c, "/no_think") {
                (*msgs)[i].Content = c + "\n/no_think"
            }
            return
        }
    }
}

func writeIMTurn(b *strings.Builder, role, content string) {
    b.WriteString(tokenIMStart)
    b.WriteString(role)
    b.WriteString("\n")
    b.WriteString(content)
    b.WriteString("\n")
    b.WriteString(tokenIMEnd)
    b.WriteByte('\n')
}

func formatPlainPrompt(messages []types.Message) string {
    msgs := prepareChatMessages(messages)
    var b strings.Builder
    for i, m := range msgs {
        role := normalizeRole(m.Role)
        switch role {
        case "system":
            b.WriteString("System: ")
        case "assistant":
            b.WriteString("Assistant: ")
        default:
            b.WriteString(role)
            b.WriteString(": ")
        }
        b.WriteString(m.Content)
        if i < len(msgs)-1 {
            b.WriteString("\n\n")
        }
    }
    b.WriteString("\n\nassistant: ")
    return b.String()
}

func normalizeRole(role string) string {
    switch strings.ToLower(strings.TrimSpace(role)) {
    case "system", "user", "assistant", "tool":
        return strings.ToLower(strings.TrimSpace(role))
    default:
        return "user"
    }
}

// QwenStopWords — стоп-токены для ChatML/Qwen.
// Не включать </think>: antiprompt в llama.cpp обрывает генерацию при
// появлении этой подстроки в выводе — до основного ответа модель не доходит.
func QwenStopWords() []string {
    return []string{tokenIMEnd, tokenIMStart}
}

// CleanAssistantReply убирает thinking-блоки и хвост шаблона из ответа модели.
func CleanAssistantReply(s string) string {
    raw := strings.TrimSpace(s)
    if raw == "" {
        return ""
    }
    cleaned := strings.TrimSpace(stripAssistantArtifacts(raw))
    if cleaned != "" {
        return cleaned
    }
    // fallback: только снять XML-теги, текст внутри thinking сохранить
    return strings.TrimSpace(reRedactedThinkTags.ReplaceAllString(raw, ""))
}

func stripAssistantArtifacts(s string) string {
    if idx := strings.LastIndex(s, "</think>"); idx >= 0 {
        s = s[idx+len("</think>"):]
    }
    s = reRedactedThinkClosed.ReplaceAllString(s, "")
    s = reThinkBlockClosed.ReplaceAllString(s, "")
    s = reRedactedThinkTags.ReplaceAllString(s, "")
    if idx := strings.Index(s, tokenIMStart); idx >= 0 {
        s = s[:idx]
    }
    if idx := strings.Index(s, tokenIMEnd); idx >= 0 {
        s = s[:idx]
    }
    return s
}
