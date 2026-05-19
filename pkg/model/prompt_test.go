package model

import (
    "strings"
    "testing"

    "llama-cpp-gpt-api/internal/config"
    "llama-cpp-gpt-api/internal/types"
)

func TestPrepareChatMessages_OpenAI(t *testing.T) {
    config.C = config.Config{SystemPrompt: "Default from config."}

    onlyUser := prepareChatMessages([]types.Message{
        {Role: "user", Content: "Hi"},
    })
    if len(onlyUser) != 2 || normalizeRole(onlyUser[0].Role) != "system" {
        t.Fatalf("expected config system first: %+v", onlyUser)
    }
    if onlyUser[0].Content != "Default from config." {
        t.Fatalf("config system content: %q", onlyUser[0].Content)
    }

    withSystem := prepareChatMessages([]types.Message{
        {Role: "system", Content: "From request."},
        {Role: "user", Content: "Hi"},
    })
    if len(withSystem) != 3 {
        t.Fatalf("expected 3 messages, got %d", len(withSystem))
    }
    if withSystem[0].Content != "Default from config." || withSystem[1].Content != "From request." {
        t.Fatalf("order: %+v", withSystem)
    }

    config.C.SystemPrompt = ""
    noConfig := prepareChatMessages([]types.Message{
        {Role: "system", Content: "Only request."},
        {Role: "user", Content: "Hi"},
    })
    if len(noConfig) != 2 || noConfig[0].Content != "Only request." {
        t.Fatalf("without config: %+v", noConfig)
    }
}

func TestFormatQwenPrompt(t *testing.T) {
    config.C = config.Config{
        SystemPrompt: "You are helpful.",
        ChatTemplate: "qwen",
    }
    config.C.DisableThinking = true
    got := FormatChatPrompt("Qwen3.6-35B", []types.Message{
        {Role: "user", Content: "Привет"},
    })
    if !strings.Contains(got, "You are helpful.") {
        t.Fatalf("missing config system: %q", got)
    }
    if !strings.Contains(got, "/no_think") {
        t.Fatalf("expected /no_think hint: %q", got)
    }
    if !strings.Contains(got, tokenIMStart+"user") {
        t.Fatalf("missing user turn: %q", got)
    }
}

func TestFormatQwenPrompt_RequestSystemAfterConfig(t *testing.T) {
    config.C = config.Config{
        SystemPrompt: "Config system.",
        ChatTemplate: "qwen",
    }
    got := FormatChatPrompt("qwen2", []types.Message{
        {Role: "system", Content: "Request system."},
        {Role: "user", Content: "Hi"},
    })
    idxCfg := strings.Index(got, "Config system.")
    idxReq := strings.Index(got, "Request system.")
    if idxCfg < 0 || idxReq < 0 || idxCfg > idxReq {
        t.Fatalf("config system must precede request system: %q", got)
    }
}

func TestCleanAssistantReply(t *testing.T) {
    raw := "\n\n\n<think>plan</think>\n\nОтвет про llama.cpp."
    got := CleanAssistantReply(raw)
    if strings.Contains(got, "redacted_thinking") {
        t.Fatalf("thinking not stripped: %q", got)
    }
    if !strings.Contains(got, "llama.cpp") {
        t.Fatalf("answer lost: %q", got)
    }
}

func TestCleanAssistantReply_UnclosedThinking(t *testing.T) {
    raw := "<think>only thought no close"
    got := CleanAssistantReply(raw)
    if got == "" {
        t.Fatalf("unclosed thinking must not erase all text: %q", raw)
    }
}

func TestCleanAssistantReply_OnlyClosingTag(t *testing.T) {
    raw := "</think>\n\nТекст ответа."
    got := CleanAssistantReply(raw)
    if !strings.Contains(got, "Текст ответа") {
        t.Fatalf("expected answer after closing tag: %q", got)
    }
}
