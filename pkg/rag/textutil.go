package rag

import (
    "strings"
    "unicode/utf8"

    "llama-cpp-gpt-api/internal/config"
)

// MaxEmbedRunes — безопасный размер одного чанка для Embeddings (токены ≤ n_ubatch).
func MaxEmbedRunes() int {
    m := config.C.RAG.Chunking.MaxEmbedRunes
    if m <= 0 {
        m = 512
    }
    if c := config.C.RAG.Chunking.MaxChunkChars; c > 0 && c < m {
        m = c
    }
    return m
}

// PrepareText нормализует UTF-8 и убирает управляющие символы (llama.cpp падает на invalid codepoint).
func PrepareText(s string) string {
    if !utf8.ValidString(s) {
        s = strings.ToValidUTF8(s, "")
    }
    return strings.Map(func(r rune) rune {
        if r == utf8.RuneError {
            return -1
        }
        if r < 32 && r != '\n' && r != '\r' && r != '\t' {
            return -1
        }
        return r
    }, s)
}

// TruncateRunes обрезает текст до max рун (безопасно для UTF-8).
func TruncateRunes(s string, maxRunes int) string {
    if maxRunes <= 0 || utf8.RuneCountInString(s) <= maxRunes {
        return s
    }
    var b strings.Builder
    b.Grow(len(s))
    n := 0
    for _, r := range s {
        if n >= maxRunes {
            break
        }
        b.WriteRune(r)
        n++
    }
    return b.String()
}

// CapSegments режет слишком длинные фрагменты перед embedding.
func CapSegments(texts []string, maxRunes int) []string {
    if maxRunes <= 0 {
        return texts
    }
    var out []string
    for _, t := range texts {
        t = strings.TrimSpace(t)
        if t == "" {
            continue
        }
        for utf8.RuneCountInString(t) > maxRunes {
            out = append(out, TruncateRunes(t, maxRunes))
            t = tailRunes(t, maxRunes)
        }
        if strings.TrimSpace(t) != "" {
            out = append(out, t)
        }
    }
    return out
}

func tailRunes(s string, skip int) string {
    n := 0
    for i := range s {
        if n == skip {
            return strings.TrimSpace(s[i:])
        }
        n++
    }
    return ""
}
