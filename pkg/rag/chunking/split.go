package chunking

import (
    "regexp"
    "strings"
    "unicode/utf8"
)

var sentenceBoundary = regexp.MustCompile(`(?m)(?:[.!?…]+\s+|\n{2,})`)

// SplitSentences делит текст на предложения/абзацы для семантического чанкинга.
func SplitSentences(text string) []string {
    text = strings.TrimSpace(text)
    if text == "" {
        return nil
    }
    parts := sentenceBoundary.Split(text, -1)
    out := make([]string, 0, len(parts))
    for _, p := range parts {
        p = strings.TrimSpace(p)
        if p == "" {
            continue
        }
        out = append(out, p)
    }
    if len(out) == 0 {
        return []string{text}
    }
    return out
}

// MergeByCharLimit объединяет предложения в чанки с ограничением по символам.
// overlapRunes — перекрытие с предыдущим чанком (0 — без перекрытия).
func MergeByCharLimit(sentences []string, maxChars, minChars, overlapRunes int) []string {
    if maxChars <= 0 {
        maxChars = 1500
    }
    if minChars <= 0 {
        minChars = 80
    }
    var chunks []string
    var cur strings.Builder
    flush := func() {
        s := strings.TrimSpace(cur.String())
        cur.Reset()
        if s != "" {
            chunks = append(chunks, s)
        }
    }
    for _, sent := range sentences {
        if cur.Len() > 0 && cur.Len()+1+len(sent) > maxChars {
            flush()
        }
        if cur.Len() > 0 {
            cur.WriteByte(' ')
        }
        cur.WriteString(sent)
    }
    flush()
    if len(chunks) == 0 {
        return nil
    }
    // склеить слишком короткий хвост с предыдущим
    if len(chunks) > 1 && utf8.RuneCountInString(chunks[len(chunks)-1]) < minChars {
        chunks[len(chunks)-2] += " " + chunks[len(chunks)-1]
        chunks = chunks[:len(chunks)-1]
    }
    return ApplyOverlap(chunks, maxChars, overlapRunes)
}
