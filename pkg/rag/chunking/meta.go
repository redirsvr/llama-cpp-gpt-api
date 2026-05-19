package chunking

import (
	"strings"
	"unicode/utf8"

	"llama-cpp-gpt-api/pkg/rag/store"
)

// ChunksWithMeta строит ChunkInput из строк семантической нарезки с char_start/char_end в исходном тексте.
func ChunksWithMeta(content string, parts []string, sourcePath string, docMeta map[string]any) []store.ChunkInput {
	out := make([]store.ChunkInput, 0, len(parts))
	pos := 0
	for _, ch := range parts {
		ch = strings.TrimSpace(ch)
		if ch == "" {
			continue
		}
		meta := map[string]any{}
		if idx := strings.Index(content[pos:], ch); idx >= 0 {
			startByte := pos + idx
			charStart := utf8.RuneCountInString(content[:startByte])
			meta["char_start"] = charStart
			meta["char_end"] = charStart + utf8.RuneCountInString(ch)
			pos = startByte + len(ch)
		}
		if page := pageAtRune(content, metaInt(meta, "char_start")); page > 0 {
			meta["page"] = page
		}
		idx := len(out)
		out = append(out, store.ChunkInput{
			Content:  ch,
			Metadata: store.MergeChunkMeta(idx, sourcePath, docMeta, meta),
		})
	}
	return out
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

// pageAtRune возвращает номер страницы по маркерам <<<PAGE:N>>> до позиции runePos.
func pageAtRune(content string, runePos int) int {
	if runePos <= 0 {
		return 0
	}
	page := 0
	pos := 0
	for _, line := range strings.Split(content, "\n") {
		if p, ok := ParsePageMarker(line); ok {
			page = p
			pos += utf8.RuneCountInString(line) + 1
			continue
		}
		lineRunes := utf8.RuneCountInString(line) + 1
		if pos+lineRunes > runePos && page > 0 {
			return page
		}
		pos += lineRunes
	}
	if page > 0 {
		return page
	}
	return 0
}
