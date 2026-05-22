package chunking

import (
	"strings"
	"unicode/utf8"
)

// clampOverlap ограничивает перекрытие разумной долей размера чанка.
func clampOverlap(maxChars, overlap int) int {
	if overlap <= 0 || maxChars <= 0 {
		return 0
	}
	if overlap >= maxChars {
		return maxChars / 4
	}
	return overlap
}

// tailRunes возвращает последние n рун строки.
func tailRunes(s string, n int) string {
	if n <= 0 || s == "" {
		return ""
	}
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[len(runes)-n:])
}

// ApplyOverlap добавляет хвост предыдущего чанка в начало каждого следующего.
// Первый чанк без изменений — для явной цепочки контекста при поиске.
func ApplyOverlap(chunks []string, maxChars, overlapRunes int) []string {
	overlapRunes = clampOverlap(maxChars, overlapRunes)
	if overlapRunes <= 0 || len(chunks) <= 1 {
		return chunks
	}
	out := make([]string, len(chunks))
	out[0] = chunks[0]
	for i := 1; i < len(chunks); i++ {
		prefix := tailRunes(chunks[i-1], overlapRunes)
		cur := chunks[i]
		if prefix == "" || strings.HasPrefix(cur, prefix) {
			out[i] = cur
			continue
		}
		out[i] = prefix + " " + cur
	}
	return out
}

// applyOverlapToFileChunks перекрытие для потоковых чанков + коррекция char_start.
func applyOverlapToFileChunks(chunks []Chunk, maxChars, overlapRunes int) {
	overlapRunes = clampOverlap(maxChars, overlapRunes)
	if overlapRunes <= 0 || len(chunks) <= 1 {
		return
	}
	for i := 1; i < len(chunks); i++ {
		prefix := tailRunes(chunks[i-1].Content, overlapRunes)
		if prefix == "" || strings.HasPrefix(chunks[i].Content, prefix) {
			continue
		}
		overlapLen := utf8.RuneCountInString(prefix)
		chunks[i].Content = prefix + " " + chunks[i].Content
		if cs, ok := chunks[i].Meta["char_start"].(int); ok {
			chunks[i].Meta["char_start"] = cs - overlapLen
			if chunks[i].Meta["char_start"].(int) < 0 {
				chunks[i].Meta["char_start"] = 0
			}
		}
		chunks[i].Meta["overlap_runes"] = overlapLen
	}
}
