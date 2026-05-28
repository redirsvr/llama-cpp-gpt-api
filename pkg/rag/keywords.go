package rag

import (
	"sort"
	"strings"
	"unicode"

	"llama-cpp-gpt-api/pkg/rag/store"
)

var stopWords = map[string]struct{}{
	"и": {}, "в": {}, "во": {}, "не": {}, "что": {}, "он": {}, "на": {},
	"я": {}, "с": {}, "со": {}, "как": {}, "а": {}, "то": {}, "все": {},
	"она": {}, "так": {}, "его": {}, "но": {}, "да": {}, "ты": {},
	"к": {}, "у": {}, "же": {}, "вы": {}, "за": {}, "бы": {}, "по": {},
	"из": {}, "ему": {}, "от": {}, "о": {}, "для": {}, "или": {},
	"это": {}, "они": {}, "её": {}, "нее": {}, "них": {}, "нам": {},
	"вам": {}, "вас": {}, "нас": {}, "меня": {}, "себя": {},
	"чем": {}, "том": {}, "тем": {}, "кто": {}, "где": {}, "когда": {},
	"чтобы": {}, "можно": {}, "при": {}, "без": {}, "еще": {}, "уже": {},
	"нет": {}, "даже": {}, "будет": {}, "быть": {}, "есть": {}, "всего": {},
	"если": {}, "потому": {}, "тоже": {}, "мочь": {}, "который": {}, "свой": {},
	"через": {}, "после": {}, "этой": {}, "этот": {}, "этого": {}, "эти": {},
	"the": {}, "a": {}, "an": {}, "is": {}, "are": {}, "was": {},
	"were": {}, "be": {}, "been": {}, "being": {}, "have": {}, "has": {},
	"had": {}, "do": {}, "does": {}, "did": {}, "will": {}, "would": {},
	"could": {}, "should": {}, "may": {}, "might": {}, "shall": {},
	"can": {}, "need": {}, "this": {}, "that": {}, "these": {}, "those": {},
	"i": {}, "me": {}, "my": {}, "we": {}, "our": {}, "you": {},
	"your": {}, "he": {}, "him": {}, "his": {}, "she": {}, "her": {},
	"it": {}, "its": {}, "they": {}, "them": {}, "their": {},
	"what": {}, "which": {}, "who": {}, "whom": {}, "when": {}, "where": {},
	"why": {}, "how": {}, "all": {}, "each": {}, "every": {}, "both": {},
	"few": {}, "more": {}, "most": {}, "other": {}, "some": {}, "such": {},
	"no": {}, "nor": {}, "not": {}, "only": {}, "own": {}, "same": {},
	"so": {}, "than": {}, "too": {}, "very": {}, "just": {},
	"as": {}, "until": {}, "while": {}, "of": {}, "at": {}, "by": {},
	"for": {}, "with": {}, "about": {}, "between": {}, "into": {},
	"through": {}, "during": {}, "before": {}, "after": {}, "above": {},
	"below": {}, "to": {}, "from": {}, "up": {}, "down": {}, "in": {},
	"out": {}, "on": {}, "off": {}, "over": {}, "under": {},
	"again": {}, "then": {}, "once": {}, "here": {}, "there": {},
	"and": {}, "but": {}, "or": {}, "if": {}, "because": {},
}

// ExtractKeywords возвращает до max ключевых слов из текста (TF-based).
// Аналог entity descriptions из RAG-Anything — выделяем значимые сущности из чанка.
func ExtractKeywords(text string, max int) []string {
	if max <= 0 {
		max = 10
	}
	if len(text) > 10000 {
		text = text[:10000]
	}
	words := tokenizeWords(text)
	if len(words) == 0 {
		return nil
	}
	freq := make(map[string]int)
	for _, w := range words {
		if len(w) < 3 {
			continue
		}
		low := strings.ToLower(w)
		if _, isStop := stopWords[low]; isStop {
			continue
		}
		if isPureNumber(low) {
			continue
		}
		freq[low]++
	}
	if len(freq) == 0 {
		return nil
	}
	ranked := rankByFreq(freq)
	out := make([]string, 0, max)
	for _, kv := range ranked {
		out = append(out, kv.k)
		if len(out) >= max {
			break
		}
	}
	return out
}

type freqEntry struct {
	k string
	v int
}

func rankByFreq(freq map[string]int) []freqEntry {
	out := make([]freqEntry, 0, len(freq))
	for k, v := range freq {
		out = append(out, freqEntry{k, v})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].v != out[j].v {
			return out[i].v > out[j].v
		}
		return out[i].k < out[j].k
	})
	return out
}

// keywordOverlap считает долю ключевых слов запроса, найденных в ключевых словах чанка.
func keywordOverlap(queryKeywords, chunkKeywords []string) float64 {
	if len(queryKeywords) == 0 || len(chunkKeywords) == 0 {
		return 0
	}
	ckSet := make(map[string]struct{}, len(chunkKeywords))
	for _, k := range chunkKeywords {
		ckSet[strings.ToLower(k)] = struct{}{}
	}
	matched := 0
	for _, qk := range queryKeywords {
		if _, ok := ckSet[strings.ToLower(qk)]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(queryKeywords))
}

// ExtractQueryKeywords извлекает ключевые слова из поискового запроса.
func ExtractQueryKeywords(query string) []string {
	return ExtractKeywords(query, 8)
}

// keywordBoost применяет бустинг чанков по совпадению ключевых слов.
// Эквивалент entity alignment из RAG-Anything — повышаем релевантность чанков,
// чьи сущности совпадают с сущностями запроса.
func keywordBoost(query string, hits []store.ChunkResult) {
	qk := ExtractQueryKeywords(query)
	if len(qk) == 0 {
		return
	}
	for i := range hits {
		kws := extractKeywordsFromMeta(hits[i].Metadata)
		if len(kws) == 0 {
			continue
		}
		overlap := keywordOverlap(qk, kws)
		if overlap > 0 {
			boost := 1.0 + overlap*0.5
			hits[i].Score *= boost
		}
	}
}

func extractKeywordsFromMeta(meta map[string]any) []string {
	if meta == nil {
		return nil
	}
	raw, ok := meta["keywords"]
	if !ok {
		return nil
	}
	list, ok := raw.([]any)
	if !ok {
		if strs, ok := raw.([]string); ok {
			return strs
		}
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func tokenizeWords(text string) []string {
	var words []string
	var cur []rune
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' {
			cur = append(cur, r)
		} else {
			if len(cur) > 0 {
				words = append(words, string(cur))
				cur = cur[:0]
			}
		}
	}
	if len(cur) > 0 {
		words = append(words, string(cur))
	}
	return words
}

func isPureNumber(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) && r != '.' && r != '-' && r != ',' {
			return false
		}
	}
	return true
}
