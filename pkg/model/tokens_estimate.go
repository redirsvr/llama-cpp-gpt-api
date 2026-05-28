package model

import "unicode/utf8"

// EstimatePromptTokens — быстрая оценка длины промпта без полной токенизации (~3 rune/token).
func EstimatePromptTokens(text string) int {
	n := utf8.RuneCountInString(text)
	if n == 0 {
		return 0
	}
	est := n / 3
	if est < 1 {
		return 1
	}
	return est
}
