package model

// EstimatePromptTokens — быстрая оценка длины промпта без полной токенизации (~3 rune/token).
func EstimatePromptTokens(text string) int {
	// Count runes without allocating
	count := 0
	for range text {
		count++
	}
	if count == 0 {
		return 0
	}
	est := count / 3
	if est < 1 {
		return 1
	}
	return est
}
