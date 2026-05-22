package model

import llama "github.com/redirsvr/go-llama-new.cpp"

// CountTokens возвращает число токенов для текста (0 при ошибке токенизации).
func CountTokens(ll *llama.LLama, text string) int {
	if ll == nil || text == "" {
		return 0
	}
	maxTokens := 32768
	if n, ok := modelOptionInt("ContextSize"); ok && n > 0 {
		maxTokens = n
	}
	n, _, err := ll.TokenizeString(text, llama.SetTokens(maxTokens))
	if err != nil || n <= 0 {
		return 0
	}
	return int(n)
}
