package model

import llama "github.com/redirsvr/go-llama-new.cpp"

// CountTokens возвращает число токенов для текста (0 при ошибке токенизации).
func CountTokens(ll *llama.LLama, text string) int {
    if ll == nil || text == "" {
        return 0
    }
    n, _, err := ll.TokenizeString(text)
    if err != nil || n <= 0 {
        return 0
    }
    return int(n)
}
