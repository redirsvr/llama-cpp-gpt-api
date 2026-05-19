package api

// OpenAIError — тело ошибки в формате OpenAI API.
func OpenAIError(err error, code string) map[string]any {
    return map[string]any{
        "error": map[string]any{
            "message": err.Error(),
            "type":    "invalid_request_error",
            "param":   nil,
            "code":    code,
        },
    }
}
