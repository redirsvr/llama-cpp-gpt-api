package gpt

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"time"

	"llama-cpp-gpt-api/internal/config"
	"llama-cpp-gpt-api/internal/types"
	"llama-cpp-gpt-api/pkg/metrics"
	"llama-cpp-gpt-api/pkg/model"
	"llama-cpp-gpt-api/pkg/rag"

	llama "github.com/redirsvr/go-llama-new.cpp"
)

type ChatCompletionsLogic struct {
	ctx context.Context
	w   http.ResponseWriter
	r   *http.Request
}

func NewChatCompletionsLogic(ctx context.Context, w http.ResponseWriter, r *http.Request) *ChatCompletionsLogic {
	return &ChatCompletionsLogic{ctx: ctx, w: w, r: r}
}

func (l *ChatCompletionsLogic) ChatCompletions(req *types.ReqChatCompletion) (resp *types.ResChatCompletion, err error) {
	if serviceResp := l.openWebUIServiceResponse(req); serviceResp != nil {
		return serviceResp, nil
	}

	// RAG до UseChat: иначе mutex реестра занят chat-моделью и embedding не загрузится (deadlock).
	l.attachRAGContext(req)

	var out *types.ResChatCompletion
	err = model.UseChat(req.Model, func(ll *llama.LLama, modelAlias string) error {
		var runErr error
		out, runErr = l.runChat(ll, modelAlias, req)
		return runErr
	})
	return out, err
}

func (l *ChatCompletionsLogic) openWebUIServiceResponse(req *types.ReqChatCompletion) *types.ResChatCompletion {
	text := req.Prompt
	if len(req.Messages) > 0 {
		text = lastUserContent(req.Messages)
	}
	kind := rag.OpenWebUIServiceTask(text)
	if kind == "" {
		return nil
	}

	content := `{"ok":true}`
	switch kind {
	case "follow_ups":
		content = `{"follow_ups":[]}`
	case "title":
		content = `{"title":"RAG запрос"}`
	case "tags":
		content = `{"tags":["Technology","RAG"]}`
	}
	log.Printf("Open WebUI: служебный запрос %q обработан без llama_predict", kind)
	return &types.ResChatCompletion{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Model:   req.Model,
		Created: int(time.Now().Unix()),
		Choices: []types.Choice{
			{
				Message: types.Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
				Index:        0,
			},
		},
		Usage: types.Usage{},
	}
}

func (l *ChatCompletionsLogic) attachRAGContext(req *types.ReqChatCompletion) {
	if !rag.UseInChatCompletions(req.UseRAG) {
		return
	}
	query := rag.ExtractSearchQuery(req.Messages, req.Prompt)
	if query == "" {
		log.Printf("RAG chat: пропуск RAG (нет вопроса или служебный запрос Open WebUI)")
		return
	}
	ctxText, hits, err := rag.FormatPromptForRAGWithHits(l.ctx, query, req.RAGTopK)
	if err != nil {
		log.Printf("RAG chat: поиск не удался: %v", err)
		return
	}
	if ctxText == "" {
		log.Printf("RAG chat: по запросу %q совпадений в базе нет", truncateLog(query, 120))
		return
	}
	log.Printf("RAG chat: запрос %q → %d фрагмент(ов), контекст %d символов",
		truncateLog(query, 80), len(hits), len(ctxText))
	req.Messages = rag.ApplyRAGToMessages(req.Messages, ctxText)
	req.RAGUsed = len(hits)
	req.RAGTitles = rag.HitTitles(hits)
	req.RAGFragments = rag.HitFragments(hits)
}

func (l *ChatCompletionsLogic) runChat(ll *llama.LLama, modelAlias string, req *types.ReqChatCompletion) (resp *types.ResChatCompletion, err error) {
	endInference := metrics.BeginInference("chat")
	defer endInference()

	stream := req.Stream
	status := "success"
	var promptTokens, completionTokens int
	var resolveDur, predictDur time.Duration

	defer func() {
		metrics.ObserveChat(modelAlias, stream, status, promptTokens, completionTokens, resolveDur, predictDur)
	}()

	w := l.w
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	}

	text := req.Prompt
	if len(req.Messages) > 0 {
		text = model.FormatChatPrompt(modelAlias, req.Messages)
	}

	if req.RAGUsed > 0 && !rag.HasRAGInPrompt(text) {
		log.Printf("RAG chat: ВНИМАНИЕ — %d чанков найдено, но «Фрагменты документов» нет в промпте", req.RAGUsed)
	} else if req.RAGUsed > 0 {
		log.Printf("RAG chat: контекст в промпте OK (%d чанков)", req.RAGUsed)
	}

	log.Print("start predict [", modelAlias, "]:\n", text)

	flusher, flusherOk := w.(http.Flusher)
	if req.Stream && !flusherOk {
		http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
		status = "error"
		return nil, nil
	}
	if req.Stream {
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = config.C.DefaultMaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	resolveStart := time.Now()
	promptTokens = model.CountTokens(ll, text)
	resolveDur = time.Since(resolveStart)
	if promptTokens > 0 {
		ctxSize := chatContextSize()
		available := ctxSize - promptTokens - 4
		if available <= 0 {
			status = "error"
			return nil, fmt.Errorf("prompt too long: %d tokens, context %d", promptTokens, ctxSize)
		}
		if maxTokens > available {
			log.Printf("predict: max_tokens reduced %d -> %d (prompt=%d context=%d)", maxTokens, available, promptTokens, ctxSize)
			maxTokens = available
		}
	}
	log.Printf("predict params [%s]: prompt_tokens=%d max_tokens=%d context=%d batch=%d stream=%v",
		modelAlias, promptTokens, maxTokens, chatContextSize(), chatBatchSize(), req.Stream)

	var streamMu sync.Mutex
	var streamWriteErr error
	writeStream := func(payload string) bool {
		if !req.Stream {
			return true
		}
		streamMu.Lock()
		defer streamMu.Unlock()
		if streamWriteErr != nil {
			return false
		}
		select {
		case <-l.ctx.Done():
			streamWriteErr = l.ctx.Err()
			return false
		default:
		}
		if _, err := fmt.Fprint(w, payload); err != nil {
			streamWriteErr = err
			return false
		}
		flusher.Flush()
		return true
	}
	getStreamErr := func() error {
		streamMu.Lock()
		defer streamMu.Unlock()
		return streamWriteErr
	}

	predict := func() (string, error) {
		return ll.Predict(text, func(p *llama.PredictOptions) {
			p.Tokens = maxTokens
			p.Batch = chatBatchSize()
			p.Temperature = 0.7
			p.TopP = 0.9
			p.Penalty = 1.15
			p.Repeat = 64
			if req.Temperature > 0 {
				p.Temperature = req.Temperature
			}
			if req.TopP > 0 {
				p.TopP = req.TopP
			}
			if req.Seed > 0 {
				p.Seed = req.Seed
			}
			p.StopPrompts = model.QwenStopWords()
			if req.Stream {
				id := 0
				p.TokenCallback = func(token string) bool {
					data := &types.ResChatCompletion{
						ID: fmt.Sprint(id),
						Choices: []types.Choice{
							{
								Delta: &types.Message{
									Role:    "assistant",
									Content: token,
								},
								Index: 0,
							},
						},
					}
					chunk, err := json.Marshal(data)
					if err != nil {
						return false
					}
					id++
					return writeStream("data: " + string(chunk) + "\n\n")
				}
			}
			p.Threads = runtime.NumCPU()
		})
	}

	predictStart := time.Now()
	var result string
	if req.Stream {
		type predictResult struct {
			text string
			err  error
		}
		done := make(chan predictResult, 1)
		go func() {
			text, err := predict()
			done <- predictResult{text: text, err: err}
		}()

		ticker := time.NewTicker(streamHeartbeatInterval())
		defer ticker.Stop()
		for {
			select {
			case r := <-done:
				result, err = r.text, r.err
				goto predictDone
			case <-ticker.C:
				writeStream(": ping\n\n")
			}
		}
	} else {
		result, err = predict()
	}

predictDone:
	predictDur = time.Since(predictStart)
	if req.Stream && getStreamErr() != nil {
		status = "cancelled"
		log.Printf("stream closed [%s]: %v", modelAlias, getStreamErr())
		return nil, nil
	}
	if err != nil {
		status = "error"
		if req.Stream {
			writeStreamError(w, flusher, err)
			return nil, nil
		}
		return nil, err
	}

	rawResult := result
	result = model.CleanAssistantReply(result)
	completionTokens = model.CountTokens(ll, rawResult)
	if completionTokens == 0 && strings.TrimSpace(result) != "" {
		completionTokens = model.CountTokens(ll, result)
	}

	if strings.TrimSpace(result) == "" && strings.TrimSpace(rawResult) != "" {
		log.Printf("warn [%s]: ответ пустой после очистки, raw len=%d: %.200q", modelAlias, len(rawResult), rawResult)
	} else if strings.TrimSpace(result) == "" {
		log.Printf("warn [%s]: модель вернула пустой ответ", modelAlias)
	}
	log.Print("end predict [", modelAlias, "] len=", len(result),
		" tokens=", promptTokens, "+", completionTokens,
		" tps=", tokensPerSecond(completionTokens, predictDur))

	usage := types.Usage{
		PromptTokens:    promptTokens,
		CompletionToken: completionTokens,
		TotalTokens:     promptTokens + completionTokens,
	}

	if !req.Stream {
		res := &types.ResChatCompletion{
			ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
			Model:   modelAlias,
			Created: int(time.Now().Unix()),
			Choices: []types.Choice{
				{
					Message: types.Message{
						Role:    "assistant",
						Content: result,
					},
					FinishReason: "stop",
					Index:        0,
				},
			},
			Usage: usage,
		}
		if req.RAGUsed > 0 {
			res.RAG = &types.RAGInfo{
				ChunksUsed: req.RAGUsed,
				Titles:     req.RAGTitles,
				Fragments:  req.RAGFragments,
			}
		}
		return res, nil
	}

	writeStream("data: [DONE]\n\n")
	return nil, nil
}

func writeStreamError(w http.ResponseWriter, flusher http.Flusher, err error) {
	payload := map[string]any{
		"error": map[string]any{
			"message": err.Error(),
			"type":    "server_error",
		},
	}
	if b, jsonErr := json.Marshal(payload); jsonErr == nil {
		fmt.Fprint(w, "data: "+string(b)+"\n\n")
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func lastUserContent(messages []types.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return strings.TrimSpace(messages[i].Content)
		}
	}
	return ""
}

func truncateLog(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func tokensPerSecond(tokens int, dur time.Duration) float64 {
	if tokens <= 0 || dur <= 0 {
		return 0
	}
	return float64(tokens) / dur.Seconds()
}

func chatContextSize() int {
	const fallback = 32768
	if n := modelOptionInt("ContextSize"); n > 0 {
		return n
	}
	return fallback
}

func chatBatchSize() int {
	const fallback = 32
	if n := modelOptionInt("NBatch"); n > 0 {
		return n
	}
	return fallback
}

func streamHeartbeatInterval() time.Duration {
	const fallback = 5 * time.Second
	if config.C.Timeout > 0 {
		d := time.Duration(config.C.Timeout) * time.Millisecond / 4
		if d >= time.Second && d < fallback {
			return d
		}
	}
	return fallback
}

func modelOptionInt(key string) int {
	if config.C.ModelOption == nil {
		return 0
	}
	v, ok := config.C.ModelOption[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case int:
		if n > 0 {
			return n
		}
	case int64:
		if n > 0 {
			return int(n)
		}
	case float64:
		if n > 0 {
			return int(n)
		}
	}
	return 0
}
