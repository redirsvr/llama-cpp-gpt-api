package gpt

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"strings"
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
		w.Header().Set("Connection", "keep-alive")
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

	predictStart := time.Now()
	result, err := ll.Predict(text, func(p *llama.PredictOptions) {
		p.Tokens = maxTokens
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
				fmt.Fprint(w, "data:"+string(chunk)+"\n\n")
				flusher.Flush()
				return true
			}
		}
		p.Threads = runtime.NumCPU()
	})
	predictDur = time.Since(predictStart)
	if err != nil {
		status = "error"
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

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	return nil, nil
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
