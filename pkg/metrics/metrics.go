// Пакет metrics — метрики Prometheus для инференса и HTTP API.
package metrics

import (
    "strconv"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

const namespace = "gpt_api"

// go_* и process_* уже регистрируются в init() пакета prometheus (client_golang ≥1.20).

var (
    HTTPRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
        Namespace: namespace,
        Name:      "http_requests_total",
        Help:      "Число HTTP-запросов к API.",
    }, []string{"method", "path", "status"})

    HTTPRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
        Namespace: namespace,
        Name:      "http_request_duration_seconds",
        Help:      "Длительность HTTP-запросов.",
        Buckets:   []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300},
    }, []string{"method", "path"})

    InferenceInFlight = promauto.NewGaugeVec(prometheus.GaugeOpts{
        Namespace: namespace,
        Name:      "inference_in_flight",
        Help:      "Текущее число активных запросов инференса.",
    }, []string{"endpoint"})

    ChatCompletionsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
        Namespace: namespace,
        Name:      "chat_completions_total",
        Help:      "Запросы chat completions.",
    }, []string{"model", "stream", "status"})

    ChatPromptTokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
        Namespace: namespace,
        Name:      "chat_prompt_tokens_total",
        Help:      "Суммарное число токенов промпта (chat).",
    }, []string{"model"})

    ChatCompletionTokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
        Namespace: namespace,
        Name:      "chat_completion_tokens_total",
        Help:      "Суммарное число сгенерированных токенов (chat).",
    }, []string{"model"})

    ChatPredictDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
        Namespace: namespace,
        Name:      "chat_predict_duration_seconds",
        Help:      "Время генерации ответа (llama Predict), без загрузки модели.",
        Buckets:   []float64{0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600},
    }, []string{"model", "stream"})

    ChatTokensPerSecond = promauto.NewHistogramVec(prometheus.HistogramOpts{
        Namespace: namespace,
        Name:      "chat_tokens_per_second",
        Help:      "Скорость генерации: completion_tokens / predict_duration.",
        Buckets:   []float64{0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300},
    }, []string{"model", "stream"})

    ChatModelResolveDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
        Namespace: namespace,
        Name:      "chat_model_resolve_duration_seconds",
        Help:      "Время получения/подгрузки модели перед predict.",
        Buckets:   []float64{0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300},
    }, []string{"model"})

    EmbeddingsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
        Namespace: namespace,
        Name:      "embeddings_total",
        Help:      "Запросы embeddings.",
    }, []string{"model", "status"})

    EmbeddingsDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
        Namespace: namespace,
        Name:      "embeddings_duration_seconds",
        Help:      "Время расчёта embedding.",
        Buckets:   []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30},
    }, []string{"model"})

    EmbeddingsInputTokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
        Namespace: namespace,
        Name:      "embeddings_input_tokens_total",
        Help:      "Токены входного текста для embeddings.",
    }, []string{"model"})

    ModelLoaded = promauto.NewGaugeVec(prometheus.GaugeOpts{
        Namespace: namespace,
        Name:      "model_loaded",
        Help:      "1 — модель сейчас в VRAM/RAM, 0 — выгружена.",
    }, []string{"model", "mode"})

    ModelLoadsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
        Namespace: namespace,
        Name:      "model_loads_total",
        Help:      "Попытки загрузки модели.",
    }, []string{"model", "mode", "status"})

    ModelLoadDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
        Namespace: namespace,
        Name:      "model_load_duration_seconds",
        Help:      "Время загрузки весов модели.",
        Buckets:   []float64{0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600},
    }, []string{"model", "mode"})
)

func ObserveHTTP(method, path string, status int, dur time.Duration) {
    code := strconv.Itoa(status)
    HTTPRequestsTotal.WithLabelValues(method, path, code).Inc()
    HTTPRequestDuration.WithLabelValues(method, path).Observe(dur.Seconds())
}

func BeginInference(endpoint string) func() {
    InferenceInFlight.WithLabelValues(endpoint).Inc()
    return func() {
        InferenceInFlight.WithLabelValues(endpoint).Dec()
    }
}

func ObserveChat(model string, stream bool, status string, promptTokens, completionTokens int, resolveDur, predictDur time.Duration) {
    streamLabel := strconv.FormatBool(stream)
    ChatCompletionsTotal.WithLabelValues(model, streamLabel, status).Inc()
    if resolveDur > 0 {
        ChatModelResolveDuration.WithLabelValues(model).Observe(resolveDur.Seconds())
    }
    if status != "success" {
        return
    }
    if promptTokens > 0 {
        ChatPromptTokensTotal.WithLabelValues(model).Add(float64(promptTokens))
    }
    if completionTokens > 0 {
        ChatCompletionTokensTotal.WithLabelValues(model).Add(float64(completionTokens))
    }
    if predictDur > 0 {
        ChatPredictDuration.WithLabelValues(model, streamLabel).Observe(predictDur.Seconds())
        if completionTokens > 0 {
            tps := float64(completionTokens) / predictDur.Seconds()
            ChatTokensPerSecond.WithLabelValues(model, streamLabel).Observe(tps)
        }
    }
}

func ObserveEmbeddings(model, status string, inputTokens int, dur time.Duration) {
    EmbeddingsTotal.WithLabelValues(model, status).Inc()
    if status != "success" {
        return
    }
    if dur > 0 {
        EmbeddingsDuration.WithLabelValues(model).Observe(dur.Seconds())
    }
    if inputTokens > 0 {
        EmbeddingsInputTokensTotal.WithLabelValues(model).Add(float64(inputTokens))
    }
}

func SetModelLoaded(model, mode string, loaded bool) {
    v := 0.0
    if loaded {
        v = 1
    }
    ModelLoaded.WithLabelValues(model, mode).Set(v)
}

func ObserveModelLoad(model, mode, status string, dur time.Duration) {
    ModelLoadsTotal.WithLabelValues(model, mode, status).Inc()
    if status == "success" && dur > 0 {
        ModelLoadDuration.WithLabelValues(model, mode).Observe(dur.Seconds())
    }
}
