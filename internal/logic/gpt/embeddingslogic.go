package gpt

import (
    "context"
    "net/http"
    "time"

    "llama-cpp-gpt-api/internal/types"
    "llama-cpp-gpt-api/pkg/metrics"
    "llama-cpp-gpt-api/pkg/model"

    llama "github.com/redirsvr/go-llama-new.cpp"
)

type EmbeddingsLogic struct {
    ctx context.Context
    w   http.ResponseWriter
    r   *http.Request
}

func NewEmbeddingsLogic(ctx context.Context, w http.ResponseWriter, r *http.Request) *EmbeddingsLogic {
    return &EmbeddingsLogic{ctx: ctx, w: w, r: r}
}

func (l *EmbeddingsLogic) Embeddings(req *types.ReqEmbeddings) (resp *types.ResEmbeddings, err error) {
    var out *types.ResEmbeddings
    err = model.UseEmbeddings(req.Model, func(ll *llama.LLama, modelAlias string) error {
        var runErr error
        out, runErr = l.runEmbeddings(ll, modelAlias, req)
        return runErr
    })
    return out, err
}

func (l *EmbeddingsLogic) runEmbeddings(ll *llama.LLama, modelAlias string, req *types.ReqEmbeddings) (*types.ResEmbeddings, error) {
    endInference := metrics.BeginInference("embeddings")
    defer endInference()

    status := "success"
    var inputTokens int
    var dur time.Duration

    defer func() {
        metrics.ObserveEmbeddings(modelAlias, status, inputTokens, dur)
    }()

    inputTokens = model.CountTokens(ll, req.Input)

    embedStart := time.Now()
    embeds, err := model.EmbedString(ll, req.Input)
    dur = time.Since(embedStart)
    if err != nil {
        status = "error"
        return nil, err
    }

    return &types.ResEmbeddings{
        Model: modelAlias,
        Data: []types.Embedding{
            {
                Embedding: embeds,
                Index:     0,
            },
        },
    }, nil
}
