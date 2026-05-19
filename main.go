// @title           llama-cpp-gpt-api
// @version         1.0
// @description     OpenAI-совместимый HTTP API (chat, embeddings, models) и RAG поверх PostgreSQL/pgvector.
// @termsOfService  http://swagger.io/terms/

// @contact.name   API Support

// @license.name  MIT

// @host      localhost:8080
// @BasePath  /

// @schemes   http

//go:generate swag init -g main.go -o docs --parseDependency --parseInternal

package main

import (
    "context"
    "flag"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "llama-cpp-gpt-api/internal/api"
    "llama-cpp-gpt-api/internal/config"
    "llama-cpp-gpt-api/pkg/model"
    "llama-cpp-gpt-api/pkg/rag"
)

var configFile = flag.String("f", "etc/gpt-api.yaml", "the config file")

func main() {
    flag.Parse()

    if err := config.Load(*configFile); err != nil {
        log.Fatal("конфиг: ", err)
    }
    if err := model.Init(); err != nil {
        log.Fatal("инициализация моделей: ", err)
    }
    log.Println("Модели готовы")

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    if config.C.RAG.Enabled {
        if err := rag.Init(ctx); err != nil {
            log.Fatal("RAG: ", err)
        }
        defer rag.Close()
        rag.StartAutoIngestWatcher(ctx)
    }

    router := api.NewRouter()

    timeout := time.Duration(config.C.Timeout) * time.Millisecond
    if timeout <= 0 {
        timeout = 3 * time.Minute
    }
    readTimeout := timeout
    writeTimeout := timeout
    if config.C.RAG.Enabled && config.C.RAG.IngestTimeoutSec > 0 {
        ingestTO := time.Duration(config.C.RAG.IngestTimeoutSec) * time.Second
        if ingestTO > writeTimeout {
            writeTimeout = ingestTO
        }
        if ingestTO > readTimeout {
            readTimeout = ingestTO
        }
    }

    addr := fmt.Sprintf("%s:%d", config.C.Host, config.C.Port)
    srv := &http.Server{
        Addr:         addr,
        Handler:      router,
        ReadTimeout:  readTimeout,
        WriteTimeout: writeTimeout,
    }

    go func() {
        sig := make(chan os.Signal, 1)
        signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
        <-sig
        cancel()
        _ = srv.Close()
    }()

    log.Printf("Сервер Gin: http://%s", addr)
    if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
        log.Fatal(err)
    }
}
