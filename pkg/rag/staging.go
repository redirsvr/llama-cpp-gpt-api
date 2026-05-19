package rag

import (
    "crypto/rand"
    "encoding/hex"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "strings"

    "llama-cpp-gpt-api/internal/config"
)

// SaveUpload сохраняет загруженный файл на диск и возвращает путь.
func SaveUpload(r io.Reader, originalName string) (string, error) {
    dir, err := stagingDir()
    if err != nil {
        return "", err
    }
    if err := os.MkdirAll(dir, 0o755); err != nil {
        return "", fmt.Errorf("каталог staging: %w", err)
    }

    base := sanitizeFilename(originalName)
    if base == "" {
        base = "upload"
    }
    id := randomID()
    path := filepath.Join(dir, id+"_"+base)

    out, err := os.Create(path)
    if err != nil {
        return "", err
    }
    defer out.Close()

    if _, err := io.Copy(out, r); err != nil {
        _ = os.Remove(path)
        return "", err
    }
    return path, nil
}

func stagingDir() (string, error) {
    d := strings.TrimSpace(config.C.RAG.StagingDir)
    if d == "" {
        d = "data/rag-staging"
    }
    return filepath.Abs(d)
}

func sanitizeFilename(name string) string {
    name = filepath.Base(name)
    var b strings.Builder
    for _, r := range name {
        if r == '/' || r == '\\' || r == 0 {
            continue
        }
        b.WriteRune(r)
    }
    s := b.String()
    if len(s) > 200 {
        s = s[:200]
    }
    return s
}

func randomID() string {
    var buf [8]byte
    _, _ = rand.Read(buf[:])
    return hex.EncodeToString(buf[:])
}

// RemoveStaging удаляет файл staging (игнорирует ошибку).
func RemoveStaging(path string) {
    if path == "" || !config.C.RAG.RemoveStagingAfterIngest {
        return
    }
    _ = os.Remove(path)
}
