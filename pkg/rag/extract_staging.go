package rag

import (
    "fmt"
    "os"
    "path/filepath"
    "strings"

    "llama-cpp-gpt-api/pkg/rag/chunking"
)

type extractWriter func(srcPath, outPath string) error

// PrepareTextPath при необходимости конвертирует бинарные форматы во временный .txt в staging.
func PrepareTextPath(src string) (workPath string, cleanup func(), err error) {
    src = filepath.Clean(src)
    cleanup = func() {}

    ext := strings.ToLower(filepath.Ext(src))
    fn, ok := extractors[ext]
    if !ok {
        return src, cleanup, nil
    }
    if fn == nil {
        if ext == ".doc" {
            return "", nil, fmt.Errorf("DOC: старый Word (.doc) не поддерживается; сохраните как .docx или .txt")
        }
        return "", nil, fmt.Errorf("%s: формат не поддерживается; сохраните как .docx, .pdf или .txt", strings.ToUpper(ext[1:]))
    }

    dir, err := stagingDir()
    if err != nil {
        return "", nil, err
    }
    if err := os.MkdirAll(dir, 0o755); err != nil {
        return "", nil, err
    }
    out := filepath.Join(dir, filepath.Base(src)+".txt")
    if err := fn(src, out); err != nil {
        return "", nil, err
    }
    cleanup = func() { _ = os.Remove(out) }
    return out, cleanup, nil
}

func writePreparedText(outPath, text string) error {
    text = PrepareText(text)
    if text == "" {
        return fmt.Errorf("пустой результат извлечения текста")
    }
    return os.WriteFile(outPath, []byte(text), 0o644)
}

func writePDFPreparedText(outPath, text string) error {
    return writePreparedText(outPath, chunking.MarkPDFPages(text))
}

// extractors — встроенные конвертеры без внешних утилит (кроме опционального pdftotext для PDF).
var extractors = map[string]extractWriter{
    ".pdf":  extractPDFToFile,
    ".docx": extractDOCXToFile,
    ".html": extractHTMLToFile,
    ".htm":  extractHTMLToFile,
    ".xhtml": extractHTMLToFile,
    ".doc": nil,
}
