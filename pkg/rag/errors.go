package rag

import (
    "errors"
    "strings"
)

// ErrPDFExtract — не удалось извлечь/конвертировать документ (PDF, DOCX, HTML).
var ErrPDFExtract = errors.New("pdf_extract_failed")

// ErrEmptyDocument — нет текста для индексации.
var ErrEmptyDocument = errors.New("empty_document")

// WrapIngestError помечает ошибки ingest для HTTP-маппинга.
func WrapIngestError(err error) error {
    if err == nil {
        return nil
    }
    msg := err.Error()
    if strings.Contains(msg, "PDF:") || strings.Contains(msg, "DOCX:") || strings.Contains(msg, "HTML:") ||
        strings.Contains(msg, "DOC:") || strings.Contains(msg, "pdftotext") || strings.Contains(msg, "извлечь текст") {
        return errors.Join(ErrPDFExtract, err)
    }
    if strings.Contains(msg, "пуст") ||
        strings.Contains(msg, "не содержит текста") ||
        strings.Contains(msg, "нет чанков с эмбеддингами") {
        return errors.Join(ErrEmptyDocument, err)
    }
    return err
}
