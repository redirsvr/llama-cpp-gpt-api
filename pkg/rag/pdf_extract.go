package rag

import (
    "bytes"
    "errors"
    "fmt"
    "io"
    "log"
    "os"
    "os/exec"
    "strings"

    alepdf "github.com/Alechan/pdf"
    "github.com/ledongthuc/pdf"
    superpdf "github.com/superpowerdotcom/go-pdf-lib"
)

type pdfPlainReader interface {
    GetPlainText() (io.Reader, error)
}

// extractPDFToFile — несколько встроенных парсеров, затем pdftotext (если есть).
func extractPDFToFile(pdfPath, outPath string) error {
    if err := validatePDFFile(pdfPath); err != nil {
        return fmt.Errorf("PDF: %w", err)
    }

    mode := pdfExtractMode()
    if mode == "pdftotext" {
        if err := extractPDFPdftotext(pdfPath, outPath); err != nil {
            return fmt.Errorf("PDF: %w", err)
        }
        log.Printf("RAG: PDF → текст (pdftotext) %q", pdfPath)
        return nil
    }

    var errs []error
    for _, b := range []struct {
        name string
        fn   func(string) (string, error)
    }{
        {"go-pdf-lib", plainTextSuperpower},
        {"alechan", plainTextAlechan},
        {"ledongthuc", plainTextLedongthuc},
    } {
        text, err := b.fn(pdfPath)
        if err != nil {
            log.Printf("RAG: PDF парсер %s: %v", b.name, err)
            errs = append(errs, fmt.Errorf("%s: %w", b.name, err))
            continue
        }
        if err := writePDFPreparedText(outPath, text); err != nil {
            errs = append(errs, fmt.Errorf("%s: %w", b.name, err))
            continue
        }
        log.Printf("RAG: PDF → текст (%s) %q", b.name, pdfPath)
        return nil
    }

    if mode == "auto" || mode == "builtin" {
        if _, lookErr := exec.LookPath("pdftotext"); lookErr == nil {
            if err := extractPDFPdftotext(pdfPath, outPath); err == nil {
                log.Printf("RAG: PDF → текст (pdftotext) %q", pdfPath)
                return nil
            } else {
                errs = append(errs, fmt.Errorf("pdftotext: %w", err))
            }
        }
    }

    hint := "проверьте, что файл не повреждён и не зашифрован; для сканов нужен OCR"
    if _, lookErr := exec.LookPath("pdftotext"); lookErr != nil {
        hint += "; для сложных PDF установите poppler-utils (pdftotext)"
    }
    return fmt.Errorf("PDF: не удалось извлечь текст (%v). %s", errors.Join(errs...), hint)
}

func validatePDFFile(path string) error {
    info, err := os.Stat(path)
    if err != nil {
        return err
    }
    if info.Size() < 64 {
        return fmt.Errorf("файл слишком маленький (%d байт), возможно загрузка оборвалась", info.Size())
    }
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()
    hdr := make([]byte, 1024)
    n, _ := io.ReadFull(f, hdr)
    if n < 5 || !bytes.HasPrefix(hdr, []byte("%PDF")) {
        return fmt.Errorf("нет заголовка %%PDF — возможно, это не PDF или файл повреждён при загрузке")
    }
    tail := make([]byte, 65536)
    if info.Size() > int64(len(tail)) {
        if _, err := f.Seek(info.Size()-int64(len(tail)), io.SeekStart); err == nil {
            if m, _ := f.Read(tail); m > 0 {
                tail = tail[:m]
            }
        }
    }
    blob := string(hdr) + string(tail)
    if strings.Contains(blob, "/Encrypt") {
        return fmt.Errorf("PDF зашифрован — сохраните копию без пароля или загрузите .txt")
    }
    return nil
}

func readPlainText(r pdfPlainReader) (string, error) {
    reader, err := r.GetPlainText()
    if err != nil {
        return "", err
    }
    var buf bytes.Buffer
    if _, err := io.Copy(&buf, reader); err != nil {
        return "", err
    }
    return buf.String(), nil
}

func plainTextFromOpen(open func(string) (*os.File, pdfPlainReader, error), path string) (string, error) {
    f, r, err := open(path)
    if err != nil {
        return "", err
    }
    defer f.Close()
    return readPlainText(r)
}

func plainTextLedongthuc(path string) (string, error) {
    return plainTextFromOpen(func(p string) (*os.File, pdfPlainReader, error) {
        return pdf.Open(p)
    }, path)
}

func plainTextAlechan(path string) (string, error) {
    return plainTextFromOpen(func(p string) (*os.File, pdfPlainReader, error) {
        return alepdf.Open(p)
    }, path)
}

func plainTextSuperpower(path string) (string, error) {
    return plainTextFromOpen(func(p string) (*os.File, pdfPlainReader, error) {
        return superpdf.Open(p)
    }, path)
}
