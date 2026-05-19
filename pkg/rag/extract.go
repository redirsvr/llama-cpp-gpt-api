package rag

import (
    "fmt"
    "os"
    "os/exec"
    "strings"

    "llama-cpp-gpt-api/internal/config"
)

// pdfExtractMode: builtin (по умолчанию) | auto | pdftotext.
func pdfExtractMode() string {
    m := strings.ToLower(strings.TrimSpace(config.C.RAG.PDFExtract))
    switch m {
    case "pdftotext", "auto":
        return m
    default:
        return "builtin"
    }
}

func extractPDFPdftotext(pdfPath, outPath string) error {
    if _, err := exec.LookPath("pdftotext"); err != nil {
        return fmt.Errorf("pdftotext не найден в PATH")
    }
    cmd := exec.Command("pdftotext", "-q", "-enc", "UTF-8", pdfPath, outPath)
    out, err := cmd.CombinedOutput()
    if err != nil {
        return fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out)))
    }
    raw, err := os.ReadFile(outPath)
    if err != nil {
        return err
    }
    return writePDFPreparedText(outPath, string(raw))
}
