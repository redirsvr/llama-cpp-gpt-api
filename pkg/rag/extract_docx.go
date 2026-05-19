package rag

import (
    "archive/zip"
    "encoding/xml"
    "fmt"
    "io"
    "log"
    "strings"
)

// extractDOCXToFile извлекает текст из Office Open XML (.docx).
func extractDOCXToFile(srcPath, outPath string) error {
    text, err := readDOCXText(srcPath)
    if err != nil {
        return fmt.Errorf("DOCX: %w", err)
    }
    if err := writePreparedText(outPath, text); err != nil {
        return fmt.Errorf("DOCX: %w", err)
    }
    log.Printf("RAG: DOCX → текст %q", srcPath)
    return nil
}

func readDOCXText(path string) (string, error) {
    zr, err := zip.OpenReader(path)
    if err != nil {
        return "", fmt.Errorf("открытие zip: %w", err)
    }
    defer zr.Close()

    var parts []string
    for _, f := range zr.File {
        if f.Name != "word/document.xml" &&
            f.Name != "word/header1.xml" &&
            f.Name != "word/footer1.xml" {
            continue
        }
        rc, err := f.Open()
        if err != nil {
            return "", err
        }
        s, err := parseWordXML(rc)
        rc.Close()
        if err != nil {
            return "", fmt.Errorf("%s: %w", f.Name, err)
        }
        if strings.TrimSpace(s) != "" {
            parts = append(parts, s)
        }
    }
    if len(parts) == 0 {
        return "", fmt.Errorf("в документе нет word/document.xml или текста")
    }
    return strings.Join(parts, "\n\n"), nil
}

func parseWordXML(r io.Reader) (string, error) {
    dec := xml.NewDecoder(r)
    var b strings.Builder
    inT := false
    for {
        tok, err := dec.Token()
        if err == io.EOF {
            break
        }
        if err != nil {
            return "", err
        }
        switch t := tok.(type) {
        case xml.StartElement:
            switch t.Name.Local {
            case "t":
                inT = true
            case "tab":
                b.WriteByte('\t')
            case "br", "cr":
                b.WriteByte('\n')
            case "p":
                if b.Len() > 0 {
                    b.WriteByte('\n')
                }
            }
        case xml.EndElement:
            if t.Name.Local == "t" {
                inT = false
            }
        case xml.CharData:
            if inT {
                b.Write(t)
            }
        }
    }
    return strings.TrimSpace(b.String()), nil
}
