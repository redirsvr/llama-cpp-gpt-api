package rag

import (
    "fmt"
    "log"
    "os"
    "strings"

    "golang.org/x/net/html"
)

// extractHTMLToFile извлекает видимый текст из HTML.
func extractHTMLToFile(srcPath, outPath string) error {
    raw, err := os.ReadFile(srcPath)
    if err != nil {
        return fmt.Errorf("HTML: %w", err)
    }
    text, err := htmlToText(string(raw))
    if err != nil {
        return fmt.Errorf("HTML: %w", err)
    }
    if err := writePreparedText(outPath, text); err != nil {
        return fmt.Errorf("HTML: %w", err)
    }
    log.Printf("RAG: HTML → текст %q", srcPath)
    return nil
}

func htmlToText(src string) (string, error) {
    doc, err := html.Parse(strings.NewReader(src))
    if err != nil {
        return "", err
    }
    var b strings.Builder
    var walk func(*html.Node)
    walk = func(n *html.Node) {
        if n.Type == html.TextNode {
            s := strings.TrimSpace(n.Data)
            if s != "" {
                if b.Len() > 0 {
                    b.WriteByte(' ')
                }
                b.WriteString(s)
            }
        }
        if n.Type == html.ElementNode {
            switch n.Data {
            case "br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote":
                if b.Len() > 0 {
                    b.WriteByte('\n')
                }
            case "script", "style", "noscript":
                return
            }
        }
        for c := n.FirstChild; c != nil; c = c.NextSibling {
            walk(c)
        }
    }
    walk(doc)
    return strings.TrimSpace(b.String()), nil
}
