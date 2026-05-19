package chunking

import (
	"fmt"
	"strings"
)

// Chunk — текст чанка и метаданные для цитирования и отладки.
type Chunk struct {
	Content string
	Meta    map[string]any
}

const pageMarkerPrefix = "<<<PAGE:"
const pageMarkerSuffix = ">>>"

// MarkPDFPages вставляет маркеры страниц по разделителю form feed (\f) из pdftotext и многих парсеров.
func MarkPDFPages(text string) string {
	parts := strings.Split(text, "\f")
	if len(parts) <= 1 {
		return text
	}
	var b strings.Builder
	page := 0
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		page++
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(fmt.Sprintf("%s%d%s\n", pageMarkerPrefix, page, pageMarkerSuffix))
		b.WriteString(p)
	}
	if b.Len() == 0 {
		return text
	}
	return b.String()
}

// ParsePageMarker возвращает номер страницы и true, если строка — маркер <<<PAGE:N>>>.
func ParsePageMarker(line string) (int, bool) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, pageMarkerPrefix) || !strings.HasSuffix(line, pageMarkerSuffix) {
		return 0, false
	}
	inner := strings.TrimSuffix(strings.TrimPrefix(line, pageMarkerPrefix), pageMarkerSuffix)
	var page int
	if _, err := fmt.Sscanf(inner, "%d", &page); err != nil || page < 1 {
		return 0, false
	}
	return page, true
}
