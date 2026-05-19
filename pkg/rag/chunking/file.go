package chunking

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"unicode/utf8"
)

// ChunksFromFile читает файл потоково, считает SHA-256 и режет на чанки с метаданными (индекс, смещения, страница PDF).
func ChunksFromFile(path string, maxChars, minChars int) (contentHash string, chunks []Chunk, err error) {
	if maxChars <= 0 {
		maxChars = 1500
	}
	if minChars <= 0 {
		minChars = 80
	}

	f, err := os.Open(path)
	if err != nil {
		return "", nil, err
	}
	defer f.Close()

	hasher := sha256.New()
	tr := io.TeeReader(f, hasher)

	var (
		cur       []rune
		line      []rune
		page      = 1
		chunkIdx  int
		charStart int
		charPos   int
	)

	flush := func() {
		if len(cur) == 0 {
			return
		}
		s := string(cur)
		charEnd := charPos
		meta := map[string]any{
			"chunk_index": chunkIdx,
			"char_start":  charStart,
			"char_end":    charEnd,
		}
		if page > 0 {
			meta["page"] = page
		}
		cur = cur[:0]
		if utf8.RuneCountInString(s) < minChars && len(chunks) > 0 {
			last := &chunks[len(chunks)-1]
			last.Content += " " + s
			last.Meta["char_end"] = charEnd
			return
		}
		chunks = append(chunks, Chunk{Content: s, Meta: meta})
		chunkIdx++
		charStart = charPos
	}

	appendLineToCur := func() {
		if len(line) == 0 {
			return
		}
		cur = append(cur, line...)
		cur = append(cur, '\n')
		charPos += len(line) + 1
		line = line[:0]
	}

	buf := make([]byte, 256*1024)
	var carry []byte
	for {
		n, readErr := tr.Read(buf)
		if n > 0 {
			carry = append(carry, buf[:n]...)
			for len(carry) > 0 {
				r, size := utf8.DecodeRune(carry)
				if r == utf8.RuneError && size == 1 {
					if len(carry) >= utf8.UTFMax {
						carry = carry[1:]
						continue
					}
					break
				}
				carry = carry[size:]
				if r < 32 && r != '\n' && r != '\r' && r != '\t' {
					continue
				}
				if r == '\r' {
					continue
				}
				if r == '\n' {
					if p, ok := ParsePageMarker(string(line)); ok {
						page = p
						line = line[:0]
						continue
					}
					appendLineToCur()
					if len(cur) >= maxChars {
						flush()
					}
					continue
				}
				line = append(line, r)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", nil, readErr
		}
	}
	if len(line) > 0 {
		if p, ok := ParsePageMarker(string(line)); ok {
			page = p
		} else {
			appendLineToCur()
		}
	}
	flush()

	if len(chunks) == 0 {
		return "", nil, fmt.Errorf("файл пуст или не содержит текста")
	}
	return hex.EncodeToString(hasher.Sum(nil)), chunks, nil
}
