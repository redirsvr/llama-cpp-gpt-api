package gpt

import (
	"strings"
	"time"
)

// streamChunkBuffer копит токены и отдаёт их пачками, чтобы не делать JSON+flush на каждый токен.
type streamChunkBuffer struct {
	buf         strings.Builder
	lastFlush   time.Time
	minInterval time.Duration
	minChars    int
}

func newStreamChunkBuffer() *streamChunkBuffer {
	return &streamChunkBuffer{
		minInterval: 40 * time.Millisecond,
		minChars:    12,
		lastFlush:   time.Now(),
	}
}

func (b *streamChunkBuffer) Push(token string) (chunk string, ready bool) {
	if token == "" {
		return "", false
	}
	b.buf.WriteString(token)
	if b.buf.Len() >= b.minChars || time.Since(b.lastFlush) >= b.minInterval {
		return b.flush()
	}
	return "", false
}

func (b *streamChunkBuffer) FlushFinal() (chunk string, ready bool) {
	if b.buf.Len() == 0 {
		return "", false
	}
	return b.flush()
}

func (b *streamChunkBuffer) flush() (string, bool) {
	out := b.buf.String()
	b.buf.Reset()
	b.lastFlush = time.Now()
	if out == "" {
		return "", false
	}
	return out, true
}
