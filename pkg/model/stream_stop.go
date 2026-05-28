package model

import "strings"

// StreamStopFilter отфильтровывает stop-последовательности из потока токенов
// (как llama-server: полные и частичные совпадения antiprompt).
type StreamStopFilter struct {
	stops   []string
	buf     string
	sentLen int
	done    bool
}

func NewStreamStopFilter(stops []string) *StreamStopFilter {
	return &StreamStopFilter{stops: stops}
}

func (f *StreamStopFilter) Push(token string) (toSend string, stopped bool) {
	if f.done {
		return "", true
	}
	if token == "" {
		return "", false
	}
	f.buf += token

	for _, stop := range f.stops {
		if stop == "" {
			continue
		}
		if idx := strings.LastIndex(f.buf, stop); idx >= 0 {
			f.buf = f.buf[:idx]
			f.done = true
			break
		}
	}

	hold := 0
	if !f.done {
		hold = partialStopHold(f.buf, f.stops)
	}

	safe := len(f.buf) - hold
	if safe > f.sentLen {
		toSend = f.buf[f.sentLen:safe]
		f.sentLen = safe
	}
	if f.done {
		stopped = true
	}
	return toSend, stopped
}

func partialStopHold(text string, stops []string) int {
	maxHold := 0
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		maxLen := len(text)
		if len(stop) < maxLen {
			maxLen = len(stop)
		}
		for l := maxLen; l > 0; l-- {
			if strings.HasSuffix(text, stop[:l]) {
				if l > maxHold {
					maxHold = l
				}
				break
			}
		}
	}
	return maxHold
}
