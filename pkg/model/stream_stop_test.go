package model

import "testing"

func TestStreamStopFilter_FullStop(t *testing.T) {
	f := NewStreamStopFilter(QwenStopWords())
	for _, tok := range []string{"Ответ", ".", tokenIMEnd} {
		chunk, stopped := f.Push(tok)
		if chunk != "" && stopped {
			t.Fatalf("unexpected chunk before stop: %q", chunk)
		}
		if stopped {
			break
		}
	}
	if f.buf != "Ответ." {
		t.Fatalf("buf = %q, want %q", f.buf, "Ответ.")
	}
}

func TestStreamStopFilter_PartialStop(t *testing.T) {
	f := NewStreamStopFilter(QwenStopWords())
	chunk, stopped := f.Push("hi<|")
	if stopped || chunk != "" {
		t.Fatalf("partial stop must hold tail: chunk=%q stopped=%v", chunk, stopped)
	}
	chunk, stopped = f.Push("im_end|>")
	if chunk != "hi" {
		t.Fatalf("chunk = %q, want hi", chunk)
	}
	if !stopped {
		t.Fatal("expected stop")
	}
}
