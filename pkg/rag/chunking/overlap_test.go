package chunking

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestApplyOverlap(t *testing.T) {
	chunks := []string{
		strings.Repeat("A", 100) + " END1",
		"START2 " + strings.Repeat("B", 100),
	}
	out := ApplyOverlap(chunks, 500, 10)
	if out[0] != chunks[0] {
		t.Fatal("first chunk must be unchanged")
	}
	prefix := tailRunes(chunks[0], 10)
	if !strings.HasPrefix(out[1], prefix) {
		t.Fatalf("chunk 2 must start with overlap %q, got %q", prefix, out[1][:min(40, len(out[1]))])
	}
}

func TestMergeByCharLimitOverlap(t *testing.T) {
	sents := make([]string, 30)
	for i := range sents {
		sents[i] = strings.Repeat("word ", 20)
	}
	chunks := MergeByCharLimit(sents, 200, 10, 30)
	if len(chunks) < 2 {
		t.Fatal("expected multiple chunks")
	}
	prefix := tailRunes(chunks[0], 30)
	if !strings.HasPrefix(chunks[1], prefix) {
		t.Fatalf("expected overlap prefix in chunk 2")
	}
}

func TestApplyOverlapToFileChunks(t *testing.T) {
	chunks := []Chunk{
		{Content: "first chunk text here.", Meta: map[string]any{"char_start": 0, "char_end": 22}},
		{Content: "second chunk continues.", Meta: map[string]any{"char_start": 22, "char_end": 46}},
	}
	applyOverlapToFileChunks(chunks, 500, 12)
	if chunks[0].Content != "first chunk text here." {
		t.Fatal("first chunk unchanged")
	}
	if !strings.Contains(chunks[1].Content, "here.") {
		t.Fatalf("second chunk should include tail of first: %q", chunks[1].Content)
	}
	cs, _ := chunks[1].Meta["char_start"].(int)
	if cs >= 22 {
		t.Fatalf("char_start should move back with overlap, got %d", cs)
	}
	if chunks[1].Meta["overlap_runes"] == nil {
		t.Fatal("expected overlap_runes metadata")
	}
	_ = utf8.RuneCountInString(chunks[1].Content)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
