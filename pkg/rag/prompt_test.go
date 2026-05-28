package rag

import (
	"testing"

	"llama-cpp-gpt-api/internal/types"
)

func TestStripRAGFromMessages(t *testing.T) {
	in := []types.Message{
		{Role: "system", Content: ragInstruction + "\n\n" + ragMarker},
		{Role: "user", Content: "Как дела?"},
	}
	out := StripRAGFromMessages(in)
	if len(out) != 1 || out[0].Role != "user" {
		t.Fatalf("expected only user message: %+v", out)
	}
}

func TestBuildContext_EmptySearch(t *testing.T) {
	if got := BuildContext(nil); got != "" {
		t.Fatalf("empty hits must not inject RAG: %q", got)
	}
}
