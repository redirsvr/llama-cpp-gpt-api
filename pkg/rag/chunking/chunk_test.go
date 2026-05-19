package chunking

import (
	"strings"
	"testing"
)

func TestMarkPDFPages(t *testing.T) {
	in := "page one\fpage two\f"
	out := MarkPDFPages(in)
	if !strings.Contains(out, "<<<PAGE:1>>>") || !strings.Contains(out, "<<<PAGE:2>>>") {
		t.Fatalf("ожидались маркеры страниц, получено: %q", out)
	}
	if p, ok := ParsePageMarker("<<<PAGE:2>>>"); !ok || p != 2 {
		t.Fatalf("ParsePageMarker: %d %v", p, ok)
	}
}
