package embedassets

import (
	"io/fs"
	"testing"
)

func TestWebUIEmbedded(t *testing.T) {
	for _, name := range []string{"ui/index.html", "ui/app.js", "ui/app.css"} {
		if _, err := WebUI.ReadFile(name); err != nil {
			t.Fatalf("нет в embed %s: %v", name, err)
		}
	}
	sub, err := UISub()
	if err != nil {
		t.Fatal(err)
	}
	entries, err := fs.ReadDir(sub, ".")
	if err != nil || len(entries) < 3 {
		t.Fatalf("ui/: %d файлов, err=%v", len(entries), err)
	}
}
