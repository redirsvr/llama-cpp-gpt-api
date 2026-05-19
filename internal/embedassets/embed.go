// Package embedassets — HTML, CSS и JS, вшитые в исполняемый файл (go:embed).
package embedassets

import (
	"embed"
	"io/fs"
)

// WebUI — файлы веб-интерфейса RAG (каталог ui/: index.html, app.js, app.css).
//
//go:embed all:ui
var WebUI embed.FS

// UISub — корень FS для раздачи по HTTP (/ui/ → index.html в корне).
func UISub() (fs.FS, error) {
	return fs.Sub(WebUI, "ui")
}
