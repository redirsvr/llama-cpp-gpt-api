package api

import (
	"net/http"

	"llama-cpp-gpt-api/internal/config"
	"llama-cpp-gpt-api/internal/embedassets"

	"github.com/gin-gonic/gin"
)

// registerUI раздаёт веб-интерфейс из go:embed (без каталога static на диске).
func registerUI(r *gin.Engine) {
	if !config.C.RAG.Enabled || config.C.RAG.UIDisabled {
		return
	}

	uiFS, err := embedassets.UISub()
	if err != nil {
		return
	}

	r.MaxMultipartMemory = 64 << 20

	r.GET("/ui", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/ui/index.html")
	})
	r.StaticFS("/ui", http.FS(uiFS))
}
