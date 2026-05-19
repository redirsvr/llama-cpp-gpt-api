package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files" // Swagger UI (HTML/JS/CSS) вшиты в модуль через go:embed
	ginSwagger "github.com/swaggo/gin-swagger"

	_ "llama-cpp-gpt-api/docs" // OpenAPI spec в бинарнике (make swagger)
)

// registerDocs подключает Swagger UI на /docs/index.html (без файлов на диске).
func registerDocs(r *gin.Engine) {
	r.GET("/docs", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/docs/index.html")
	})
	r.GET("/docs/*any", ginSwagger.WrapHandler(swaggerFiles.Handler, ginSwagger.URL("/docs/doc.json")))
}
