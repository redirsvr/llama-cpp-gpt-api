package metrics

import (
    "time"

    "github.com/gin-gonic/gin"
)

// GinMiddleware записывает метрики HTTP для маршрута Gin.
func GinMiddleware(routePath string) gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        c.Next()
        status := c.Writer.Status()
        if status == 0 {
            status = 200
        }
        ObserveHTTP(c.Request.Method, routePath, status, time.Since(start))
    }
}
