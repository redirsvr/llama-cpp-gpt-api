package api

import (
    "errors"
    "net/http"

    "llama-cpp-gpt-api/internal/logic/gpt"
    "llama-cpp-gpt-api/internal/types"
    "llama-cpp-gpt-api/pkg/model"

    "github.com/gin-gonic/gin"
)

// ChatCompletions godoc
// @Summary      Chat completions
// @Description  OpenAI-совместимая генерация ответа. При включённом RAG в system подмешиваются фрагменты из базы.
// @Tags         chat
// @Accept       json
// @Produce      json
// @Param        body  body      types.ReqChatCompletion  true  "Запрос"
// @Success      200   {object}  types.ResChatCompletion
// @Failure      400   {object}  types.OpenAIErrorBody
// @Failure      500   {object}  types.OpenAIErrorBody
// @Router       /v1/chat/completions [post]
func ChatCompletions(c *gin.Context) {
    var req types.ReqChatCompletion
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
        return
    }

    w := c.Writer
    r := c.Request
    l := gpt.NewChatCompletionsLogic(c.Request.Context(), w, r)
    resp, err := l.ChatCompletions(&req)
    if err != nil {
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "server_error"))
        return
    }
    if resp != nil {
        c.JSON(http.StatusOK, resp)
    }
}

// Embeddings godoc
// @Summary      Embeddings
// @Description  Векторное представление текста (POST JSON или GET query).
// @Tags         embeddings
// @Accept       json
// @Produce      json
// @Param        body   body      types.ReqEmbeddings  false  "Тело (POST)"
// @Param        model  query     string               false  "Модель"
// @Param        input  query     string               true   "Текст"
// @Success      200    {object}  types.ResEmbeddings
// @Failure      400    {object}  types.OpenAIErrorBody
// @Failure      500    {object}  types.OpenAIErrorBody
// @Router       /v1/embeddings [post]
// @Router       /v1/embeddings [get]
func Embeddings(c *gin.Context) {
    var req types.ReqEmbeddings
    switch c.Request.Method {
    case http.MethodPost:
        if err := c.ShouldBindJSON(&req); err != nil {
            c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
            return
        }
    default:
        if err := c.ShouldBindQuery(&req); err != nil {
            c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
            return
        }
    }

    w := c.Writer
    r := c.Request
    l := gpt.NewEmbeddingsLogic(c.Request.Context(), w, r)
    resp, err := l.Embeddings(&req)
    if err != nil {
        if errors.Is(err, model.ErrEmbeddingNotSupported) {
            c.JSON(http.StatusBadRequest, OpenAIError(err, "model_not_supported"))
            return
        }
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "server_error"))
        return
    }
    if resp != nil {
        c.JSON(http.StatusOK, resp)
    }
}

// ListModels godoc
// @Summary      Список моделей
// @Tags         models
// @Produce      json
// @Success      200  {object}  types.ResModelsList
// @Failure      500  {object}  types.OpenAIErrorBody
// @Router       /v1/models [get]
func ListModels(c *gin.Context) {
    l := gpt.NewModelsLogic(c.Request.Context())
    resp, err := l.ListModels()
    if err != nil {
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "server_error"))
        return
    }
    c.JSON(http.StatusOK, resp)
}

// RetrieveModel godoc
// @Summary      Модель по ID
// @Tags         models
// @Produce      json
// @Param        model  path      string  true  "ID модели"
// @Success      200    {object}  types.ModelObject
// @Failure      404    {object}  types.OpenAIErrorBody
// @Failure      500    {object}  types.OpenAIErrorBody
// @Router       /v1/models/{model} [get]
func RetrieveModel(c *gin.Context) {
    id := c.Param("model")
    l := gpt.NewModelsLogic(c.Request.Context())
    resp, err := l.RetrieveModel(id)
    if err != nil {
        if errors.Is(err, model.ErrModelNotFound) {
            c.JSON(http.StatusNotFound, OpenAIError(err, "model_not_found"))
            return
        }
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "server_error"))
        return
    }
    c.JSON(http.StatusOK, resp)
}
