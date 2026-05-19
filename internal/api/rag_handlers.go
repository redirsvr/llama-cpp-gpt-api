package api

import (
    "net/http"
    "strings"

    "llama-cpp-gpt-api/internal/types"
    "llama-cpp-gpt-api/pkg/rag"
    "llama-cpp-gpt-api/pkg/rag/store"

    "github.com/gin-gonic/gin"
)

type ragDisabledError struct{}

func (e *ragDisabledError) Error() string {
    return "RAG отключён в конфиге (RAG.Enabled: false)"
}

type missingContentError struct{}

func (e *missingContentError) Error() string {
    return "укажите content или path"
}

func ragDisabled(c *gin.Context) bool {
    if !rag.Enabled() {
        c.JSON(http.StatusServiceUnavailable, OpenAIError(&ragDisabledError{}, "rag_disabled"))
        return true
    }
    return false
}

// RAGIngestDocument godoc
// @Summary      Индексация документа (JSON)
// @Description  Индексирует content или файл по path на сервере.
// @Tags         rag
// @Accept       json
// @Produce      json
// @Param        body  body      types.ReqRAGIngest  true  "Документ"
// @Success      200   {object}  types.ResRAGIngest
// @Failure      400   {object}  types.OpenAIErrorBody
// @Failure      503   {object}  types.OpenAIErrorBody
// @Failure      504   {object}  types.OpenAIErrorBody
// @Failure      500   {object}  types.OpenAIErrorBody
// @Router       /v1/rag/documents [post]
func RAGIngestDocument(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    var req types.ReqRAGIngest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
        return
    }

    ingestCtx, cancel := rag.IngestContext()
    defer cancel()
    var (
        doc *store.Document
        err error
    )

    switch {
    case strings.TrimSpace(req.Content) != "":
        doc, err = rag.IngestDocument(ingestCtx, rag.IngestInput{
            Title:      req.Title,
            Content:    req.Content,
            SourcePath: req.Path,
            Metadata:   req.Metadata,
        })
    case strings.TrimSpace(req.Path) != "":
        doc, err = rag.IngestFile(ingestCtx, req.Path)
    default:
        c.JSON(http.StatusBadRequest, OpenAIError(&missingContentError{}, "invalid_request"))
        return
    }
    if err != nil {
        respondRAGIngestError(c, err)
        return
    }
    c.JSON(http.StatusOK, types.ResRAGIngest{Document: *doc})
}

// RAGIngestUpload godoc
// @Summary      Загрузка файла
// @Description  Multipart upload: PDF, DOCX, TXT и др. — конвертация и индексация.
// @Tags         rag
// @Accept       multipart/form-data
// @Produce      json
// @Param        file   formData  file    true   "Файл"
// @Param        title  formData  string  false  "Название документа"
// @Success      200    {object}  types.ResRAGIngest
// @Failure      400    {object}  types.OpenAIErrorBody
// @Failure      503    {object}  types.OpenAIErrorBody
// @Failure      500    {object}  types.OpenAIErrorBody
// @Router       /v1/rag/documents/upload [post]
func RAGIngestUpload(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    file, err := c.FormFile("file")
    if err != nil {
        c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
        return
    }
    f, err := file.Open()
    if err != nil {
        c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
        return
    }
    defer f.Close()

    title := c.PostForm("title")
    if title == "" {
        title = file.Filename
    }
    meta := map[string]any{
        "upload":   true,
        "filename": file.Filename,
    }
    ingestCtx, cancel := rag.IngestContext()
    defer cancel()
    doc, err := rag.IngestReader(ingestCtx, title, f, meta)
    if err != nil {
        respondRAGIngestError(c, err)
        return
    }
    c.JSON(http.StatusOK, types.ResRAGIngest{Document: *doc})
}

// RAGIngestScan godoc
// @Summary      Сканирование каталогов
// @Description  Индексирует файлы из каталогов RAG.AutoIngest в конфиге.
// @Tags         rag
// @Produce      json
// @Success      200  {object}  types.ResRAGScan
// @Failure      503  {object}  types.OpenAIErrorBody
// @Failure      500  {object}  types.OpenAIErrorBody
// @Router       /v1/rag/documents/scan [post]
func RAGIngestScan(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    n, err := rag.ScanDirectories(c.Request.Context())
    if err != nil {
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "scan_failed"))
        return
    }
    c.JSON(http.StatusOK, types.ResRAGScan{Ingested: n})
}

// RAGReindexEmbeddings godoc
// @Summary      Пересчёт эмбеддингов
// @Description  Пересчитывает embedding для всех чанков (после смены префиксов search_document/search_query).
// @Tags         rag
// @Produce      json
// @Success      200  {object}  types.ResRAGReindex
// @Failure      503  {object}  types.OpenAIErrorBody
// @Failure      500  {object}  types.OpenAIErrorBody
// @Router       /v1/rag/reindex-embeddings [post]
func RAGReindexEmbeddings(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    n, err := rag.ReindexAllEmbeddings(c.Request.Context())
    if err != nil {
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "reindex_failed"))
        return
    }
    c.JSON(http.StatusOK, types.ResRAGReindex{ReindexedChunks: n})
}

// RAGQuery godoc
// @Summary      Поиск по RAG
// @Description  Гибридный поиск (вектор + полнотекстовый) по проиндексированным документам.
// @Tags         rag
// @Accept       json
// @Produce      json
// @Param        body  body      types.ReqRAGQuery  true  "Запрос"
// @Success      200   {object}  types.ResRAGQuery
// @Failure      400   {object}  types.OpenAIErrorBody
// @Failure      503   {object}  types.OpenAIErrorBody
// @Failure      500   {object}  types.OpenAIErrorBody
// @Router       /v1/rag/query [post]
func RAGQuery(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    var req types.ReqRAGQuery
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, OpenAIError(err, "invalid_request"))
        return
    }
    results, err := rag.Query(c.Request.Context(), req.Query, req.TopK)
    if err != nil {
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "query_failed"))
        return
    }
    c.JSON(http.StatusOK, types.ResRAGQuery{Query: req.Query, Results: results})
}

// RAGListDocuments godoc
// @Summary      Список документов
// @Tags         rag
// @Produce      json
// @Success      200  {object}  types.ResRAGDocuments
// @Failure      503  {object}  types.OpenAIErrorBody
// @Failure      500  {object}  types.OpenAIErrorBody
// @Router       /v1/rag/documents [get]
func RAGListDocuments(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    docs, err := rag.ListDocuments(c.Request.Context(), 200)
    if err != nil {
        c.JSON(http.StatusInternalServerError, OpenAIError(err, "list_failed"))
        return
    }
    if docs == nil {
        docs = []store.Document{}
    }
    c.JSON(http.StatusOK, types.ResRAGDocuments{Data: docs})
}

// RAGGetDocument godoc
// @Summary      Документ по ID
// @Tags         rag
// @Produce      json
// @Param        id   path      string  true  "UUID документа"
// @Success      200  {object}  store.Document
// @Failure      404  {object}  types.OpenAIErrorBody
// @Failure      503  {object}  types.OpenAIErrorBody
// @Router       /v1/rag/documents/{id} [get]
func RAGGetDocument(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    doc, err := rag.GetDocument(c.Request.Context(), c.Param("id"))
    if err != nil {
        c.JSON(http.StatusNotFound, OpenAIError(err, "not_found"))
        return
    }
    c.JSON(http.StatusOK, doc)
}

// RAGDeleteDocument godoc
// @Summary      Удалить документ
// @Tags         rag
// @Param        id  path  string  true  "UUID документа"
// @Success      204
// @Failure      404  {object}  types.OpenAIErrorBody
// @Failure      503  {object}  types.OpenAIErrorBody
// @Router       /v1/rag/documents/{id} [delete]
func RAGDeleteDocument(c *gin.Context) {
    if ragDisabled(c) {
        return
    }
    if err := rag.DeleteDocument(c.Request.Context(), c.Param("id")); err != nil {
        c.JSON(http.StatusNotFound, OpenAIError(err, "not_found"))
        return
    }
    c.Status(http.StatusNoContent)
}
