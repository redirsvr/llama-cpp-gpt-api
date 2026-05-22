package model

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"llama-cpp-gpt-api/internal/config"
	"llama-cpp-gpt-api/pkg/metrics"

	llama "github.com/redirsvr/go-llama-new.cpp"
)

var modelExtensions = map[string]struct{}{
	".gguf": {},
	".bin":  {},
}

type loadMode int

const (
	loadModeChat loadMode = iota
	loadModeEmbeddings
)

type Registry struct {
	mu                    sync.Mutex
	dir                   string
	aliases               map[string]string
	embeddingAliases      map[string]struct{}
	loaded                *llama.LLama
	loadedAlias           string
	loadedMode            loadMode
	defaultAlias          string
	defaultEmbeddingAlias string
}

var registry *Registry

var (
	ErrModelNotFound         = errors.New("model not found")
	ErrEmbeddingNotSupported = errors.New("model does not support embeddings")
)

// Init сканирует каталог моделей и при необходимости предзагружает DefaultModel.
func Init() error {
	dir, singleFile, err := resolveModelsLocation()
	if err != nil {
		return err
	}

	registry = &Registry{
		aliases:               make(map[string]string),
		embeddingAliases:      buildEmbeddingAliasSet(),
		defaultAlias:          NormalizeAlias(config.C.DefaultModel),
		defaultEmbeddingAlias: NormalizeAlias(config.C.DefaultEmbeddingModel),
	}

	if singleFile != "" {
		alias := NormalizeAlias(filepath.Base(singleFile))
		registry.aliases[alias] = singleFile
		if registry.defaultAlias == "" {
			registry.defaultAlias = alias
		}
		log.Printf("models: один файл %q → алиас %q", singleFile, alias)
	} else {
		registry.dir = dir
		if err := registry.scan(); err != nil {
			return err
		}
		log.Printf("models: каталог %q, алиасы: %v", dir, registry.ListAliases())
	}

	if err := ValidateModelOptionGPU(); err != nil {
		return err
	}
	if err := registry.validateEmbeddingModels(); err != nil {
		return err
	}
	if len(registry.embeddingAliases) > 0 {
		log.Printf("models: embeddings: %v", registry.ListEmbeddingAliases())
	}

	if registry.defaultAlias != "" {
		if _, ok := registry.aliases[registry.defaultAlias]; !ok {
			return fmt.Errorf("модель по умолчанию %q не найдена", registry.defaultAlias)
		}
		if config.C.PreloadDefaultModel {
			log.Printf("models: предзагрузка %q", registry.defaultAlias)
			registry.mu.Lock()
			_, _, err := registry.getLocked(registry.defaultAlias, loadModeChat)
			registry.mu.Unlock()
			if err != nil {
				return err
			}
		}
		log.Printf("models: предзагрузка отключена, %q загрузится по первому запросу", registry.defaultAlias)
	}

	return nil
}

func buildEmbeddingAliasSet() map[string]struct{} {
	set := make(map[string]struct{}, len(config.C.EmbeddingModels))
	for _, name := range config.C.EmbeddingModels {
		key := NormalizeAlias(name)
		if key == "" {
			continue
		}
		set[key] = struct{}{}
	}
	return set
}

func (r *Registry) validateEmbeddingModels() error {
	for alias := range r.embeddingAliases {
		if _, ok := r.aliases[alias]; !ok {
			return fmt.Errorf("EmbeddingModels: модель %q не найдена в каталоге", alias)
		}
	}
	if r.defaultEmbeddingAlias != "" {
		if _, ok := r.embeddingAliases[r.defaultEmbeddingAlias]; !ok {
			return fmt.Errorf("DefaultEmbeddingModel %q отсутствует в EmbeddingModels", r.defaultEmbeddingAlias)
		}
	}
	return nil
}

func (r *Registry) supportsEmbeddings(alias string) bool {
	_, ok := r.embeddingAliases[NormalizeAlias(alias)]
	return ok
}

func (r *Registry) ListEmbeddingAliases() []string {
	out := make([]string, 0, len(r.embeddingAliases))
	for a := range r.embeddingAliases {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

func resolveModelsLocation() (dir string, singleFile string, err error) {
	path := config.C.ModelsDir
	if path == "" {
		path = config.C.ModelPath
	}
	if path == "" {
		return "", "", fmt.Errorf("не задан ModelsDir (или ModelPath) в конфигурации")
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", "", fmt.Errorf("models path: %w", err)
	}
	if !info.IsDir() {
		abs, err := filepath.Abs(path)
		if err != nil {
			return "", "", err
		}
		return "", abs, nil
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", "", err
	}
	return abs, "", nil
}

func (r *Registry) scan() error {
	entries, err := os.ReadDir(r.dir)
	if err != nil {
		return fmt.Errorf("чтение каталога моделей: %w", err)
	}

	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if _, ok := modelExtensions[ext]; !ok {
			continue
		}
		alias := strings.TrimSuffix(e.Name(), ext)
		full := filepath.Join(r.dir, e.Name())
		if prev, dup := r.aliases[alias]; dup {
			return fmt.Errorf("дублирующийся алиас %q: %s и %s", alias, prev, full)
		}
		r.aliases[alias] = full
	}

	if len(r.aliases) == 0 {
		return fmt.Errorf("в каталоге %q нет файлов моделей (.gguf, .bin)", r.dir)
	}
	return nil
}

// NormalizeAlias возвращает алиас модели: имя файла без .gguf/.bin.
func NormalizeAlias(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}
	base := filepath.Base(name)
	ext := strings.ToLower(filepath.Ext(base))
	if _, ok := modelExtensions[ext]; ok {
		return strings.TrimSuffix(base, ext)
	}
	return base
}

func (r *Registry) ListAliases() []string {
	out := make([]string, 0, len(r.aliases))
	for a := range r.aliases {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

// UseChat выполняет fn с эксклюзивным доступом к модели (один инференс за раз).
func UseChat(alias string, fn func(*llama.LLama, string) error) error {
	if registry == nil {
		return fmt.Errorf("реестр моделей не инициализирован")
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	ll, key, err := registry.getLocked(alias, loadModeChat)
	if err != nil {
		return err
	}
	return fn(ll, key)
}

// UseEmbeddings — как UseChat, но для embedding-модели.
func UseEmbeddings(alias string, fn func(*llama.LLama, string) error) error {
	if registry == nil {
		return fmt.Errorf("реестр моделей не инициализирован")
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	ll, key, err := registry.getLocked(alias, loadModeEmbeddings)
	if err != nil {
		return err
	}
	return fn(ll, key)
}

func (r *Registry) getLocked(alias string, mode loadMode) (*llama.LLama, string, error) {
	key, path, err := r.resolve(alias, mode)
	if err != nil {
		return nil, "", err
	}

	if r.loaded != nil && r.loadedAlias == key && r.loadedMode == mode {
		return r.loaded, key, nil
	}

	if r.loaded != nil {
		metrics.SetModelLoaded(r.loadedAlias, loadModeLabel(r.loadedMode), false)
		r.loaded.Free()
		r.loaded = nil
		r.loadedAlias = ""
	}

	embeddings := mode == loadModeEmbeddings
	modeLabel := loadModeLabel(mode)
	log.Printf("models: загрузка %q из %s (embeddings=%v)", key, path, embeddings)
	loadStart := time.Now()
	l, err := openLlama(path, embeddings)
	loadDur := time.Since(loadStart)
	if err != nil {
		metrics.ObserveModelLoad(key, modeLabel, "error", loadDur)
		return nil, "", err
	}
	metrics.ObserveModelLoad(key, modeLabel, "success", loadDur)
	metrics.SetModelLoaded(key, modeLabel, true)
	r.loaded = l
	r.loadedAlias = key
	r.loadedMode = mode
	return l, key, nil
}

func loadModeLabel(mode loadMode) string {
	if mode == loadModeEmbeddings {
		return "embeddings"
	}
	return "chat"
}

func (r *Registry) resolve(alias string, mode loadMode) (key, path string, err error) {
	key = NormalizeAlias(alias)
	if key == "" {
		if mode == loadModeEmbeddings {
			key = r.defaultEmbeddingAlias
			if key == "" && len(r.embeddingAliases) == 1 {
				for k := range r.embeddingAliases {
					key = k
					break
				}
			}
		} else {
			key = r.defaultAlias
		}
	}
	if key == "" {
		if mode == loadModeEmbeddings {
			return "", "", fmt.Errorf("модель не указана; для embeddings доступны: %s", strings.Join(r.ListEmbeddingAliases(), ", "))
		}
		return "", "", fmt.Errorf("модель не указана; доступны: %s", strings.Join(r.ListAliases(), ", "))
	}

	if mode == loadModeEmbeddings && !r.supportsEmbeddings(key) {
		return "", "", fmt.Errorf("%w: %q (укажите алиас из EmbeddingModels: %s)",
			ErrEmbeddingNotSupported, key, strings.Join(r.ListEmbeddingAliases(), ", "))
	}

	path, ok := r.aliases[key]
	if !ok {
		return "", "", fmt.Errorf("неизвестная модель %q; доступны: %s", key, strings.Join(r.ListAliases(), ", "))
	}
	return key, path, nil
}

type ModelDescriptor struct {
	ID           string
	Created      int64
	Capabilities []string
}

func ListDescriptors() ([]ModelDescriptor, error) {
	if registry == nil {
		return nil, fmt.Errorf("реестр моделей не инициализирован")
	}
	return registry.listDescriptors(), nil
}

func DescriptorByID(id string) (ModelDescriptor, error) {
	if registry == nil {
		return ModelDescriptor{}, fmt.Errorf("реестр моделей не инициализирован")
	}
	key := NormalizeAlias(id)
	path, ok := registry.aliases[key]
	if !ok {
		return ModelDescriptor{}, fmt.Errorf("%w: %q", ErrModelNotFound, key)
	}
	return registry.descriptor(key, path), nil
}

func (r *Registry) listDescriptors() []ModelDescriptor {
	aliases := r.ListAliases()
	out := make([]ModelDescriptor, 0, len(aliases))
	for _, alias := range aliases {
		out = append(out, r.descriptor(alias, r.aliases[alias]))
	}
	return out
}

func (r *Registry) descriptor(id, path string) ModelDescriptor {
	var created int64
	if info, err := os.Stat(path); err == nil {
		created = info.ModTime().Unix()
	}
	return ModelDescriptor{
		ID:           id,
		Created:      created,
		Capabilities: r.capabilities(id),
	}
}

func (r *Registry) capabilities(alias string) []string {
	caps := []string{"chat"}
	if r.supportsEmbeddings(alias) {
		caps = append(caps, "embeddings")
	}
	return caps
}

func openLlama(path string, embeddings bool) (*llama.LLama, error) {
	return llama.New(path, func(p *llama.ModelOptions) {
		applyModelOptions(p, embeddings)
	})
}

func applyModelOptions(p *llama.ModelOptions, embeddings bool) {
	defer func() {
		if err := recover(); err != nil {
			fmt.Println("panic occurred:", err)
			os.Exit(1)
		}
	}()
	src := config.C.ModelOption
	if embeddings && len(config.C.EmbeddingModelOption) > 0 {
		src = config.C.EmbeddingModelOption
	}
	for k, v := range src {
		if k == "Embeddings" {
			continue
		}
		ReflectVal(k, v, p)
	}
	p.Embeddings = embeddings
	if embeddings && len(config.C.EmbeddingModelOption) == 0 {
		applyEmbeddingSafeDefaults(p)
	}
	ApplyAutoGPUFit(p, src)
	if fitMinCtx, ok := optionalIntField(p, "FitParamsMinCtx"); ok && fitMinCtx <= 0 && p.NGPULayers < 0 && p.ContextSize > 0 {
		setOptionalIntField(p, "FitParamsMinCtx", p.ContextSize)
	}
	switch {
	case p.NGPULayers < 0:
		log.Printf("Model options: %+v (NGPULayers=%d → fit_params: слои и tensor_split по VRAM на всех GPU)", p, p.NGPULayers)
	default:
		log.Print("Model options: ", p)
	}
}

// applyEmbeddingSafeDefaults — chat ModelOption (32768 ctx, TensorSplit) ломает маленькие embed-модели.
func applyEmbeddingSafeDefaults(p *llama.ModelOptions) {
	ctx := config.C.RAG.EmbeddingContextSize
	if ctx <= 0 {
		ctx = 2048
	}
	p.ContextSize = ctx
	p.TensorSplit = ""
	p.NGPULayers = -1
	if p.NBatch < ctx {
		p.NBatch = ctx
	}
	log.Printf("models: embedding defaults: ContextSize=%d, NBatch=%d, TensorSplit=off, NGPULayers=-1", ctx, p.NBatch)
}
