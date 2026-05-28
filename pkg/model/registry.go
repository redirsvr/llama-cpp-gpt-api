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

type loadedEntry struct {
	ll *llama.LLama
	mu sync.Mutex
}

type Registry struct {
	mu                    sync.RWMutex
	dir                   string
	aliases               map[string]string
	embeddingAliases      map[string]struct{}
	loaded                map[string]*loadedEntry
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
		loaded:                make(map[string]*loadedEntry),
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
	}
	if err := registry.applyPresetAliases(singleFile); err != nil {
		return err
	}
	if singleFile == "" {
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
			log.Printf("models: предзагрузка %q (chat)", registry.defaultAlias)
			if key, path, err := registry.resolve(registry.defaultAlias, loadModeChat); err == nil {
				if _, err := registry.getOrLoad(key, path, loadModeChat); err != nil {
					return err
				}
			} else {
				return err
			}
		}
	}
	if registry.defaultEmbeddingAlias != "" {
		if config.C.PreloadDefaultModel {
			log.Printf("models: предзагрузка %q (embeddings)", registry.defaultEmbeddingAlias)
			if key, path, err := registry.resolve(registry.defaultEmbeddingAlias, loadModeEmbeddings); err == nil {
				if _, err := registry.getOrLoad(key, path, loadModeEmbeddings); err != nil {
					return err
				}
			} else {
				return err
			}
		}
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

func (r *Registry) applyPresetAliases(singleFile string) error {
	for key, preset := range config.ModelPresets {
		alias := NormalizeAlias(preset.Alias)
		if alias == "" {
			alias = NormalizeAlias(key)
		}
		if alias == "" {
			return fmt.Errorf("models-preset: пустой Alias для %q", key)
		}

		file := strings.TrimSpace(preset.File)
		if file == "" {
			file = key
		}
		path, err := r.resolvePresetFile(file, singleFile)
		if err != nil {
			return fmt.Errorf("models-preset %q: %w", alias, err)
		}
		if prev, dup := r.aliases[alias]; dup && prev != path {
			return fmt.Errorf("models-preset: алиас %q уже указывает на %s, нельзя заменить на %s", alias, prev, path)
		}
		r.aliases[alias] = path
	}
	return nil
}

func (r *Registry) resolvePresetFile(file, singleFile string) (string, error) {
	file = strings.TrimSpace(file)
	if file == "" {
		return "", fmt.Errorf("не задан File")
	}
	var path string
	switch {
	case filepath.IsAbs(file):
		path = file
	case r.dir != "":
		path = filepath.Join(r.dir, file)
	case singleFile != "":
		path = filepath.Join(filepath.Dir(singleFile), file)
	default:
		path = file
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(abs); err != nil {
		return "", fmt.Errorf("File %q: %w", file, err)
	}
	return abs, nil
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

func loadModeKey(alias string, mode loadMode) string {
	return alias + ":" + loadModeLabel(mode)
}

// UseChat выполняет fn с эксклюзивным доступом к модели.
// Разные модели могут работать параллельно.
func UseChat(alias string, fn func(*llama.LLama, string) error) error {
	return useModel(alias, loadModeChat, fn)
}

// UseEmbeddings — как UseChat, но для embedding-модели.
func UseEmbeddings(alias string, fn func(*llama.LLama, string) error) error {
	return useModel(alias, loadModeEmbeddings, fn)
}

func useModel(alias string, mode loadMode, fn func(*llama.LLama, string) error) error {
	if registry == nil {
		return fmt.Errorf("реестр моделей не инициализирован")
	}

	key, path, err := registry.resolve(alias, mode)
	if err != nil {
		return err
	}

	entry, err := registry.getOrLoad(key, path, mode)
	if err != nil {
		return err
	}

	entry.mu.Lock()
	defer entry.mu.Unlock()
	return fn(entry.ll, key)
}

func (r *Registry) getOrLoad(key, path string, mode loadMode) (*loadedEntry, error) {
	loadKey := loadModeKey(key, mode)

	r.mu.RLock()
	entry, ok := r.loaded[loadKey]
	r.mu.RUnlock()
	if ok {
		return entry, nil
	}

	r.mu.Lock()
	entry, ok = r.loaded[loadKey]
	if ok {
		r.mu.Unlock()
		return entry, nil
	}

	embeddings := mode == loadModeEmbeddings
	modeLabel := loadModeLabel(mode)
	log.Printf("models: загрузка %q из %s (embeddings=%v)", key, path, embeddings)
	loadStart := time.Now()
	l, err := openLlama(key, path, embeddings)
	loadDur := time.Since(loadStart)
	if err != nil {
		r.mu.Unlock()
		metrics.ObserveModelLoad(key, modeLabel, "error", loadDur)
		return nil, err
	}
	metrics.ObserveModelLoad(key, modeLabel, "success", loadDur)
	metrics.SetModelLoaded(key, modeLabel, true)
	entry = &loadedEntry{ll: l}
	r.loaded[loadKey] = entry
	r.mu.Unlock()
	return entry, nil
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

func openLlama(alias, path string, embeddings bool) (*llama.LLama, error) {
	return llama.New(path, func(p *llama.ModelOptions) {
		applyModelOptions(p, alias, path, embeddings)
	})
}

func applyModelOptions(p *llama.ModelOptions, alias, path string, embeddings bool) {
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
	preset, hasPreset := modelPresetFor(alias, path)
	hasPresetEmbeddingOptions := hasPreset && len(preset.EmbeddingModelOption) > 0
	src = modelOptionsWithPreset(src, alias, path, embeddings)
	for k, v := range src {
		if k == "Embeddings" {
			continue
		}
		ReflectVal(k, v, p)
	}
	p.Embeddings = embeddings
	if embeddings && len(config.C.EmbeddingModelOption) == 0 && !hasPresetEmbeddingOptions {
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

func modelOptionsWithPreset(base map[string]interface{}, alias, path string, embeddings bool) map[string]interface{} {
	preset, ok := modelPresetFor(alias, path)
	if !ok {
		return base
	}
	override := preset.ModelOption
	if embeddings && len(preset.EmbeddingModelOption) > 0 {
		override = preset.EmbeddingModelOption
	}
	if len(override) == 0 {
		return base
	}
	out := make(map[string]interface{}, len(base)+len(override))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range override {
		out[k] = v
	}
	log.Printf("models: preset %q применён к %q (embeddings=%v)", alias, filepath.Base(path), embeddings)
	return out
}

func modelPresetFor(alias, path string) (config.ModelPreset, bool) {
	keys := []string{
		alias,
		NormalizeAlias(alias),
		filepath.Base(path),
		NormalizeAlias(filepath.Base(path)),
	}
	for _, key := range keys {
		if key == "" {
			continue
		}
		if preset, ok := config.ModelPresets[key]; ok {
			return preset, true
		}
	}
	return config.ModelPreset{}, false
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
