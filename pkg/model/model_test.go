package model

import (
	"reflect"
	"testing"

	llama "github.com/redirsvr/go-llama-new.cpp"
)

func TestReflectValYAMLInts(t *testing.T) {
	p := &llama.ModelOptions{}
	ReflectVal("NGPULayers", -1, p)
	if p.NGPULayers != -1 {
		t.Fatalf("NGPULayers = %d, want -1", p.NGPULayers)
	}
	ReflectVal("ContextSize", int64(8192), p)
	if p.ContextSize != 8192 {
		t.Fatalf("ContextSize = %d, want 8192", p.ContextSize)
	}
	ReflectVal("NBatch", float64(512), p)
	if p.NBatch != 512 {
		t.Fatalf("NBatch = %d, want 512", p.NBatch)
	}
	ReflectVal("FitParamsMinCtx", 32768, p)
	if got, ok := optionalIntField(p, "FitParamsMinCtx"); ok && got != 32768 {
		t.Fatalf("FitParamsMinCtx = %d, want 32768", got)
	}
	ReflectVal("KVOffload", false, p)
	if got, ok := optionalBoolFieldForTest(p, "KVOffload"); ok && got {
		t.Fatal("KVOffload = true, want false")
	}
}

func optionalBoolFieldForTest(dst interface{}, name string) (bool, bool) {
	rv := reflect.ValueOf(dst)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return false, false
	}
	field := rv.Elem().FieldByName(name)
	if !field.IsValid() || field.Kind() != reflect.Bool {
		return false, false
	}
	return field.Bool(), true
}
