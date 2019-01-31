package gotorch

import (
	"reflect"
	"testing"
)

func TestCompileError(t *testing.T) {
	if _, err := Compile(`some garbo`); err == nil {
		t.Fatalf("garbage compiled")
	}
}

func TestScript(t *testing.T) {
	m, err := Compile(`def relu_script(a, b):
		return torch.relu(a + b)
	`)
	if err != nil {
		t.Fatal(err)
	}
	a, err := TensorFromBlob([]float32{1, 2, -1}, []int64{3})
	if err != nil {
		t.Fatal(err)
	}
	b, err := TensorFromBlob([]float32{2, 3, -2}, []int64{3})
	if err != nil {
		t.Fatal(err)
	}
	out := m.RunMethod("relu_script", []*Tensor{a, b})
	outData := out.Blob()
	if !reflect.DeepEqual(outData, []float32{3, 5, 0}) {
		t.Fatal("output wrong", outData)
	}
}
