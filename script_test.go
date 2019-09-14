package torch

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
	a.SetRequiresGrad(true)
	optim := SGD([]*Tensor{a}, 0.1)
	optim.ZeroGrad()
	b, err := TensorFromBlob([]float32{2, 3, -2}, []int64{3})
	if err != nil {
		t.Fatal(err)
	}
	out := m.RunMethod("relu_script", []*Tensor{a, b})
	outData := out.Blob()
	if !reflect.DeepEqual(outData, []float32{3, 5, 0}) {
		t.Error("output wrong", outData)
	}
	out.Backward()

	grad := a.Grad()
	gradData := grad.Blob()
	if !reflect.DeepEqual(gradData, []float32{1, 1, 0}) {
		t.Error("output wrong", gradData)
	}

	optim.Step()

	aData := a.Blob()
	if !reflect.DeepEqual(aData, []float32{0.9, 1.9, -1}) {
		t.Error("output wrong", aData)
	}
}

func TestScriptRequiresGrad(t *testing.T) {
	a, err := TensorFromBlob([]float32{1, 2, -1}, []int64{3})
	if err != nil {
		t.Fatal(err)
	}
	if a.RequiresGrad() != false {
		t.Errorf("should have not required grad")
	}
	a.SetRequiresGrad(true)
	if a.RequiresGrad() != true {
		t.Errorf("should have required grad")
	}
}

func TestStack(t *testing.T) {
	a, err := TensorFromBlob([]float32{1}, []int64{1})
	if err != nil {
		t.Fatal(err)
	}
	b, err := TensorFromBlob([]float32{1}, []int64{1})
	if err != nil {
		t.Fatal(err)
	}

	out := Stack(0, a, b)
	data := out.Blob()
	if !reflect.DeepEqual(data, []float32{1, 1}) {
		t.Error("output wrong", data)
	}
	dims := out.Sizes()
	if !reflect.DeepEqual(dims, []int64{2, 1}) {
		t.Error("output wrong", dims)
	}
}

func TestReshape(t *testing.T) {
	a, err := TensorFromBlob([]float32{1, 2, 3, 4, 5, 6}, []int64{6})
	if err != nil {
		t.Fatal(err)
	}

	out := a.Reshape(2, 3)
	dims := out.Sizes()
	if !reflect.DeepEqual(dims, []int64{2, 3}) {
		t.Error("output wrong", dims)
	}
}

func TestScalar(t *testing.T) {
	a := TensorFromScalar(1)
	if a.Dim() != 0 {
		t.Error("expected 0 dim")
	}

	out := a.Blob()
	if !reflect.DeepEqual(out, []float32{1}) {
		t.Error("output wrong", out)
	}
}
