package torch

// #include "gotorch.h"
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
)

type Tensor struct {
	ptr  C.AtTensor
	data []float32
}

func TensorFromScalar(f float32) *Tensor {
	t, err := TensorFromBlob([]float32{f}, nil)
	if err != nil {
		panic(err)
	}
	return t
}

func TensorFromBlob(data []float32, sizes []int64) (*Tensor, error) {
	if totalSize(sizes) != len(data) {
		return nil, errors.Errorf("input data doesn't match size")
	}
	var startPtr *C.int64_t
	if len(sizes) > 0 {
		startPtr = (*C.int64_t)(&sizes[0])
	}
	t := tensorFromPtr(C.TorchTensorFromBlob(
		unsafe.Pointer(&data[0]),
		startPtr,
		C.int(len(sizes)),
	))
	// we still own the data so we have to hold on to it
	t.data = data
	return t, nil
}

func tensorFromPtr(ptr C.AtTensor) *Tensor {
	t := &Tensor{
		ptr: ptr,
	}
	runtime.SetFinalizer(t, func(t *Tensor) {
		C.TorchTensorDelete(t.ptr)
		t.ptr = nil
		t.data = nil
	})
	return t
}

func (t *Tensor) Dim() int {
	return int(C.TorchTensorDim(t.ptr))
}

func (t *Tensor) Sizes() []int64 {
	if t.Dim() == 0 {
		return nil
	}
	sizes := make([]int64, t.Dim())
	C.TorchTensorSizes(t.ptr, (*C.int64_t)(&sizes[0]))
	return sizes
}

func totalSize(sizes []int64) int {
	length := 1
	for _, size := range sizes {
		length *= int(size)
	}
	return length
}

func (t *Tensor) Blob() []float32 {
	data_ptr := C.TorchTensorData(t.ptr)
	length := totalSize(t.Sizes())
	data := (*[1 << 28]C.float)(unsafe.Pointer(data_ptr))[:length:length]
	out := make([]float32, length)
	for i, v := range data {
		out[i] = float32(v)
	}
	return out
}

func (t *Tensor) Backward() {
	C.TorchTensorBackward(t.ptr)
}

func (t *Tensor) Grad() *Tensor {
	return tensorFromPtr(C.TorchTensorGrad(t.ptr))
}

func (t *Tensor) RequiresGrad() bool {
	return bool(C.TorchTensorRequiresGrad(t.ptr))
}

func (t *Tensor) SetRequiresGrad(requiresGrad bool) {
	C.TorchTensorSetRequiresGrad(t.ptr, C.bool(requiresGrad))
}

func RandN(sizes ...int64) *Tensor {
	return tensorFromPtr(C.TorchRandN(
		(*C.int64_t)(&sizes[0]),
		C.int(len(sizes)),
	))
}

func (t *Tensor) Dot(b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchDot(t.ptr, b.ptr))
}

func (t *Tensor) Add(b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchAdd(t.ptr, b.ptr))
}

func (t *Tensor) Sub(b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchSub(t.ptr, b.ptr))
}

func (t *Tensor) Div(b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchDiv(t.ptr, b.ptr))
}

func (t *Tensor) Eq(b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchEq(t.ptr, b.ptr))
}

func NLLLoss(a, b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchNLLLoss(a.ptr, b.ptr))
}

func L1Loss(a, b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchL1Loss(a.ptr, b.ptr))
}

func MSELoss(a, b *Tensor) *Tensor {
	return tensorFromPtr(C.TorchMSELoss(a.ptr, b.ptr))
}

func Stack(dim int64, tensors ...*Tensor) *Tensor {
	tptrs := tensorToTensorPtrs(tensors)
	return tensorFromPtr(C.TorchStack(
		(*C.AtTensor)(unsafe.Pointer(&tptrs[0])),
		C.int(len(tptrs)),
		C.int64_t(dim),
	))
}

func (t *Tensor) Reshape(sizes ...int64) *Tensor {
	return tensorFromPtr(C.TorchReshape(
		t.ptr,
		(*C.int64_t)(&sizes[0]),
		C.int(len(sizes)),
	))
}
