package gotorch

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

func TensorFromBlob(data []float32, sizes []int64) (*Tensor, error) {
	if totalSize(sizes) != len(data) {
		return nil, errors.Errorf("input data doesn't match size")
	}
	t := tensorFromPtr(C.TorchTensorFromBlob(
		unsafe.Pointer(&data[0]),
		(*C.int64_t)(&sizes[0]),
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
