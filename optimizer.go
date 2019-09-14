package torch

// #include "gotorch.h"
import "C"
import (
	"runtime"
	"unsafe"
)

type Optimizer struct {
	ptr C.TorchOptimizer
}

func tensorToTensorPtrs(tensors []*Tensor) []C.AtTensor {
	tptrs := make([]C.AtTensor, len(tensors))
	for i, t := range tensors {
		tptrs[i] = t.ptr
	}
	return tptrs
}

func Adam(params []*Tensor, lr float32) *Optimizer {
	tptrs := tensorToTensorPtrs(params)
	return optimizerFromPtr(C.TorchAdam(
		(*C.AtTensor)(unsafe.Pointer(&tptrs[0])),
		C.int(len(params)),
		C.float(lr),
	))
}

func SGD(params []*Tensor, lr float32) *Optimizer {
	tptrs := tensorToTensorPtrs(params)
	return optimizerFromPtr(C.TorchSGD(
		(*C.AtTensor)(unsafe.Pointer(&tptrs[0])),
		C.int(len(params)),
		C.float(lr),
	))
}

func optimizerFromPtr(ptr C.TorchOptimizer) *Optimizer {
	o := &Optimizer{
		ptr: ptr,
	}
	runtime.SetFinalizer(o, func(o *Optimizer) {
		C.TorchOptimizerDelete(o.ptr)
		o.ptr = nil
	})
	return o
}

func (o *Optimizer) Step() {
	C.TorchOptimizerStep(o.ptr)
}

func (o *Optimizer) ZeroGrad() {
	C.TorchOptimizerZeroGrad(o.ptr)
}
