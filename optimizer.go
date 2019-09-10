package gotorch

// #include "gotorch.h"
import "C"
import (
	"runtime"
	"unsafe"
)

type Optimizer struct {
	ptr C.TorchOptimizer
}

func Adam(params []*Tensor, lr float32) *Optimizer {
	var tptrs []C.AtTensor
	for _, t := range params {
		tptrs = append(tptrs, t.ptr)
	}
	return optimizerFromPtr(C.TorchAdam(
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
