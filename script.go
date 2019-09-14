package torch

// #include "gotorch.h"
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
)

type Module struct {
	ptr C.TorchJitScriptModule
}

func Compile(script string) (*Module, error) {
	cs := C.CString(script)
	defer C.free(unsafe.Pointer(cs))

	res := C.TorchJitCompile(cs)
	if res.err != nil {
		defer C.free(unsafe.Pointer(res.err))
		return nil, errors.Errorf(C.GoString(res.err))
	}

	m := &Module{
		ptr: res.res,
	}
	runtime.SetFinalizer(m, func(m *Module) {
		C.TorchJitScriptModuleDelete(m.ptr)
		m.ptr = nil
	})
	return m, nil
}

func (m *Module) RunMethod(name string, inputs []*Tensor) *Tensor {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	var input_ptrs []C.AtTensor
	for _, in := range inputs {
		input_ptrs = append(input_ptrs, in.ptr)
	}
	return tensorFromPtr(
		C.TorchJitScriptModuleRunMethod(m.ptr, cname, &input_ptrs[0], C.int(len(input_ptrs))))
}
