#include "gotorch.h"

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <sstream>

using namespace torch;

TorchJitCompileResult TorchJitCompile(char* cscript) {
  TorchJitCompileResult res = {};
  std::string script(cscript);
  try {
    auto cu = jit::compile(script);
    assert(cu != nullptr);
    res.res = (void*)new jit::script::CompilationUnit(std::move(*cu));
  } catch (const std::exception& e) {
    res.err = strdup(e.what());
  }
  return res;
}

void TorchJitScriptModuleDelete(TorchJitScriptModule mptr) {
  delete static_cast<jit::script::CompilationUnit*>(mptr);
}

AtTensor TorchJitScriptModuleRunMethod(TorchJitScriptModule mptr, char* cmethod,
				       AtTensor* valsptr, int num_vals) {
  auto* m = static_cast<jit::script::CompilationUnit*>(mptr);
  std::vector<IValue> vals;
  vals.reserve(num_vals);
  for (int i = 0; i < num_vals; i++) {
    auto* val = static_cast<at::Tensor*>(valsptr[i]);
    vals.push_back(*val);
  }
  std::string method(cmethod);
  auto res = m->get_function(method)(vals);
  return (void*)new at::Tensor(res.toTensor());
}

void tensorDeleter(void* ptr) { free(ptr); }

AtTensor TorchTensorFromBlob(void* data_ptr, int64_t* sizes_ptr,
			     int sizes_len) {
  std::vector<int64_t> sizes(sizes_ptr, sizes_ptr + sizes_len);
  at::TensorOptions opts(torch::kFloat32);
  return (void*)new torch::Tensor(
      torch::from_blob(data_ptr, sizes, tensorDeleter, opts));
}

void TorchTensorDelete(AtTensor tptr) { delete static_cast<at::Tensor*>(tptr); }

int TorchTensorDim(AtTensor tptr) {
  auto* t = static_cast<at::Tensor*>(tptr);
  return t->dim();
}

void TorchTensorSizes(AtTensor tptr, int64_t* data) {
  auto* t = static_cast<at::Tensor*>(tptr);
  auto sizes = t->sizes();
  for (size_t i = 0; i < sizes.size(); i++) {
    data[i] = sizes.at(i);
  }
}

void* TorchTensorData(AtTensor tptr) {
  auto* t = static_cast<at::Tensor*>(tptr);
  return (void*)t->data<float>();
}
