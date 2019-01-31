#include "gotorch.h"

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

using namespace torch;

TorchJitCompileResult TorchJitCompile(char* cscript) {
  TorchJitCompileResult res = {};
  std::string script(cscript);
  try {
    std::shared_ptr<jit::script::Module> module = jit::compile(script);
    assert(module != nullptr);
    res.res = (void*)new std::shared_ptr<jit::script::Module>(module);
  } catch (std::exception& e) {
    res.err = strdup(e.what());
  }
  return res;
}

void TorchJitScriptModuleDelete(TorchJitScriptModule mptr) {
  delete static_cast<std::shared_ptr<jit::script::Module>*>(mptr);
}

AtTensor TorchJitScriptModuleRunMethod(TorchJitScriptModule mptr, char* cmethod,
                                       AtTensor* valsptr, int num_vals) {
  auto* m = static_cast<std::shared_ptr<jit::script::Module>*>(mptr);
  std::vector<IValue> vals;
  vals.reserve(num_vals);
  for (int i = 0; i < num_vals; i++) {
    auto* val = static_cast<at::Tensor*>(valsptr[i]);
    vals.push_back(*val);
  }
  std::string method(cmethod);
  auto res = (*m)->get_method(method)(vals);
  return (void*)new at::Tensor(res.toTensor());
}

AtTensor TorchTensorFromBlob(void* data_ptr, int64_t* sizes_ptr,
                             int sizes_len) {
  std::vector<int64_t> sizes(sizes_ptr, sizes_ptr + sizes_len);
  return (void*)new at::Tensor(torch::autograd::make_variable(
      at::getType(at::kFloat).tensorFromBlob(data_ptr, sizes)));
}

void TorchTensorDelete(AtTensor tptr) { delete static_cast<at::Tensor*>(tptr); }

int TorchTensorDim(AtTensor tptr) {
  auto* t = static_cast<at::Tensor*>(tptr);
  return t->dim();
}

void TorchTensorSizes(AtTensor tptr, int64_t* data) {
  auto* t = static_cast<at::Tensor*>(tptr);
  auto sizes = t->sizes();
  for (int i = 0; i < sizes.size(); i++) {
    data[i] = sizes.at(i);
  }
}

void* TorchTensorData(AtTensor tptr) {
  auto* t = static_cast<at::Tensor*>(tptr);
  return (void*)t->data<float>();
}
