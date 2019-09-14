#include "gotorch.h"

#include <torch/jit.h>
#include <torch/optim.h>
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

AtTensor TorchTensorFromBlob(void* data_ptr, int64_t* sizes_ptr,
                             int sizes_len) {
  std::vector<int64_t> sizes(sizes_ptr, sizes_ptr + sizes_len);
  at::TensorOptions opts(torch::kFloat32);
  return (void*)new torch::Tensor(torch::from_blob(data_ptr, sizes, opts));
}

AtTensor TorchRandN(int64_t* sizes_ptr, int sizes_len) {
  std::vector<int64_t> sizes(sizes_ptr, sizes_ptr + sizes_len);
  return (void*)new torch::Tensor(torch::randn(sizes));
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

void TorchTensorBackward(AtTensor tptr) {
  auto* t = static_cast<at::Tensor*>(tptr);
  t->backward();
}

AtTensor TorchTensorGrad(AtTensor tptr) {
  auto* t = static_cast<at::Tensor*>(tptr);
  return (void*)new torch::Tensor(t->grad());
}

void TorchTensorSetRequiresGrad(AtTensor tptr, bool requires_grad) {
  auto* t = static_cast<at::Tensor*>(tptr);
  t->set_requires_grad(requires_grad);
}

bool TorchTensorRequiresGrad(AtTensor tptr) {
  auto* t = static_cast<at::Tensor*>(tptr);
  return t->requires_grad();
}

namespace {
std::vector<Tensor> tensorsFromTensorPtrs(AtTensor* tptrs, int tcount) {
  std::vector<Tensor> params;
  for (size_t i = 0; i < tcount; i++) {
    params.emplace_back(*static_cast<at::Tensor*>(tptrs[i]));
  }
  return params;
}
}  // namespace

TorchOptimizer TorchAdam(AtTensor* tptrs, int tcount, float lr) {
  auto params = tensorsFromTensorPtrs(tptrs, tcount);
  optim::AdamOptions opts(lr);
  return (void*)new optim::Adam(std::move(params), opts);
}

TorchOptimizer TorchSGD(AtTensor* tptrs, int tcount, float lr) {
  auto params = tensorsFromTensorPtrs(std::move(tptrs), tcount);
  optim::SGDOptions opts(lr);
  return (void*)new optim::SGD(params, opts);
}

void TorchOptimizerDelete(TorchOptimizer optr) {
  auto* o = static_cast<optim::Optimizer*>(optr);
  delete o;
}

void TorchOptimizerZeroGrad(TorchOptimizer optr) {
  auto* o = static_cast<optim::Optimizer*>(optr);
  o->zero_grad();
}

void TorchOptimizerStep(TorchOptimizer optr) {
  auto* o = static_cast<optim::Optimizer*>(optr);
  o->step();
}

AtTensor TorchStack(AtTensor* tptrs, int tcount, int64_t dim) {
  auto tensors = tensorsFromTensorPtrs(tptrs, tcount);
  return (void*)new Tensor(stack(std::move(tensors), dim));
}

AtTensor TorchReshape(AtTensor aptr, int64_t* sizes_ptr, int sizes_len) {
  auto* a = static_cast<Tensor*>(aptr);
  std::vector<int64_t> sizes(sizes_ptr, sizes_ptr + sizes_len);
  return (void*)new Tensor(reshape(*a, sizes));
}

#define TENSOR_BI_IMPL(name, method)          \
  TENSOR_BI(name) {                           \
    auto* a = static_cast<Tensor*>(aptr);     \
    auto* b = static_cast<Tensor*>(bptr);     \
    return (void*)new Tensor(method(*a, *b)); \
  }

TENSOR_BI_IMPL(Dot, dot);
TENSOR_BI_IMPL(Add, add);
TENSOR_BI_IMPL(Sub, sub);
TENSOR_BI_IMPL(Div, div);
TENSOR_BI_IMPL(Eq, eq);
TENSOR_BI_IMPL(L1Loss, l1_loss);
TENSOR_BI_IMPL(NLLLoss, nll_loss);
TENSOR_BI_IMPL(MSELoss, mse_loss);
