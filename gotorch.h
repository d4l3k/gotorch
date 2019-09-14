#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* TorchJitScriptModule;
typedef void* AtTensor;
typedef void* TorchOptimizer;

typedef struct {
  TorchJitScriptModule res;
  const char* err;
} TorchJitCompileResult;
TorchJitCompileResult TorchJitCompile(char*);

void TorchJitScriptModuleDelete(TorchJitScriptModule);
AtTensor TorchJitScriptModuleRunMethod(TorchJitScriptModule, char*, AtTensor*,
                                       int);

AtTensor TorchTensorFromBlob(void*, int64_t*, int);
AtTensor TorchRandN(int64_t* sizes, int count);

int TorchTensorDim(AtTensor);
void TorchTensorDelete(AtTensor);
void TorchTensorSizes(AtTensor tptr, int64_t* data);
void* TorchTensorData(AtTensor tptr);
void TorchTensorBackward(AtTensor tptr);
AtTensor TorchTensorGrad(AtTensor tptr);
void TorchTensorSetRequiresGrad(AtTensor tptr, bool requires_grad);
bool TorchTensorRequiresGrad(AtTensor tptr);
AtTensor TorchStack(AtTensor* tptrs, int tcount, int64_t dim);
AtTensor TorchReshape(AtTensor aptr, int64_t* sizes_ptr, int sizes_len);

TorchOptimizer TorchAdam(AtTensor* tptrs, int tcount, float lr);
TorchOptimizer TorchSGD(AtTensor* tptrs, int tcount, float lr);
void TorchOptimizerDelete(TorchOptimizer optr);
void TorchOptimizerZeroGrad(TorchOptimizer optr);
void TorchOptimizerStep(TorchOptimizer optr);

#define TENSOR_BI(name) AtTensor Torch##name(AtTensor aptr, AtTensor bptr)

TENSOR_BI(Dot);
TENSOR_BI(Add);
TENSOR_BI(Sub);
TENSOR_BI(Div);
TENSOR_BI(Eq);
TENSOR_BI(L1Loss);
TENSOR_BI(NLLLoss);
TENSOR_BI(MSELoss);

#ifdef __cplusplus
} /* end extern "C" */
#endif
