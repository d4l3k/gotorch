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

typedef struct {
  TorchJitScriptModule res;
  const char* err;
} TorchJitCompileResult;
TorchJitCompileResult TorchJitCompile(char*);

void TorchJitScriptModuleDelete(TorchJitScriptModule);
AtTensor TorchJitScriptModuleRunMethod(TorchJitScriptModule, char*, AtTensor*,
                                       int);

AtTensor TorchTensorFromBlob(void*, int64_t*, int);
int TorchTensorDim(AtTensor);
void TorchTensorDelete(AtTensor);
void TorchTensorSizes(AtTensor tptr, int64_t* data);
void* TorchTensorData(AtTensor tptr);

#ifdef __cplusplus
} /* end extern "C" */
#endif