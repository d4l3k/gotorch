package gotorch

// #cgo CPPFLAGS: -I/usr/lib/python3.7/site-packages/torch/lib/include/
// #cgo LDFLAGS: -L/usr/lib/python3.7/site-packages/torch/lib -ltorch -lc10
// #include "gotorch.h"
import "C"
