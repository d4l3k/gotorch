#!/bin/bash

set -ex

rm -r build || true
mkdir build

cp -r /usr/include/torch/csrc/api/include/* build/

replace () {
	find ./build -type f -name "*.h" -exec sed -i'' -e "$1" {} +
}

replace "s/typename =/typename A =/g"
replace "s/ = torch::enable_if_t<(sizeof...(Args) > 0)>//g"
replace "s/ReturnType (.*::\*)(ArgumentTypes...)/ReturnType/g"
replace "s/constexpr auto k/constexpr c10::ScalarType k/g"


swig -go -cgo -intgosize 64 -Ibuild -c++ torch.i

go build -v .
