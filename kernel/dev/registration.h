// Development shim for building the kernel sources without kernel-builder (torch.utils.cpp_extension / CMake).
// kernel-builder generates the real registration.h; never publish a build made with this shim.
#pragma once
#include <torch/library.h>
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define REGISTER_EXTENSION(NAME)
