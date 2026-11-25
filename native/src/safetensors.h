// Minimal safetensors reader/writer on top of libtorch (CPU tensors).
#pragma once
#include <ATen/ATen.h>

#include <map>
#include <string>

namespace idol {

// Reads every tensor of a .safetensors file into CPU tensors (owned copies).
std::map<std::string, at::Tensor> load_safetensors(const std::string& path);

// Writes CPU or CUDA tensors (copied to CPU, made contiguous) to a .safetensors file.
void save_safetensors(const std::string& path, const std::map<std::string, at::Tensor>& tensors,
                      const std::map<std::string, std::string>& metadata = {});

}  // namespace idol
