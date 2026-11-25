# idol_native: libtorch C++/CUDA build of the Idol pipeline.
#
# Links the libtorch shared libraries of the installed torch wheel (the same .so files the Python reference loads)
# and puts the wheel's nvidia/* library folders first on the RPATH, so cuBLAS/cuBLASLt/cuDNN are the same files.
# Image decoding links the libwebp bundled in the Pillow wheel (same decoder as the reference).
#
#   cmake -S native -B native/build -DTORCH_SITE=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))') \
#         -DPILLOW_LIBS=$(python -c 'import PIL,os;print(os.path.join(os.path.dirname(PIL.__file__),"..","pillow.libs"))')
#   cmake --build native/build -j
cmake_minimum_required(VERSION 3.24)
project(idol_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES "90a" CACHE STRING "CUDA architectures")

set(TORCH_SITE "" CACHE PATH "site-packages/torch of the reference environment")
set(PILLOW_LIBS "" CACHE PATH "site-packages/pillow.libs of the reference environment")
if(NOT TORCH_SITE)
  message(FATAL_ERROR "set -DTORCH_SITE=<site-packages>/torch")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_SITE}/share/cmake")
find_package(Torch REQUIRED)

# Determinism of host code: no fast-math, no FP contraction (the Pillow and numpy ports depend on it).
set(HOST_FLAGS -O2 -ffp-contract=off -fno-fast-math)

file(GLOB WEBP_LIB "${PILLOW_LIBS}/libwebp-*.so*")
file(GLOB WEBPDEMUX_LIB "${PILLOW_LIBS}/libwebpdemux-*.so*")
file(GLOB SHARPYUV_LIB "${PILLOW_LIBS}/libsharpyuv-*.so*")
if(NOT WEBP_LIB OR NOT WEBPDEMUX_LIB)
  message(FATAL_ERROR "libwebp/libwebpdemux not found in PILLOW_LIBS=${PILLOW_LIBS}")
endif()
find_package(ZLIB REQUIRED)

add_executable(idol_native
  src/main.cpp src/common.cpp src/dit.cpp src/decoders.cpp src/encoders.cpp src/image.cpp src/safetensors.cpp
  src/fast.cpp src/fast_kernels.cu ../kernel/idol_cuda/idol_kernels.cu
  third_party/torchvision/ops/deform_conv2d.cpp third_party/torchvision/ops/cuda/deform_conv2d_kernel.cu)
target_include_directories(idol_native PRIVATE src third_party/libwebp-1.6.0 third_party/torchvision ../kernel/idol_cuda)
target_compile_options(idol_native PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${HOST_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -lineinfo>)
target_link_libraries(idol_native PRIVATE ${TORCH_LIBRARIES} ${WEBP_LIB} ${WEBPDEMUX_LIB} ${SHARPYUV_LIB} ZLIB::ZLIB)
# make sure the torchvision op registration (static initialisers) is kept
target_link_options(idol_native PRIVATE -Wl,--no-as-needed)

# RPATH: the wheel's nvidia libraries first, then torch/lib, then pillow.libs
file(GLOB NV_LIB_DIRS LIST_DIRECTORIES true "${TORCH_SITE}/../nvidia/*/lib")
set(RP "")
foreach(d ${NV_LIB_DIRS})
  get_filename_component(d "${d}" REALPATH)
  list(APPEND RP "${d}")
endforeach()
list(APPEND RP "${TORCH_SITE}/lib" "${PILLOW_LIBS}")
set_target_properties(idol_native PROPERTIES BUILD_RPATH "${RP}" INSTALL_RPATH "${RP}" SKIP_BUILD_RPATH OFF)
target_link_options(idol_native PRIVATE -Wl,--disable-new-dtags)
