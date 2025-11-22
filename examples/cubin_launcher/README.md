# CUBIN Launcher Example

This example demonstrates how to use the `tvm::ffi::CubinModule` and `tvm::ffi::CubinKernel` classes to load and execute CUDA kernels from CUBIN files.

## Overview

The `cubin_launcher.h` header provides a lightweight C++ wrapper around CUDA Driver API for:
- Loading CUBIN modules from memory or files
- Managing kernel functions
- Launching kernels with specified configurations
- Multi-GPU support using CUDA primary contexts

## Files

- `src/kernel.cu` - Simple CUDA kernels (`add_one_cuda`, `mul_two_cuda`)
- `src/main_embedded.cc` - Example using embedded CUBIN data (linked at compile time)
- `src/main_dynamic.cc` - Example loading CUBIN from file at runtime
- `CMakeLists.txt` - Build configuration

## Building

### Prerequisites

- CUDA Toolkit (with driver API support)
- CMake 3.20+
- TVM-FFI installed (`pip install tvm-ffi`)

### Build Commands

```bash
# Configure the build
mkdir build && cd build
cmake ..

# Build the examples
make

# This will generate:
# - kernel.cubin (the compiled CUDA kernel)
# - main_embedded (executable with embedded CUBIN)
# - main_dynamic (executable that loads CUBIN from file)
```

## Running

### Embedded Example

The embedded example links the CUBIN data directly into the executable:

```bash
./main_embedded
```

**Output:**
```
=== Testing add_one_cuda kernel with embedded CUBIN ===
CUBIN size: XXXX bytes
CUBIN module loaded successfully
Kernel 'add_one_cuda' loaded successfully
Launching kernel with grid(4) block(256)
Kernel launched successfully
✓ Verification passed! All 1024 elements computed correctly.

=== Testing mul_two_cuda kernel with embedded CUBIN ===
...
```

### Dynamic Example

The dynamic example loads the CUBIN file at runtime:

```bash
./main_dynamic
# Or specify CUBIN path:
./main_dynamic kernel.cubin
```

**Output:**
```
CUBIN Launcher Dynamic Loading Example
=======================================

=== Testing CUBIN loading from file ===
Loading CUBIN from: kernel.cubin
CUBIN module loaded successfully
...
✓ All tests passed!
```

## API Usage

### Basic Usage Pattern

```cpp
#include <tvm/ffi/extra/cubin_launcher.h>

// Load CUBIN from memory
extern "C" const char cubin_data[];
extern "C" const uint64_t cubin_size;
tvm::ffi::CubinModule mod(cubin_data, cubin_size);

// Or load from file
tvm::ffi::CubinModule mod("kernel.cubin");

// Get kernel
tvm::ffi::CubinKernel kernel = mod["kernel_name"];

// Prepare arguments
void* args[] = {&ptr1, &ptr2, &n};

// Launch configuration
tvm::ffi::dim3 grid(blocks);
tvm::ffi::dim3 block(threads);

// Get CUDA stream
CUstream stream = (CUstream)TVMFFIEnvGetStream(kDLCUDA, device_id);

// Launch kernel
CUresult result = kernel.Launch(args, grid, block, stream);
TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
```

### Error Handling

Use the `TVM_FFI_CHECK_CUDA_DRIVER_ERROR` macro for checking CUDA Driver API results:

```cpp
CUresult result = kernel.Launch(args, grid, block, stream);
TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
// Throws RuntimeError with detailed message if result != CUDA_SUCCESS
```

### Multi-GPU Support

The launcher uses CUDA primary contexts and supports multi-GPU execution:

```cpp
// Each device has its own context
// The kernel will execute on the current device context
CUstream stream = (CUstream)TVMFFIEnvGetStream(kDLCUDA, device_id);
kernel.Launch(args, grid, block, stream);
```

## Implementation Details

### CUBIN Embedding

The CMakeLists.txt uses `ld` and `objcopy` to embed the CUBIN binary:

1. Compile `kernel.cu` to CUBIN
2. Use `ld -r -b binary` to create an object file
3. Use `objcopy` to rename symbols (`_binary_*` → `__cubin_data`)
4. Link the object file into the final executable

### CUDA Driver API

The implementation uses:
- `cuLibraryLoadData` - Load CUBIN from memory
- `cuLibraryGetKernel` - Get kernel handle from library
- `cuKernelGetFunction` - Get function handle for current context
- `cuLaunchKernel` - Launch the kernel

## Notes

- The `CubinModule` and `CubinKernel` classes are movable but not copyable (RAII)
- `cuInit(0)` is called automatically in `CubinModule` constructor
- Kernel handles are context-independent; function handles are context-specific
- Static initialization is recommended for module/kernel loading (load once, use many times)

## See Also

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [Library Management API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html)
- TVM-FFI documentation
