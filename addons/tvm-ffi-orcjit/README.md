<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM-FFI OrcJIT

A Python package that enables dynamic loading of TVM-FFI exported object files (`.o`) using LLVM ORC JIT v2.

## Features

- **Dynamic Loading**: Load compiled object files at runtime using LLVM's ORC JIT v2
- **Incremental Loading**: Add multiple object files to the same loader instance
- **TVM-FFI Integration**: Seamlessly works with TVM-FFI's stable C ABI
- **Python API**: Simple Pythonic interface for loading and calling compiled functions
- **Standalone Package**: Works alongside apache-tvm-ffi without conflicts

## Installation

### Prerequisites

- Python 3.8+
- CMake 3.18+
- LLVM 14+ (with ORC JIT support)
- Ninja build system (recommended)

### Build from Source

1. Clone the repository with submodules:

```bash
git clone --recursive https://github.com/apache/tvm-ffi.git
cd tvm-ffi/addons/tvm-ffi-orcjit
```

1. Build TVM-FFI dependency (from the root of tvm-ffi repository):

```bash
cd ../..  # Go to tvm-ffi root
mkdir -p build && cd build
cmake .. -G Ninja
ninja
cd addons/tvm-ffi-orcjit
```

1. Create build directory and configure with CMake:

```bash
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -G Ninja
```

1. Build the project:

```bash
cmake --build . -j$(nproc)
cd ..
```

The shared library will be created at: `build/libtvm_ffi_orcjit.so`

1. Install the Python package:

```bash
# Using pip
pip install .

# Or for development (editable install)
pip install -e .
```

## Usage

### Basic Example

```python
from tvm_ffi_orcjit import ObjectLoader

# Create a loader instance
loader = ObjectLoader()

# Load an object file
loader.load("example.o")

# Get and call a function
add_func = loader.get_function("simple_add")
result = add_func(1, 2)
print(f"Result: {result}")  # Output: Result: 3
```

### Incremental Loading

Load multiple object files and access functions from all of them:

```python
from tvm_ffi_orcjit import ObjectLoader

loader = ObjectLoader()

# Load first object file
loader.load("math_ops.o")
add = loader.get_function("simple_add")

# Load second object file - functions from first remain accessible
loader.load("string_ops.o")
concat = loader.get_function("string_concat")

# Both functions work
print(add(10, 20))           # From math_ops.o
print(concat("Hello", "World"))  # From string_ops.o
```

### Direct Module Access

You can also use TVM-FFI's `load_module` directly (`.o` files are automatically handled):

```python
import tvm_ffi

# Load object file as a module
module = tvm_ffi.load_module("example.o")

# Get function
func = module.get_function("my_function")
result = func(arg1, arg2)
```

## How It Works

1. **C++ Backend**: The package implements a `Library` subclass using LLVM's ORC JIT v2 (`LLJIT`)
2. **Registration**: Registers with TVM-FFI as a loader for `.o` files via `ffi.Module.load_from_file.o`
3. **Symbol Resolution**: Uses LLJIT's `lookup()` to resolve TVM-FFI exported symbols
4. **Module Integration**: Wraps the ORC JIT library in `LibraryModuleObj` for seamless TVM-FFI integration

## Development

### Building with CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build .
```

### Project Structure

```text
tvm-ffi-orcjit/
├── CMakeLists.txt              # CMake build configuration
├── pyproject.toml              # Python package metadata
├── README.md                   # This file
├── example.py                  # Usage example
├── include/
│   └── tvm/ffi/orcjit/
│       └── orcjit_library.h    # C++ header
├── src/
│   └── ffi/
│       └── orcjit_library.cc   # C++ implementation
└── python/
    └── tvm_ffi_orcjit/
        ├── __init__.py         # Module exports
        └── session.py          # Python ExecutionSession class
        └── dylib.py            # Python DynamicLibrary class
```

## Examples

See `example.py` for a complete demonstration of incremental loading.

## Requirements

The package depends on:

- `apache-tvm-ffi>=0.1.0` - TVM-FFI library
- LLVM 14+ (linked at build time) - For ORC JIT functionality

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please ensure that:

1. Code follows the existing style
2. New features include tests
3. Documentation is updated

## Troubleshooting

### Symbol not found errors

Make sure your object file was compiled with TVM-FFI export macros:

```cpp
#include <tvm/ffi/c_ffi_api.h>

TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_function, [](int a, int b) {
    return a + b;
});
```

### LLVM version mismatch

Ensure the LLVM version used to build this package matches your system's LLVM installation.

### TVM-FFI not found

Make sure TVM-FFI is built in the parent repository:

```bash
cd ../../  # Go to tvm-ffi root
mkdir -p build && cd build
cmake .. -G Ninja && ninja
```
