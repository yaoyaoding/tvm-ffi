cmake_minimum_required(VERSION 3.18)
project(
  tvm_ffi_orcjit
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add LLVM definitions and include directories early
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})

# Find tvm-ffi Look for tvm-ffi in the parent repository first, then system
set(TVM_FFI_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE PATH "Path to tvm-ffi"
)

if (EXISTS "${TVM_FFI_ROOT}/include/tvm/ffi/c_api.h")
  message(STATUS "Using tvm-ffi from: ${TVM_FFI_ROOT}")

  # Create imported target for tvm_ffi
  add_library(tvm_ffi SHARED IMPORTED)
  set_target_properties(
    tvm_ffi
    PROPERTIES IMPORTED_LOCATION "${TVM_FFI_ROOT}/build/lib/libtvm_ffi.so"
               INTERFACE_INCLUDE_DIRECTORIES
               "${TVM_FFI_ROOT}/include;${TVM_FFI_ROOT}/3rdparty/dlpack/include"
  )

  # Set include directories (including src for internal headers)
  include_directories(${TVM_FFI_ROOT}/include)
  include_directories(${TVM_FFI_ROOT}/src)
  include_directories(${TVM_FFI_ROOT}/3rdparty/dlpack/include)
else ()
  message(STATUS "Looking for system tvm-ffi")
  find_package(tvm-ffi REQUIRED CONFIG)
endif ()

# LLVM components needed for ORC JIT v2
llvm_map_components_to_libnames(LLVM_LIBS Core OrcJIT Support native)

# Filter out non-existent targets
set(LLVM_LIBS_FILTERED)
foreach (lib ${LLVM_LIBS})
  if (TARGET ${lib})
    list(APPEND LLVM_LIBS_FILTERED ${lib})
  else ()
    message(STATUS "Skipping non-existent LLVM target: ${lib}")
  endif ()
endforeach ()
set(LLVM_LIBS ${LLVM_LIBS_FILTERED})

# Source files
set(SOURCES src/ffi/orcjit_session.cc src/ffi/orcjit_dylib.cc)

# Build shared library
add_library(tvm_ffi_orcjit SHARED ${SOURCES})

target_include_directories(
  tvm_ffi_orcjit
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(
  tvm_ffi_orcjit
  PUBLIC tvm_ffi
  PRIVATE LLVM
)

# Compile definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS_LIST})

# Installation rules
install(
  TARGETS tvm_ffi_orcjit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
)

# For Python package building
if (SKBUILD)
  # Install shared library alongside Python modules
  install(TARGETS tvm_ffi_orcjit LIBRARY DESTINATION .)
endif ()
