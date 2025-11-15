.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
..
..   http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.

====================
Kernel Library Guide
====================

This guide serves as a quick start for shipping python version and machine learning(ML) framework agnostic kernel libraries with TVM FFI. With the help of TVM FFI, we can connect the kernel libraries to multiple ML framework, such as PyTorch, XLA, JAX, together with the minimal efforts.

Tensor
======

Almost all kernel libraries are about tensor computation and manipulation. For better adaptation to different ML frameworks, TVM FFI provides a minimal set of data structures to represent tensors from ML frameworks, including the tensor basic attributes and storage pointer. To be specific, in TVM FFI, two types of tensor constructs, ``ffi::Tensor`` and ``ffi::TensorView``, can be used to represent a tensor from ML frameworks.

Tensor and TensorView
---------------------

Though both ``ffi::Tensor`` and ``ffi::TensorView`` are designed to represent tensors from ML frameworks that interact with the TVM FFI ABI. The main difference is whether it is an owning tensor structure.

ffi::Tensor
 ``ffi::Tensor`` is a completely onwing tensor pointer, pointing to a TVM FFI tensor object. TVM FFI handles the lifetime of ``ffi::Tensor`` by retaining a strong reference.

ffi::TensorView
 ``ffi::TensorView`` is a non-owning view of an existing tensor, pointint to an existing ML framework tensor. It is backed by ``DLTensor`` structure in DLPack in practice. And TVM FFI does not guarantee its lifetime also.

It is **recommended** to use ``ffi::TensorView`` when possible, that helps us to support more cases, including cases where only view but not strong reference are passed, like XLA buffer. It is also more lightweight. However, since ``ffi::TensorView`` is a non-owning view, it is the user's responsibility to ensure the lifetime of underlying tensor data and attributes of the viewed tensor object.

Tensor Attributes
-----------------

For the sake of convenience, ``ffi::TensorView`` and ``ffi::Tensor`` align the following attributes retrieval mehtods to ``at::Tensor`` interface, to obtain tensor basic attributes and storage pointer:

``dim``, ``sizes``, ``size``, ``strides``, ``stride``, ``numel``, ``data_ptr``, ``device``, ``is_contiguous``

DLDataType
 In TVM FFI, tensor data types are stored as ``DLDataType`` which is defined by DLPack protocol.

DLDevice
 In TVM FFI, tensor device information are stored as ``DLDevice`` which is defined by DLPack protocol.

ShapeView
 In TVM FFI, tensor shapes and strides attributes retrieval are returned as ``ShapeView``. It is an iterate-able data structure storing the shapes or strides data as ``int64_t`` array.

Tensor Allocation
-----------------

TVM FFI provides several methods to allocate tensors at C++ runtime. Generally, there are two types of tensor allocation:

* Allocate a tensor with new storage from scratch, i.e. ``FromEnvAlloc`` and ``FromNDAlloc``. By this types of methods, the shapes, strides, data types, devices and other attributes are required for the allocation.
* Allocate a tensor with existing storage following DLPack protocol, i.e. ``FromDLPack`` and ``FromDLPackVersioned``. By this types of methods, the shapes, data types, devices and other attributes can be inferred from the DLPack attributes.

FromEnvAlloc
^^^^^^^^^^^^

To better adapt to the ML framework, it is **recommended** to reuse the framework tensor allocator anyway, instead of directly allocating the tensors via CUDA runtime API, like ``cudaMalloc``. Since reusing the framework tensor allocator:

* Benefit from the framework's native caching allocator or related allocation mechanism.
* Help framework tracking memory usage and planning globally.

For this case, TVM FFI provides ``FromEnvAlloc``. It internally calls the framework tensor allocator. To determine which framework tensor allocator, TVM FFI infers it from the passed-in framework tensors. For example, when calling the kernel library at Python side, there is an input framework tensor if of type ``torch.Tensor``, TVM FFI will automatically bind the ``at::empty`` as the current framework tensor allocator by ``TVMFFIEnvTensorAlloc``. And then the ``FromEnvAlloc`` is calling the ``at::empty`` actually:

.. code-block:: c++

 ffi::Tensor tensor = ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ...);

which is equivalent to:

.. code-block:: c++

 at::Tensor tensor = at::empty(...);

FromNDAlloc
^^^^^^^^^^^

``FromNDAlloc`` is the most basic tensor allocator. It is designed for simple cases where framework tensor allocator is no longer needed. ``FromNDAlloc`` just requires a custom allocator struct to handle the tensor allocation and free, with fixed interface ``void AllocData(DLTensor*)`` and ``void FreeData(DLTensor*)`` methods. Here are the examples of CPU, CUDA and NVSHMEM allocation:

.. code-block:: c++

 // CPU Allocator
 struct CPUNDAlloc {
   void AllocData(DLTensor* tensor) { tensor->data = malloc(ffi::GetDataSize(*tensor)); }
   void FreeData(DLTensor* tensor) { free(tensor->data); }
 };

 // CUDA Allocator
 struct CUDANDAlloc {
   void AllocData(DLTensor* tensor) {
     size_t data_size = ffi::GetDataSize(*tensor);
     void* ptr = nullptr;
     cudaError_t err = cudaMalloc(&ptr, data_size);
     TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
     tensor->data = ptr;
   }
   void FreeData(DLTensor* tensor) {
     if (tensor->data != nullptr) {
       cudaError_t err = cudaFree(tensor->data);
       TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
       tensor->data = nullptr;
     }
   }
 };

  // NVSHMEM Allocator
  struct NVSHMEMNDAlloc {
    void AllocData(DLTensor* tensor) {
      size_t size = tvm::ffi::GetDataSize(*tensor);
      tensor->data = nvshmem_malloc(size);
      TVM_FFI_ICHECK_NE(tensor->data, nullptr) << "nvshmem_malloc failed. size: " << size;
    }
    void FreeData(DLTensor* tensor) { nvshmem_free(tensor->data); }
  };

  // Allocator usage
  ffi::Tensor cpu_tensor = ffi::Tensor::FromNDAlloc(CPUNDAlloc(), ...);
  ffi::Tensor cuda_tensor = ffi::Tensor::FromNDAlloc(CUDANDAlloc(), ...);
  ffi::Tensor nvshmem_tensor = ffi::Tensor::FromNDAlloc(NVSHMEMNDAlloc(), ...);

FromDLPack
^^^^^^^^^^

``FromDLPack`` enables creating ``ffi::Tensor`` from ``DLManagedTensor*``, working with ``ToDLPack`` for DLPack C Tensor Object ``DLTensor`` exchange protocol. Both are used for DLPack pre V1.0 API. It is used for wrapping the existing framework tensor to ``ffi::Tensor``.

FromDLPackVersioned
^^^^^^^^^^^^^^^^^^^

``FromDLPackVersioned`` enables creating ``ffi::Tensor`` from ``DLManagedTensorVersioned*``, working with ``ToDLPackVersioned`` for DLPack C Tensor Object ``DLTensor`` exchange protocol. Both are used for DLPack post V1.0 API. It is used for wrapping the existing framework tensor to ``ffi::Tensor`` too.

Python Calling FFI
==================

As we already have our kernel library wrapped with TVM FFI interface, our next and final step is exporting kernel library to Python side and enabling interaction with runtime environment or context.

Function Exporting
------------------

TVM FFI provides macro ``TVM_FFI_DLL_EXPORT_TYPED_FUNC`` for exporting the kernel functions to the output library files. So that at Python side, it is possible to load the library files and call the kernel functions directly. For example, we export our kernels as:

.. code-block:: c++

 void func(ffi::TensorView input, ffi::TensorView output);
 TVM_FFI_DLL_EXPORT_TYPED_FUNC(func, func);

And then we compile the sources into ``func.so``, or ``func.dylib`` for macOS, or ``func.dll`` for Windows. Finally, we can load and call our kernel functions at Python side as:

.. code-block:: python

 mod = tvm_ffi.load_module("func.so")
 x = ...
 y = ...
 mod.func(x, y)

``x`` and ``y`` here can be any ML framework tensors, such as ``torch.Tensor``, ``numpy.NDArray``, ``cupy.ndarray``, or other tensors as long as TVM FFI supports. TVM FFI detects the tensor types in arguments and converts them into ``ffi::TensorView`` automatically. So that we do not have to write the specific conversion codes per framework.

In constrast, if the kernel function returns ``ffi::Tensor`` instead of ``void`` in the example above. TVM FFI automatically converts the output ``ffi::Tensor`` to framework tensors also. The output framework is inferred from the input framework tensors. For example, if the input framework tensors are of ``torch.Tensor``, TVM FFI will convert the output tensor to ``torch.Tensor``. And if none of the input tensors are from ML framework, the output tensor will be the ``tvm_ffi.core.Tensor`` as fallback.

Actually, it is **recommended** to pre-allocated input and output tensors from framework at Python side alreadly. So that the return type of kernel functions at C++ side should be ``void`` always.

Context Inherit
---------------

Also, when calling our kernel library at Python side, we usually need to pass the important context to the kernel library, for example, the CUDA stream context from ``torch.cuda.stream`` or ``torch.cuda.graph``. So that the kernels can be dispatched to the expected CUDA stream. TVM FFI has already made it by maintaining the stream context table per device type and index. And when converting the framework tensors as mentioned above, TVM FFI automatically updates the stream context table, by the device on which the converted framework tensors. For example, if there is an framework tensor as ``torch.Tensor(device="cuda:3")``, TVM FFI will automatically update the current stream of cuda device 3 to torch current context stream, by ``TVMFFIEnvSetStream``. And then at C++ side, we just use ``TVMFFIEnvGetStream`` to get the updated current stream on the specific device. Here is an example:

.. code-block:: c++

 void func(ffi::TensorView input, ...) {
   ffi::DLDevice device = input.device();
   cudaStream_t stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
 }

which is equivalent to:


.. code-block:: c++

 void func(at::Tensor input, ...) {
   c10::Device = input.device();
   cudaStream_t stream = reinterpret_cast<cudaStream_t>(c10::cuda::getCurrentCUDAStream(device.index()).stream());
 }
