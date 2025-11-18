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

Almost all kernel libraries are about tensor computation and manipulation. For better adaptation to different ML frameworks, TVM FFI provides a minimal set of data structures to represent tensors from ML frameworks, including the tensor basic attributes and storage pointer.
To be specific, in TVM FFI, two types of tensor constructs, :cpp:class:`~tvm::ffi::Tensor` and :cpp:class:`~tvm::ffi::TensorView`, can be used to represent a tensor from ML frameworks.

Tensor and TensorView
---------------------

Though both :cpp:class:`~tvm::ffi::Tensor` and :cpp:class:`~tvm::ffi::TensorView` are designed to represent tensors from ML frameworks that interact with the TVM FFI ABI. They are backed by the `DLTensor` in DLPack in practice. The main difference is whether it is an owning tensor structure.

:cpp:class:`tvm::ffi::Tensor`
 :cpp:class:`~tvm::ffi::Tensor` is a completely owning tensor with reference counting. It can be created and passed between C++ and Python side safely. When the counting reference goes to zero, its underlying deleter function will be called to free the tensor storage.

:cpp:class:`tvm::ffi::TensorView`
 :cpp:class:`~tvm::ffi::TensorView` is a non-owning view of an existing tensor, pointing to an existing tensor (e.g., a tensor allocated by PyTorch).

It is **recommended** to use :cpp:class:`~tvm::ffi::TensorView` when possible, that helps us to support more cases, including cases where only view but not strong reference are passed, like XLA buffer.
It is also more lightweight. However, since :cpp:class:`~tvm::ffi::TensorView` is a non-owning view, it is the user's responsibility to ensure the lifetime of underlying tensor data and attributes of the viewed tensor object.

Tensor Attributes
-----------------

For the sake of convenience, :cpp:class:`~tvm::ffi::TensorView` and :cpp:class:`~tvm::ffi::Tensor` align the following attributes retrieval mehtods to :cpp:class:`at::Tensor` interface, to obtain tensor basic attributes and storage pointer:
``dim``, ``dtype``, ``sizes``, ``size``, ``strides``, ``stride``, ``numel``, ``data_ptr``, ``device``, ``is_contiguous``

:c:struct:`DLDataType`
 The ``dtype`` of the tensor. It's represented by a struct with three fields: code, bits, and lanes, defined by DLPack protocol.

:c:struct:`DLDevice`
 The ``device`` where the tensor is stored. It is represented by a struct with two fields: device_type and device_id, defined by DLPack protocol.

:cpp:class:`tvm::ffi::ShapeView`
 The ``sizes`` and ``strides`` attributes retrieval are returned as :cpp:class:`~tvm::ffi::ShapeView`. It is an iterate-able data structure storing the shapes or strides data as ``int64_t`` array.

Tensor Allocation
-----------------

TVM FFI provides several methods to create or allocate tensors at C++ runtime. Generally, there are two types of tensor creation methods:

* Allocate a tensor with new storage from scratch, i.e. :cpp:func:`~tvm::ffi::Tensor::FromEnvAlloc` and :cpp:func:`~tvm::ffi::Tensor::FromNDAlloc`. By this types of methods, the shapes, strides, data types, devices and other attributes are required for the allocation.
* Create a tensor with existing storage following DLPack protocol, i.e. :cpp:func:`~tvm::ffi::Tensor::FromDLPack` and :cpp:func:`~tvm::ffi::Tensor::FromDLPackVersioned`. By this types of methods, the shapes, data types, devices and other attributes can be inferred from the DLPack attributes.

FromEnvAlloc
^^^^^^^^^^^^

To better adapt to the ML framework, it is **recommended** to reuse the framework tensor allocator anyway, instead of directly allocating the tensors via CUDA runtime API, like ``cudaMalloc``. Since reusing the framework tensor allocator:

* Benefit from the framework's native caching allocator or related allocation mechanism.
* Help framework tracking memory usage and planning globally.

For this case, TVM FFI provides :cpp:func:`tvm::ffi::Tensor::FromEnvAlloc`. It internally calls the framework tensor allocator. To determine which framework tensor allocator, TVM FFI infers it from the passed-in framework tensors. For example, when calling the kernel library at Python side, there is an input framework tensor if of type ``torch.Tensor``, TVM FFI will automatically bind the :cpp:function:`at::empty` as the current framework tensor allocator by ``TVMFFIEnvTensorAlloc``. And then the :cpp:func:`~tvm::ffi::Tensor::FromEnvAlloc` is calling the :cpp:class:`at::empty` actually:

.. code-block:: c++

 ffi::Tensor tensor = ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ...);

which is equivalent to:

.. code-block:: c++

 at::Tensor tensor = at::empty(...);

FromNDAlloc
^^^^^^^^^^^

:cpp:func:`tvm::ffi::Tensor::FromNDAlloc` can be used to create a tensor with custom memory allocator. It's used by the kernel provider if they don't want to rely on the framework tensor allocator. Instead, they provide their own custom allocator and deleter for tensor allocation and free. However, the tensors allocated by ``FromNDAlloc`` only retain the function pointer to its custom deleter for deconstruction. The custom deletes are all owned by the kernel library still. So it is important to make sure the loaded kernel library, :py:class:`tvm_ffi.Module`, outlives the tensors allocated by ``FromNDAlloc``. Otherwise, the function pointers to the custom deleter will be invalid. Here a typical approach is to retain the loaded :py:class:`tvm_ffi.Module` globally or for the period of time. But ``FromEnvAlloc`` is free of this issue, which is more **recommended** in practice.


FromDLPack
^^^^^^^^^^

:cpp:func:`tvm::ffi::Tensor::FromDLPack` enables creating :cpp:class:`~tvm::ffi::Tensor` from ``DLManagedTensor*``, working with ``ToDLPack`` for DLPack C Tensor Object ``DLTensor`` exchange protocol. Both are used for DLPack pre V1.0 API. It is used for wrapping the existing framework tensor to :cpp:class:`~tvm::ffi::Tensor`.

FromDLPackVersioned
^^^^^^^^^^^^^^^^^^^

:cpp:func:`tvm::ffi::Tensor::FromDLPackVersioned` enables creating :cpp:class:`~tvm::ffi::Tensor` from ``DLManagedTensorVersioned*``, working with ``ToDLPackVersioned`` for DLPack C Tensor Object ``DLTensor`` exchange protocol. Both are used for DLPack post V1.0 API. It is used for wrapping the existing framework tensor to :cpp:class:`~tvm::ffi::Tensor` too.

Python Calling FFI
==================

As we already have our kernel library wrapped with TVM FFI interface, our next and final step is exporting kernel library to Python side and enabling interaction with runtime environment or context.

Function Exporting
------------------

TVM FFI provides macro :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` for exporting the kernel functions to the output library files. So that at Python side, it is possible to load the library files and call the kernel functions directly. For example, we export our kernels as:

.. code-block:: c++

 void func(ffi::TensorView input, ffi::TensorView output);
 TVM_FFI_DLL_EXPORT_TYPED_FUNC(func, func);

And then we compile the sources into ``func.so``, or ``func.dylib`` for macOS, or ``func.dll`` for Windows. Finally, we can load and call our kernel functions at Python side as:

.. code-block:: python

 mod = tvm_ffi.load_module("func.so")
 x = ...
 y = ...
 mod.func(x, y)

``x`` and ``y`` here can be any ML framework tensors, such as ``torch.Tensor``, ``numpy.NDArray``, ``cupy.ndarray``, or other tensors as long as TVM FFI supports. TVM FFI detects the tensor types in arguments and converts them into :cpp:class:`~tvm::ffi::TensorView` automatically. So that we do not have to write the specific conversion codes per framework.

In constrast, if the kernel function returns :cpp:class:`~tvm::ffi::Tensor` instead of ``void`` in the example above. TVM FFI automatically converts the output :cpp:class:`~tvm::ffi::Tensor` to framework tensors also. The output framework is inferred from the input framework tensors. For example, if the input framework tensors are of ``torch.Tensor``, TVM FFI will convert the output tensor to ``torch.Tensor``. And if none of the input tensors are from ML framework, the output tensor will be the ``tvm_ffi.core.Tensor`` as fallback.

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
