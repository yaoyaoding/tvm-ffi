/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file orcjit_session.h
 * \brief LLVM ORC JIT ExecutionSession wrapper
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_SESSION_H_
#define TVM_FFI_ORCJIT_ORCJIT_SESSION_H_

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace tvm {
namespace ffi {
namespace orcjit {

// Forward declaration
class ORCJITDynamicLibrary;

/*!
 * \brief ExecutionSession wrapper for LLVM ORC JIT v2
 *
 * This class manages the lifetime of an LLVM ExecutionSession and provides
 * functionality to create and manage multiple JITDylibs (DynamicLibraries).
 */
class ORCJITExecutionSession : public Object {
 public:
  /*!
   * \brief Create a new ExecutionSession
   * \return The created execution session instance
   */
  static ObjectPtr<ORCJITExecutionSession> Create();

  /*!
   * \brief Create a new DynamicLibrary (JITDylib) in this session
   * \param name Optional name for the library (for debugging)
   * \return The created dynamic library instance
   */
  ObjectPtr<ORCJITDynamicLibrary> CreateDynamicLibrary(const String& name);

  /*!
   * \brief Get the underlying LLVM ExecutionSession
   * \return Reference to the LLVM ExecutionSession
   */
  llvm::orc::ExecutionSession& GetLLVMExecutionSession();

  /*!
   * \brief Get the underlying LLJIT instance
   * \return Reference to the LLJIT instance
   */
  llvm::orc::LLJIT& GetLLJIT();

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("orcjit.ExecutionSession", ORCJITExecutionSession, Object);

  /*!
   * \brief Default constructor (for make_object)
   */
  ORCJITExecutionSession();

 private:
  /*!
   * \brief Initialize the LLJIT instance
   */
  void Initialize();

  /*! \brief The LLVM ORC JIT instance */
  std::unique_ptr<llvm::orc::LLJIT> jit_;

  /*! \brief Counter for auto-generating library names */
  int dylib_counter_ = 0;

  /*! \brief Map of created dynamic libraries for lifetime management */
  std::unordered_map<std::string, ObjectPtr<ORCJITDynamicLibrary>> dylibs_;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_SESSION_H_
