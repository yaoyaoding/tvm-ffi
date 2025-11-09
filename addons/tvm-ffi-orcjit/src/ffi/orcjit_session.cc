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
 * \file orcjit_session.cc
 * \brief LLVM ORC JIT ExecutionSession implementation
 */

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/orcjit/orcjit_dylib.h>
#include <tvm/ffi/orcjit/orcjit_session.h>
#include <tvm/ffi/reflection/registry.h>

#include <sstream>

namespace tvm {
namespace ffi {
namespace orcjit {

// Initialize LLVM native target (only once)
struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }
};

static LLVMInitializer llvm_initializer;

// Provide __dso_handle for C++ runtime
static char dso_handle_storage;

ORCJITExecutionSession::ORCJITExecutionSession() : jit_(nullptr), dylib_counter_(0) {}

void ORCJITExecutionSession::Initialize() {
  // Create LLJIT instance
  auto jit_or_err = llvm::orc::LLJITBuilder().create();
  if (!jit_or_err) {
    auto err = jit_or_err.takeError();
    std::string err_msg;
    llvm::handleAllErrors(std::move(err),
                          [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
    TVM_FFI_THROW(InternalError) << "Failed to create LLJIT: " << err_msg;
  }
  jit_ = std::move(*jit_or_err);
}

ObjectPtr<ORCJITExecutionSession> ORCJITExecutionSession::Create() {
  auto session = make_object<ORCJITExecutionSession>();
  session->Initialize();
  return session;
}

ObjectPtr<ORCJITDynamicLibrary> ORCJITExecutionSession::CreateDynamicLibrary(const String& name) {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";

  // Generate name if not provided
  String lib_name = name;
  if (lib_name.empty()) {
    std::ostringstream oss;
    oss << "dylib_" << dylib_counter_++;
    lib_name = oss.str();
  }

  // Check if library with this name already exists
  TVM_FFI_CHECK(dylibs_.find(lib_name) == dylibs_.end(), ValueError)
      << "DynamicLibrary with name '" << lib_name << "' already exists";

  // Create a new JITDylib
  auto& jd = jit_->getExecutionSession().createBareJITDylib(lib_name.c_str());

  // Add process symbol resolver to make C/C++ stdlib available
  auto dlsg = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      jit_->getDataLayout().getGlobalPrefix());
  if (!dlsg) {
    TVM_FFI_THROW(InternalError) << "Failed to create process symbol resolver";
  }
  jd.addGenerator(std::move(*dlsg));

  // Add __dso_handle as a weak symbol (use static storage)
  auto& es = jit_->getExecutionSession();
  auto dso_symbol = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(&dso_handle_storage), llvm::JITSymbolFlags::Exported);
  llvm::orc::SymbolMap symbols;
  symbols[es.intern("__dso_handle")] = dso_symbol;
  if (auto err = jd.define(llvm::orc::absoluteSymbols(std::move(symbols)))) {
    TVM_FFI_THROW(InternalError) << "Failed to define __dso_handle";
  }

  // Create the wrapper object
  auto dylib = make_object<ORCJITDynamicLibrary>(GetObjectPtr<ORCJITExecutionSession>(this), &jd,
                                                 jit_.get(), lib_name);

  // Store for lifetime management
  dylibs_[lib_name] = dylib;

  return dylib;
}

llvm::orc::ExecutionSession& ORCJITExecutionSession::GetLLVMExecutionSession() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return jit_->getExecutionSession();
}

llvm::orc::LLJIT& ORCJITExecutionSession::GetLLJIT() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return *jit_;
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
