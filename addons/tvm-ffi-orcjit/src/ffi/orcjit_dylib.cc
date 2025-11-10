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
 * \file orcjit_dylib.cc
 * \brief LLVM ORC JIT DynamicLibrary implementation
 */

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/orcjit/orcjit_dylib.h>
#include <tvm/ffi/orcjit/orcjit_session.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {
namespace orcjit {

ORCJITDynamicLibrary::ORCJITDynamicLibrary(ObjectPtr<ORCJITExecutionSession> session,
                                           llvm::orc::JITDylib* dylib, llvm::orc::LLJIT* jit,
                                           String name)
    : session_(std::move(session)), dylib_(dylib), jit_(jit), name_(std::move(name)) {
  TVM_FFI_CHECK(dylib_ != nullptr, ValueError) << "JITDylib cannot be null";
  TVM_FFI_CHECK(jit_ != nullptr, ValueError) << "LLJIT cannot be null";
}

void ORCJITDynamicLibrary::AddObjectFile(const String& path) {
  // Read object file
  auto buffer_or_err = llvm::MemoryBuffer::getFile(path.c_str());
  if (!buffer_or_err) {
    TVM_FFI_THROW(IOError) << "Failed to read object file: " << path;
  }

  // Add object file to this JITDylib
  auto err = jit_->addObjectFile(*dylib_, std::move(*buffer_or_err));
  if (err) {
    std::string err_msg;
    llvm::handleAllErrors(std::move(err),
                          [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
    TVM_FFI_THROW(ValueError) << "Failed to add object file '" << path << "': " << err_msg;
  }
}

void ORCJITDynamicLibrary::LinkAgainst(const ORCJITDynamicLibrary& other) {
  // Set up link order: this dylib should search in other dylib
  llvm::orc::JITDylibSearchOrder search_order;
  search_order.push_back({other.dylib_, llvm::orc::JITDylibLookupFlags::MatchAllSymbols});

  dylib_->setLinkOrder(search_order, false);
}

void* ORCJITDynamicLibrary::GetSymbol(const String& name) {
  // Look up symbol
  auto symbol_or_err =
      jit_->getExecutionSession().lookup({dylib_}, jit_->mangleAndIntern(name.c_str()));
  if (!symbol_or_err) {
    auto err = symbol_or_err.takeError();
    std::string err_msg;
    llvm::handleAllErrors(std::move(err),
                          [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
    TVM_FFI_THROW(ValueError) << "Failed to find symbol '" << name << "': " << err_msg;
  }

  // Convert ExecutorAddr to pointer
  return symbol_or_err->getAddress().toPtr<void*>();
}

llvm::orc::JITDylib& ORCJITDynamicLibrary::GetJITDylib() {
  TVM_FFI_CHECK(dylib_ != nullptr, InternalError) << "JITDylib is null";
  return *dylib_;
}

//-------------------------------------
// Module wrapper for DynamicLibrary
//-------------------------------------

class DynamicLibraryModuleObj : public ModuleObj {
 public:
  explicit DynamicLibraryModuleObj(ObjectPtr<ORCJITDynamicLibrary> dylib)
      : dylib_(std::move(dylib)) {}

  const char* kind() const final { return "orcjit_dynamic_library"; }

  Optional<Function> GetFunction(const String& name) override {
    // TVM-FFI exports have __tvm_ffi_ prefix
    std::string symbol_name = "__tvm_ffi_" + std::string(name);

    // Try to get the symbol - return NullOpt if not found
    void* symbol = nullptr;
    try {
      symbol = dylib_->GetSymbol(symbol_name);
    } catch (const Error& e) {
      // Symbol not found
      return Optional<Function>();
    }

    // Wrap C function pointer as tvm-ffi Function
    using TVMFFISafeCallType =
        int (*)(void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv);
    auto c_func = reinterpret_cast<TVMFFISafeCallType>(symbol);

    return Function::FromPacked([c_func, name](PackedArgs args, Any* rv) {
      std::vector<AnyView> arg_views;
      arg_views.reserve(args.size());
      for (int i = 0; i < args.size(); ++i) {
        arg_views.push_back(args[i]);
      }

      int ret_code = c_func(nullptr, reinterpret_cast<const TVMFFIAny*>(arg_views.data()),
                            static_cast<int32_t>(args.size()), reinterpret_cast<TVMFFIAny*>(rv));

      if (ret_code != 0) {
        TVM_FFI_THROW(RuntimeError) << "Function '" << name << "' returned error code " << ret_code;
      }
    });
  }

 private:
  ObjectPtr<ORCJITDynamicLibrary> dylib_;
};

//-------------------------------------
// Registration
//-------------------------------------

static void RegisterOrcJITFunctions() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef()
      .def("orcjit.CreateExecutionSession",
           []() -> ObjectRef { return ObjectRef(ORCJITExecutionSession::Create()); })
      .def("orcjit.SessionCreateDynamicLibrary",
           [](ORCJITExecutionSession* session, String name) -> ObjectRef {
             return ObjectRef(session->CreateDynamicLibrary(name));
           })
      .def("orcjit.DynamicLibraryAdd",
           [](ORCJITDynamicLibrary* dylib, String path) { dylib->AddObjectFile(path); })
      .def("orcjit.DynamicLibraryLinkAgainst",
           [](ORCJITDynamicLibrary* dylib, ORCJITDynamicLibrary* other) {
             dylib->LinkAgainst(*other);
           })
      .def("orcjit.DynamicLibraryGetName",
           [](ORCJITDynamicLibrary* dylib) -> String { return dylib->GetName(); })
      .def("orcjit.DynamicLibraryToModule", [](ORCJITDynamicLibrary* dylib) -> Module {
        return Module(
            make_object<DynamicLibraryModuleObj>(GetObjectPtr<ORCJITDynamicLibrary>(dylib)));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // This block may not execute when loaded via dlopen on some platforms.
  // Call TVMFFIOrcJITInitialize() explicitly if functions are not registered.
  RegisterOrcJITFunctions();
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

// C API for explicit initialization
extern "C" {

TVM_FFI_DLL_EXPORT void TVMFFIOrcJITInitialize() { tvm::ffi::orcjit::RegisterOrcJITFunctions(); }
}
