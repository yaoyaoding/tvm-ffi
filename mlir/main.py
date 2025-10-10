import tvm_ffi
from typing import Optional, Sequence
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from kernel_spec import FunctionAdapter, Param, Var
from codegen import generate_tvm_ffi_func
from utils import dump_to_shared_library, run_passes

class EmptyAdapter(FunctionAdapter):
    """A simple concrete implementation of FunctionAdapter"""
    def get_param_spec(self) -> list[Param]:
        return []
    
    def call(self, mapping: dict[Var, ir.Value]) -> None:
        pass



def main():
    adapter = EmptyAdapter()
    ctx = ir.Context()
    
    with ctx, ir.Location.unknown():
        module = ir.Module.create()
        generate_tvm_ffi_func("__tvm_ffi_add_one", module, adapter)

    print(module)
    print("\nCompiling to shared library...")
    dump_to_shared_library(module, "libtvm_ffi_add_one.so", context=ctx, shared_libs=[tvm_ffi.libinfo.find_libtvm_ffi()])

    mod = tvm_ffi.load_module("./libtvm_ffi_add_one.so")
    mod.add_one()


if __name__ == "__main__":
    main()
