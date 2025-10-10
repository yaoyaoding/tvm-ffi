from typing import Optional, Sequence
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from kernel_spec import FunctionAdapter, Param, Var

class TVMFFIType:
    @staticmethod
    def get() -> ir.Type:
        raise NotImplementedError()

class TVMFFIAny(TVMFFIType):
    @staticmethod
    def get() -> ir.Type:
        # TVMFFIAny struct layout:
        # - i32: type_index
        # - i32: union of zero_padding/small_str_len
        # - i64: union of 8-byte values (int64, float64, pointers, etc.)
        return ir.Type.parse("!llvm.struct<(i32, i32, i64)>")

class FunctionGenerator:
    def __init__(self, adapter: FunctionAdapter):
        self.adapter = adapter
    
    def raise_error_if(cond: ir.Value, err_message: str) -> None:
        pass

    def integer_constant(self, value: int) -> ir.Value:
        return llvm.ConstantOp(ir.IntegerType.get_signless(32), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)).res
    
    def extract_var_value(self, param: Param, tvm_ffi_any: ir.Value) -> dict[Var, ir.Value]:
        pass

    def generate(self, tvm_ffi_func_name: str) -> None:
        """
        Add a LLVM function to the current MLIR module with the given `tvm_ffi_func_name`.
        The function satisfies the tvm-ffi ABI specification.

        It takes the following steps to generate the function:
        1. create a LLVM function type matching the tvm-ffi ABI specification
        2. get the parameter specification from the `adapter.get_params()`
        3. extract the MLIR values of the LLVM function parameters
          3.1 generate code to check the type equivalence
          3.2 generate code to extract the MLIR values of all Var in parameter specification.
        4. call the `adapter.call()` to use the extracted MLIR values to launch the kernel
        """
        # 1. create a LLVM function type matching the tvm-ffi ABI specification
        # ```c++
        #   int func(void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result);
        # ```
        fn_type = ir.Type.parse("!llvm.func<i32 (ptr, ptr, i32, ptr)>")
        func_op = llvm.LLVMFuncOp(sym_name=tvm_ffi_func_name, function_type=ir.TypeAttr.get(fn_type))
        func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get() # Add the emit_c_interface attribute
        
        entry_block = ir.Block.create_at_start(func_op.body)

        ptr_type = ir.Type.parse("!llvm.ptr")
        i32_type = ir.IntegerType.get_signless(32)
        handle = entry_block.add_argument(ptr_type, ir.Location.unknown())
        args = entry_block.add_argument(ptr_type, ir.Location.unknown())
        num_args = entry_block.add_argument(i32_type, ir.Location.unknown())
        result = entry_block.add_argument(ptr_type, ir.Location.unknown())
        
        with ir.InsertionPoint(entry_block):
            # 2. get the parameter specification from the `adapter.get_params()`
            param_spec: list[Param] = self.adapter.get_param_spec()

            # 3. extract the MLIR values of the LLVM function parameters
            pass

            # Return 0 (success)
            zero = llvm.ConstantOp(ir.IntegerType.get_signless(32), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))
            llvm.return_(arg=zero)



def generate_tvm_ffi_func(symbol_name: str, adapter: FunctionAdapter) -> None:
    generator = FunctionGenerator(adapter)
    generator.generate(symbol_name)
