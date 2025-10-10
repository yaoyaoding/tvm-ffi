from typing import Optional, Sequence, Callable
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from kernel_spec import FunctionAdapter, Param, Var
from type_builder import MLIRTypeBuilder


class MLIRBuilder(MLIRTypeBuilder):
    def __init__(self, module: ir.Module):
        self.module: ir.Module = module
    
    # create constants
    def integer_constant(self, tp: ir.Type, value: int) -> ir.Value:
        return llvm.ConstantOp(
            tp,
            ir.IntegerAttr.get(tp, value)
        ).res
    
    def i32(self, value: int) -> ir.Value:
        return self.integer_constant(self.i32_type, value)
    
    def ui32(self, value: int) -> ir.Value:
        return self.integer_constant(self.ui32_type, value)
    
    # comparison expressions
    def not_equal(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return llvm.icmp(
            llvm.ICmpPredicate.ne,
            lhs,
            rhs
        ).res

    # statements
    def if_else(self, cond: ir.Value, true_branch: Callable[[], None], false_branch: Callable[[], None]) -> None:
        pass

    def define_global_string(self, symbol: str, content: str) -> None:
        """ Define a global string symbol with the given content. """
        with ir.InsertionPoint(self.module.body):
            parsed_op = ir.Operation.parse(f'llvm.mlir.global private constant @{symbol}("{content}\\00")')
            self.module.body.append(parsed_op)


class TVMFFIBuilder(MLIRBuilder):
    """ A builder for TVM-FFI related types and operations. """
    def tvm_ffi_any(self) -> ir.Type:
        """ struct<i32, i32, ui64> """
        return self.struct_type(fields=[self.i32_type, self.i32_type, self.ui32_type])

    def tvm_ffi_func(self) -> ir.Type:
        """ !llvm.func<i32 (ptr, ptr, i32, ptr)> """
        return self.func_type(ret=self.i32_type, params=[self.ptr_type, self.ptr_type, self.i32_type, self.ptr_type])
    
    def check_cond(self, cond: ir.Value, exception_name: str, err_message: str):
        """ Check whether the condition is true, and if not, set the error message and return -1. """
        pass


class FunctionGenerator(TVMFFIBuilder):
    def __init__(self, module: ir.Module, adapter: FunctionAdapter):
        super().__init__(module)
        self.adapter = adapter

    def extract_var_values(self, params: Sequence[Param], num_args: ir.Value, args: ir.Value) -> dict[Var, ir.Value]:
        self.check_cond(
            self.not_equal(num_args, self.i32(len(params))),
            exception_name="ValueError",
            err_message=f"Expects {len(params)} parameters"
        )

    def generate(self, tvm_ffi_func_name: str) -> None:
        """
        Add a LLVM function to the current MLIR module with the given `tvm_ffi_func_name`.
        The function satisfies the tvm-ffi ABI specification.

        It takes the following steps to generate the function:
        1. create a LLVM function type matching the tvm-ffi ABI specification
        2. get the parameter specification from the `adapter.get_param_spec()`
        3. extract the MLIR values of the LLVM function parameters
          3.1 generate code to check the type equivalence
          3.2 generate code to extract the MLIR values of all Var in parameter specification.
        4. call the `adapter.call()` to use the extracted MLIR values to launch the kernel
        """
        # 1. create a LLVM function type matching the tvm-ffi ABI specification
        # ```c++
        #   int func(void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result);
        # ```
        with ir.InsertionPoint(self.module.body):
            func_op = llvm.func(tvm_ffi_func_name, function_type=self.as_attr(self.tvm_ffi_func()))
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get() # Add the emit_c_interface attribute
            
            entry_block = ir.Block.create_at_start(func_op.body)

            handle = entry_block.add_argument(self.ptr_type, ir.Location.unknown())
            args = entry_block.add_argument(self.ptr_type, ir.Location.unknown())
            num_args = entry_block.add_argument(self.i32_type, ir.Location.unknown())
            result = entry_block.add_argument(self.ptr_type, ir.Location.unknown())
            
            with ir.InsertionPoint(entry_block):
                # 2. get the parameter specification from the `adapter.get_param_spec()`
                param_spec: list[Param] = self.adapter.get_param_spec()

                # 3. extract the MLIR values of the LLVM function parameters
                mapping: dict[Var, ir.Value] = self.extract_var_values(param_spec, num_args, args)

                # Return 0 (success)
                zero = llvm.ConstantOp(ir.IntegerType.get_signless(32), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))
                llvm.return_(arg=zero)



def generate_tvm_ffi_func(symbol_name: str, module: ir.Module, adapter: FunctionAdapter) -> None:
    generator = FunctionGenerator(module, adapter)
    generator.generate(symbol_name)
