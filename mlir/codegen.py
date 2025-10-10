from typing import Optional, Sequence, Callable
from contextlib import contextmanager
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from kernel_spec import FunctionAdapter, Param, Var
from type_builder import MLIRTypeBuilder


class MLIRBuilder(MLIRTypeBuilder):
    """ A builder for MLIR related types and operations. 

    Convention:
    all statement-generation methods expect we are inside a insertion point of the current block.
    If the statement terminates the current block, it will assign the new block to the current block but not set the insersion point.
    """
    def __init__(self):
        self.module: Optional[ir.Module] = None
        self.current_block: Optional[ir.Block] = None
    
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
    
    # expressions
    def not_equal(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return llvm.icmp(
            llvm.ICmpPredicate.ne,
            lhs,
            rhs
        )
    
    def address_of(self, name: str, tp: ir.Type) -> ir.Value:
        return llvm.AddressOfOp(tp, name).result

    # statements

    def return_(self, ret: Optional[ir.Value] = None) -> None:
        llvm.return_(arg=ret)

    def cond_br(self, cond: ir.Value, true_block: ir.Block, false_block: ir.Block) -> None:
        llvm.cond_br(cond, true_dest_operands=[], false_dest_operands=[], true_dest=true_block, false_dest=false_block)

    def if_statement(self, cond: ir.Value, true_branch: Callable[[], None]) -> None:
        then_block = self.current_block.create_after()
        else_block = then_block.create_after()
        self.current_block = else_block

        self.cond_br(cond, then_block, else_block)

        with ir.InsertionPoint(then_block):
            true_branch()

    def define_global_string(self, symbol: str, content: str) -> None:
        """ Define a global string symbol with the given content. """
        with ir.InsertionPoint(self.module.body):
            parsed_op = ir.Operation.parse(f'llvm.mlir.global private constant @{symbol}("{content}\\00")')
            self.module.body.append(parsed_op)
        
    # function
    def function(self, name: str, params_type: Sequence[ir.Type], ret_type: ir.Type) -> list[ir.Value]:
        func_op = llvm.func(name, function_type=self.as_attr(self.func_type(ret=ret_type, params=params_type)))
        func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get() 

        params = []
        entry_block = ir.Block.create_at_start(func_op.body)
        self.current_block = entry_block
        for param_type in params_type:
            params.append(entry_block.add_argument(param_type, ir.Location.unknown()))
        
        return params

class TVMFFIBuilder(MLIRBuilder):
    TVMFFI_ERROR_SET_RAISED_FROM_C_STR_NAME = "TVMFFIErrorSetRaisedFromCStr"

    """ A builder for TVM-FFI related types and operations. """
    def tvm_ffi_any(self) -> ir.Type:
        """ struct<i32, i32, ui64> """
        return self.struct_type(fields=[self.i32_type, self.i32_type, self.ui32_type])

    def tvm_ffi_func(self) -> ir.Type:
        """ !llvm.func<i32 (ptr, ptr, i32, ptr)> """
        return self.func_type(ret=self.i32_type, params=[self.ptr_type, self.ptr_type, self.i32_type, self.ptr_type])
    
    def define_tvm_ffi_error_set_raised_from_c_str(self) -> None:
        # TODO: encounter this error:
        # ```
        # Could not compile TVMFFIErrorSetRaisedFromCStr:
        #   Symbols not found: [ _mlir_TVMFFIErrorSetRaisedFromCStr ]
        # ```
        # skip for now
        # error_func_type = self.func_type(params=[self.ptr_type, self.ptr_type], ret=self.void_type())
        # func_op = llvm.func(self.TVMFFI_ERROR_SET_RAISED_FROM_C_STR_NAME, function_type=self.as_attr(error_func_type))
        # func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get() 
        # func_op.attributes["llvm.linkage"] = ir.StringAttr.get("external")
        pass
    
    def raise_exception_and_return(self, exception_name: str, err_message: str) -> None:
        self.define_global_string(symbol='str_exception_name', content=exception_name)
        self.define_global_string(symbol='str_error_message', content=err_message)
        # llvm.call(
        #     result=None,
        #     callee=self.TVMFFI_ERROR_SET_RAISED_FROM_C_STR_NAME,
        #     callee_operands=[
        #         self.address_of('str_exception_name', self.ptr_type), 
        #         self.address_of('str_error_message', self.ptr_type)
        #     ],
        #     op_bundle_sizes=[],
        #     op_bundle_operands=[],
        # )
        self.return_(self.i32(-1))


class FunctionGenerator(TVMFFIBuilder):
    def __init__(self, adapter: FunctionAdapter):
        self.adapter = adapter

    def extract_var_values(self, params: Sequence[Param], num_args: ir.Value, args: ir.Value) -> dict[Var, ir.Value]:
        with ir.InsertionPoint(self.current_block):
            self.if_statement(
                cond=self.not_equal(num_args, self.i32(len(params))),
                true_branch=lambda: self.raise_exception_and_return(
                    exception_name="ValueError",
                    err_message=f"Expects {len(params)} parameters"
                )
            )
        return {}


    def generate(self, module: ir.Module, tvm_ffi_func_name: str) -> None:
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
        self.module = module
        with ir.InsertionPoint(module.body):
            self.define_tvm_ffi_error_set_raised_from_c_str()  

            handle, args, num_args, result = self.function(
                name=tvm_ffi_func_name, 
                params_type=[self.ptr_type, self.ptr_type, self.i32_type, self.ptr_type], ret_type=self.i32_type
            )
            # 2. get the parameter specification from the `adapter.get_param_spec()`
            param_spec: list[Param] = self.adapter.get_param_spec()

            # 3. extract the MLIR values of the LLVM function parameters
            mapping = self.extract_var_values(param_spec, num_args, args)

            # Return 0 (success)
            with ir.InsertionPoint(self.current_block):
                self.return_(self.i32(0))



def generate_tvm_ffi_func(symbol_name: str, module: ir.Module, adapter: FunctionAdapter) -> None:
    generator = FunctionGenerator(adapter)
    generator.generate(module, symbol_name)
