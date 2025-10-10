"""
Standalone script to generate the __tvm_ffi_add_one_c function using MLIR LLVM dialect.

This script generates MLIR IR equivalent to:
    int __tvm_ffi_add_one_c(
        void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
      ) {
        if(num_args != 2) {
            TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input");
            return -1;
        }
        return 0;
    }
"""

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from type_builder import MLIRTypeBuilder


def create_string_global(module_body, builder: MLIRTypeBuilder, string_value: str, global_name: str):
    """Create a global string constant by parsing MLIR text."""
    # Create a null-terminated string
    string_bytes = string_value.encode('utf-8') + b'\0'
    string_length = len(string_bytes)
    
    # Create the MLIR text for the global
    # Format the string bytes as hex values for the dense attribute
    hex_values = ', '.join([f'0x{b:02X}' for b in string_bytes])
    
    mlir_text = f'''
    llvm.mlir.global private constant @{global_name}("{string_value}\\00")
    '''
    
    # Parse and add to module
    with ir.InsertionPoint(module_body):
        parsed_op = ir.Operation.parse(mlir_text.strip())
        module_body.append(parsed_op)


def get_string_address(builder: MLIRTypeBuilder, global_name: str) -> ir.Value:
    """Get the address of a global string."""
    # Get address of the global
    addr = llvm.AddressOfOp(builder.pointer_type(), global_name)
    return addr.result


def generate_tvm_ffi_add_one_c():
    """Generate the __tvm_ffi_add_one_c function using MLIR LLVM dialect."""
    
    builder = MLIRTypeBuilder()
    ctx = ir.Context()
    
    with ctx, ir.Location.unknown():
        module = ir.Module.create()
        
        with ir.InsertionPoint(module.body):
            # Create global string constants first
            create_string_global(module.body, builder, "ValueError", "str_valueerror")
            create_string_global(module.body, builder, "Expects a Tensor input", "str_error_msg")
            
            # First, declare the external function TVMFFIErrorSetRaisedFromCStr
            # void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message)
            error_func_type = builder.func_type(
                params=[builder.ptr_type, builder.ptr_type],
                ret=builder.void_type()
            )
            
            error_func = llvm.func(
                "TVMFFIErrorSetRaisedFromCStr",
                function_type=builder.as_attr(error_func_type)
            )
            # Mark as external (no body - don't create blocks)
            
            # Now create the main function: __tvm_ffi_add_one_c
            func_type = builder.func_type(
                params=[builder.ptr_type, builder.ptr_type, builder.i32_type, builder.ptr_type],
                ret=builder.i32_type
            )
            
            func_op = llvm.func(
                "__tvm_ffi_add_one_c",
                function_type=builder.as_attr(func_type)
            )
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            
            # Create the function body
            entry_block = ir.Block.create_at_start(func_op.body)
            
            # Add function arguments
            handle = entry_block.add_argument(builder.ptr_type, ir.Location.unknown())
            args = entry_block.add_argument(builder.ptr_type, ir.Location.unknown())
            num_args = entry_block.add_argument(builder.i32_type, ir.Location.unknown())
            result = entry_block.add_argument(builder.ptr_type, ir.Location.unknown())
            
            with ir.InsertionPoint(entry_block):
                # Create basic blocks for control flow
                error_block = entry_block.create_after()
                check_type_block = error_block.create_after()
                error_return_block = check_type_block.create_after()
                success_block = error_return_block.create_after()
                
                # Create constants
                two = llvm.ConstantOp(
                    ir.IntegerType.get_signless(32),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2)
                )
                
                # Check: if (num_args != 2)
                cmp = llvm.ICmpOp(
                    llvm.ICmpPredicate.ne,
                    num_args,
                    two.result
                )
                
                # Conditional branch
                llvm.CondBrOp(
                    cmp.result,
                    trueDest=error_block,
                    falseDest=check_type_block,
                    trueDestOperands=[],
                    falseDestOperands=[]
                )
            
            # Error block: call TVMFFIErrorSetRaisedFromCStr and return -1
            with ir.InsertionPoint(error_block):
                # Get addresses of string constants
                kind_str = get_string_address(builder, "str_valueerror")
                msg_str = get_string_address(builder, "str_error_msg")
                
                # Create call operation using generic Operation API
                # operandSegmentSizes should be [num_operands, num_branch_weights]
                call_op = ir.Operation.create(
                    "llvm.call",
                    operands=[kind_str, msg_str],
                    results=[],
                    attributes={
                        "callee": ir.FlatSymbolRefAttr.get("TVMFFIErrorSetRaisedFromCStr"),
                        "operandSegmentSizes": ir.DenseI32ArrayAttr.get([2, 0]),  # 2 operands, 0 branch weights
                        "op_bundle_sizes": ir.DenseI32ArrayAttr.get([]),  # No operation bundles
                    }
                )
                
                # Return -1
                minus_one = llvm.ConstantOp(
                    ir.IntegerType.get_signless(32),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(32), -1)
                )
                llvm.return_(arg=minus_one)
            
            # Check type block: if (args[0].type_index != 1)
            with ir.InsertionPoint(check_type_block):
                # Access args[0].type_index
                # args is a pointer to TVMFFIAny struct
                # Define the TVMFFIAny struct type: struct { i32, i32, union { i32, float, ptr } }
                # For simplicity, we'll treat the union as i64 (8 bytes)
                tvmffi_any_type = builder.struct_type(fields=[
                    builder.i32_type,  # type_index
                    builder.i32_type,  # padding
                    ir.IntegerType.get_signless(64)  # union (8 bytes)
                ])
                
                # GEP to get pointer to args[0].type_index
                # Use constant indices: [0, 0] means args[0].field[0]
                # No dynamic indices needed
                type_index_ptr = llvm.GEPOp(
                    builder.ptr_type,
                    args,
                    [],  # No dynamic indices
                    rawConstantIndices=ir.DenseI32ArrayAttr.get([0, 0]),
                    elem_type=tvmffi_any_type
                )
                
                # Load the type_index value
                type_index_val = llvm.LoadOp(
                    ir.IntegerType.get_signless(32),
                    type_index_ptr.result
                )
                
                # Compare with 1
                one = llvm.ConstantOp(
                    ir.IntegerType.get_signless(32),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1)
                )
                
                type_cmp = llvm.ICmpOp(
                    llvm.ICmpPredicate.ne,
                    type_index_val.result,
                    one.result
                )
                
                # Branch: if type_index != 1, go to error_return_block, else success_block
                llvm.CondBrOp(
                    type_cmp.result,
                    trueDest=error_return_block,
                    falseDest=success_block,
                    trueDestOperands=[],
                    falseDestOperands=[]
                )
            
            # Error return block: just return -1 (no error message)
            with ir.InsertionPoint(error_return_block):
                minus_one_2 = llvm.ConstantOp(
                    ir.IntegerType.get_signless(32),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(32), -1)
                )
                llvm.return_(arg=minus_one_2)
            
            # Success block: return 0
            with ir.InsertionPoint(success_block):
                zero = llvm.ConstantOp(
                    ir.IntegerType.get_signless(32),
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
                )
                llvm.return_(arg=zero)
        
        return module, ctx


def main():
    """Generate, print, and optionally compile the MLIR module."""
    module, ctx = generate_tvm_ffi_add_one_c()
    
    print("=" * 80)
    print("Generated MLIR IR:")
    print("=" * 80)
    print(module)
    print()
    
    # Optionally compile to shared library
    try:
        import tvm_ffi
        from utils import mlir_to_shared_library_via_llvm
        import os
        
        # Use the build directory instead of installed package
        tvm_ffi_lib = tvm_ffi.libinfo.find_libtvm_ffi()
        
        # Use the new LLVM IR-based compilation method
        # This bypasses ExecutionEngine's symbol validation issues
        mlir_to_shared_library_via_llvm(
            module=module,
            output_path="libtvm_ffi_add_one_c.so",
            context=ctx,
            compiler="gcc",
            link_libs=[tvm_ffi_lib],
            opt_level=2,
            keep_intermediates=False  # Set to True for debugging
        )
        mod = tvm_ffi.load_module("./libtvm_ffi_add_one_c.so")
        mod.add_one_c()
    except Exception as e:
        import traceback
        print(f"Note: Could not compile to shared library: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

