from typing import Optional, Sequence
from cutlass._mlir import ir, passmanager
from cutlass._mlir.dialects import llvm

def run_passes(module: ir.Module, context: ir.Context) -> ir.Module:
    with context:
        # Module is already in LLVM dialect, so we just run standard optimization passes
        pm = passmanager.PassManager.parse("builtin.module()")
        # Note: IR printing requires disabling multi-threading first
        # pm.enable_ir_printing(...)
        pm.run(module.operation)
        return module

def run_module(module: ir.Module, func_name: str, args, context: ir.Context):
    import ctypes
    from cutlass._mlir.execution_engine import ExecutionEngine
    
    with context:
        # Verify the module
        module.operation.verify()
        
        # Create execution engine
        engine = ExecutionEngine(module, opt_level=2, shared_libs=[])
        
        # Lookup the function directly
        func_ptr = engine.raw_lookup(func_name)
        if not func_ptr:
            raise RuntimeError(f"Function '{func_name}' not found in module")
        
        
        # Create a ctypes function prototype for void(void)
        # For functions with different signatures, adjust this accordingly
        func_proto = ctypes.CFUNCTYPE(None)
        func = func_proto(func_ptr)
        
        # Call the function
        result = func()
        return result


def dump_to_object_file(module: ir.Module, output_path: str, context: ir.Context, shared_libs: list[str] = []):
    """Dump the MLIR module to an object file using ExecutionEngine"""
    from cutlass._mlir.execution_engine import ExecutionEngine
    
    with context:
        # Verify the module
        module.operation.verify()
        
        # Create execution engine with object file dumping
        engine = ExecutionEngine(module, opt_level=2, shared_libs=shared_libs)
        
        # Dump the compiled object file
        engine.dump_to_object_file(output_path)
        print(f"Successfully created object file: {output_path}")


def compile_to_shared_library(object_file: str, output_path: str, compiler: str = "gcc"):
    """Compile an object file to a shared library"""
    import subprocess
    import shutil
    
    # Check if compiler is available
    if not shutil.which(compiler):
        raise RuntimeError(f"Compiler '{compiler}' not found. Please install it or specify a different compiler.")
    
    print(f"Compiling {object_file} to shared library: {output_path}")
    
    # Compile object file to shared library
    cmd = [compiler, "-shared", "-fPIC", object_file, "-o", output_path]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling to shared library: {e.stderr}")
        raise


def dump_to_shared_library(module: ir.Module, output_path: str, context: ir.Context, compiler: str = "gcc", shared_libs: list[str] = []):
    """Dump the MLIR module directly to a shared library"""
    import os
    import tempfile
    
    # Create a temporary object file
    with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp_obj:
        tmp_obj_path = tmp_obj.name
    
    try:
        # First dump to object file
        dump_to_object_file(module, tmp_obj_path, context, shared_libs)
        
        # Then compile to shared library
        compile_to_shared_library(tmp_obj_path, output_path, compiler)
    finally:
        # Clean up temporary object file
        if os.path.exists(tmp_obj_path):
            os.unlink(tmp_obj_path)
