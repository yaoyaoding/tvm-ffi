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


def compile_llvm_ir_to_object(llvm_ir_path: str, output_obj_path: str, opt_level: int = 2) -> None:
    """
    Compile LLVM IR (.ll) file to object file (.o) using llc.
    
    Args:
        llvm_ir_path: Path to LLVM IR file (.ll)
        output_obj_path: Path to output object file (.o)
        opt_level: Optimization level (0-3)
    """
    import subprocess
    import shutil
    
    # Check if llc is available (try versioned names too)
    llc_path = None
    for name in ["llc", "llc-18", "llc-19", "llc-17", "llc-20"]:
        llc_path = shutil.which(name)
        if llc_path:
            break
    
    if not llc_path:
        raise RuntimeError(
            "llc (LLVM compiler) not found. Please install LLVM tools or ensure they are in PATH.\n"
            "On Ubuntu/Debian: sudo apt-get install llvm-18"
        )
    
    print(f"Compiling LLVM IR to object file: {llvm_ir_path} -> {output_obj_path}")
    
    # Compile LLVM IR to object file
    # -filetype=obj: output object file
    # -O{opt_level}: optimization level
    # -relocation-model=pic: position independent code for shared libraries
    cmd = [
        llc_path,
        f"-O{opt_level}",
        "-filetype=obj",
        "-relocation-model=pic",
        llvm_ir_path,
        "-o",
        output_obj_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LLVM IR to object file:")
        print(f"Command: {' '.join(cmd)}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        raise


def mlir_to_shared_library_via_llvm(
    module: ir.Module,
    output_path: str,
    context: ir.Context,
    compiler: str = "gcc",
    link_libs: list[str] = [],
    opt_level: int = 2,
    keep_intermediates: bool = False
) -> None:
    """
    Compile MLIR module to shared library via LLVM IR, bypassing ExecutionEngine.
    
    This method avoids the ExecutionEngine's symbol validation issues by:
    1. Converting MLIR (LLVM dialect) to LLVM IR text (.ll file) using mlir-translate
    2. Using llc to compile LLVM IR to object file (.o)
    3. Using gcc/clang to link object file into shared library
    
    This allows external symbols to remain unresolved during object file creation,
    and they will be resolved at link time or runtime.
    
    Args:
        module: MLIR module in LLVM dialect
        output_path: Path to output shared library (.so)
        context: MLIR context
        compiler: C compiler to use for linking (default: gcc)
        link_libs: List of libraries to link against
        opt_level: Optimization level (0-3)
        keep_intermediates: If True, keep intermediate .ll and .o files for debugging
        
    Example:
        >>> from utils import mlir_to_shared_library_via_llvm
        >>> mlir_to_shared_library_via_llvm(
        ...     module=module,
        ...     output_path="my_function.so",
        ...     context=ctx,
        ...     link_libs=["build/lib/libtvm_ffi.so"],
        ...     keep_intermediates=True  # Keep .ll and .o for debugging
        ... )
    """
    import os
    import subprocess
    import tempfile
    import shutil
    
    print("=" * 80)
    print("Compiling MLIR to shared library via LLVM IR")
    print("=" * 80)
    
    # Check if mlir-translate is available (try versioned names too)
    mlir_translate = None
    for name in ["mlir-translate", "mlir-translate-18", "mlir-translate-19", "mlir-translate-17", "mlir-translate-20"]:
        mlir_translate = shutil.which(name)
        if mlir_translate:
            print(f"Found {name} at {mlir_translate}")
            break
    
    if not mlir_translate:
        raise RuntimeError(
            "mlir-translate not found. Please install MLIR tools or ensure they are in PATH.\n"
            "On Ubuntu/Debian: sudo apt-get install mlir-18-tools\n"
            "Alternative: Install LLVM/MLIR from your package manager or build from source."
        )
    
    # Create temporary files
    if keep_intermediates:
        # Use current directory
        base_name = os.path.splitext(output_path)[0]
        ll_path = f"{base_name}.ll"
        obj_path = f"{base_name}.o"
    else:
        # Use temporary files
        ll_fd, ll_path = tempfile.mkstemp(suffix='.ll', text=True)
        os.close(ll_fd)
        obj_fd, obj_path = tempfile.mkstemp(suffix='.o')
        os.close(obj_fd)
    
    try:
        # Step 1: Save MLIR module to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as mlir_file:
            mlir_path = mlir_file.name
            with context:
                mlir_file.write(str(module))
        
        try:
            # Step 2: Translate MLIR to LLVM IR using mlir-translate
            print(f"Translating MLIR to LLVM IR: {mlir_path} -> {ll_path}")
            cmd = [mlir_translate, "--mlir-to-llvmir", mlir_path, "-o", ll_path]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stderr:
                print(f"mlir-translate warnings/info: {result.stderr}")
            
            # Verify LLVM IR was created
            if not os.path.exists(ll_path) or os.path.getsize(ll_path) == 0:
                raise RuntimeError("Failed to generate LLVM IR file")
            
            print(f"Successfully generated LLVM IR file ({os.path.getsize(ll_path)} bytes)")
            
            # Step 3: Compile LLVM IR to object file using llc
            compile_llvm_ir_to_object(ll_path, obj_path, opt_level)
            
            # Verify object file was created
            if not os.path.exists(obj_path) or os.path.getsize(obj_path) == 0:
                raise RuntimeError("Failed to generate object file")
            
            print(f"Successfully generated object file ({os.path.getsize(obj_path)} bytes)")
            
            # Step 4: Link object file to shared library
            # Update compile_to_shared_library call to support link_libs
            cmd_link = [compiler, "-shared", "-fPIC", obj_path, "-o", output_path]
            for lib in link_libs:
                cmd_link.append(lib)
            
            result = subprocess.run(
                cmd_link,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(result.stdout)
            
            print(f"Successfully created shared library: {output_path}")
            
            if keep_intermediates:
                print(f"Intermediate files kept:")
                print(f"  LLVM IR: {ll_path}")
                print(f"  Object:  {obj_path}")
        
        finally:
            # Clean up MLIR temp file
            if os.path.exists(mlir_path):
                os.unlink(mlir_path)
    
    finally:
        # Clean up intermediate files if not keeping them
        if not keep_intermediates:
            if os.path.exists(ll_path):
                os.unlink(ll_path)
            if os.path.exists(obj_path):
                os.unlink(obj_path)
