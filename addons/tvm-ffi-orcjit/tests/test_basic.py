# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Basic tests for tvm-ffi-orcjit functionality."""

import subprocess
import tempfile
from pathlib import Path

import pytest
from tvm_ffi_orcjit import create_session


def compile_simple_function() -> Path:
    """Compile a simple C function with TVM-FFI exports for testing.

    Returns
    -------
    Path
        Path to the compiled object file.

    """
    c_code = """
#include <tvm/ffi/c_ffi_api.h>

TVM_FFI_EXPORT_FUNC("test_add")
int test_add(TVMFFIFunctionHandle handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
    if (num_args != 2) return -1;
    int a = args[0].v_int64;
    int b = args[1].v_int64;
    rv->v_int64 = a + b;
    rv->type_code = kTVMFFIArgTypeInt;
    return 0;
}

TVM_FFI_EXPORT_FUNC("test_multiply")
int test_multiply(TVMFFIFunctionHandle handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
    if (num_args != 2) return -1;
    int a = args[0].v_int64;
    int b = args[1].v_int64;
    rv->v_int64 = a * b;
    rv->type_code = kTVMFFIArgTypeInt;
    return 0;
}
"""
    # Create temporary directory
    tmpdir = Path(tempfile.mkdtemp())
    src_file = tmpdir / "test_func.c"
    obj_file = tmpdir / "test_func.o"

    # Write C code
    src_file.write_text(c_code)

    # Compile with clang
    subprocess.run(
        [
            "clang",
            "-c",
            "-fPIC",
            "-O2",
            str(src_file),
            "-o",
            str(obj_file),
        ],
        check=True,
    )

    return obj_file


def test_create_session() -> None:
    """Test creating an execution session."""
    session = create_session()
    assert session is not None


def test_create_library() -> None:
    """Test creating a dynamic library."""
    session = create_session()
    lib = session.create_library()
    assert lib is not None


def test_load_and_execute_function() -> None:
    """Test loading an object file and executing a function."""
    # Compile test function
    obj_file = compile_simple_function()

    try:
        # Create session and library
        session = create_session()
        lib = session.create_library()

        # Load object file
        lib.add(str(obj_file))

        # Get and call test_add function
        add_func = lib.get_function("test_add")
        result = add_func(10, 20)
        assert result == 30

        # Get and call test_multiply function
        mul_func = lib.get_function("test_multiply")
        result = mul_func(7, 6)
        assert result == 42

    finally:
        # Clean up
        obj_file.unlink()
        obj_file.parent.rmdir()


def test_multiple_libraries() -> None:
    """Test creating and using multiple libraries."""
    session = create_session()

    lib1 = session.create_library("lib1")
    lib2 = session.create_library("lib2")

    assert lib1 is not None
    assert lib2 is not None


def test_function_not_found() -> None:
    """Test that getting a non-existent function raises an error."""
    obj_file = compile_simple_function()

    try:
        session = create_session()
        lib = session.create_library()
        lib.add(str(obj_file))

        with pytest.raises(AttributeError, match="Module has no function"):
            lib.get_function("nonexistent_function")

    finally:
        obj_file.unlink()
        obj_file.parent.rmdir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
