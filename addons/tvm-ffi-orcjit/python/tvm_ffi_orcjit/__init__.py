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

"""TVM-FFI OrcJIT.

This module provides functionality to load object files (.o) compiled with TVM-FFI
exports using LLVM ORC JIT v2.

Example:
    >>> from tvm_ffi_orcjit import create_session
    >>> session = create_session()
    >>> lib = session.create_library()
    >>> lib.add("example.o")
    >>> func = lib.get_function("my_function")
    >>> result = func(arg1, arg2)

"""

import sys
from pathlib import Path

from tvm_ffi import load_module

# Load the orcjit extension library to register functions
_LIB_PATH = Path(__file__).parent.parent.parent / "libtvm_ffi_orcjit.so"
if _LIB_PATH.exists():
    load_module(str(_LIB_PATH))
else:
    # Fallback: search in site-packages (installed location)
    for site_pkg in sys.path:
        candidate = Path(site_pkg) / "libtvm_ffi_orcjit.so"
        if candidate.exists():
            load_module(str(candidate))
            break

from .dylib import DynamicLibrary
from .session import ExecutionSession, create_session

__all__ = ["DynamicLibrary", "ExecutionSession", "create_session"]
__version__ = "0.1.0"
