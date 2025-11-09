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
"""ORC JIT Execution Session."""

from __future__ import annotations

from typing import Any

from tvm_ffi import get_global_func

from .dylib import DynamicLibrary


class ExecutionSession:
    """ORC JIT Execution Session.

    Manages the LLVM ORC JIT execution environment and creates dynamic libraries (JITDylibs).
    This is the top-level context for JIT compilation and symbol management.

    Examples
    --------
    >>> session = create_session()
    >>> lib = session.create_library(name="main")
    >>> lib.add("add.o")
    >>> add_func = lib.get_function("add")

    """

    def __init__(self, handle: Any) -> None:
        """Initialize ExecutionSession from a handle.

        Parameters
        ----------
        handle : object
            The underlying C++ ORCJITExecutionSession object.

        """
        self._handle = handle
        self._create_dylib_func = get_global_func("orcjit.SessionCreateDynamicLibrary")

    def create_library(self, name: str = "") -> DynamicLibrary:
        """Create a new dynamic library associated with this execution session.

        Args:
            name: Optional name for the library. If empty, a unique name will be generated.

        Returns:
            A new DynamicLibrary instance.

        """
        handle = self._create_dylib_func(self._handle, name)
        return DynamicLibrary(handle, self)


def create_session() -> ExecutionSession:
    """Create a new ORC JIT execution session.

    This is the main entry point for using the ORC JIT system. The session
    manages the LLVM ORC JIT infrastructure and allows creating dynamic libraries.

    Returns
    -------
    ExecutionSession
        A new execution session instance.

    Examples
    --------
    >>> session = create_session()
    >>> lib = session.create_library()

    """
    create_func = get_global_func("orcjit.CreateExecutionSession")
    handle = create_func()
    return ExecutionSession(handle)
