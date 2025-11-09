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
"""ORC JIT Dynamic Library."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tvm_ffi import Function, get_global_func
from tvm_ffi._ffi_api import ModuleGetFunction

if TYPE_CHECKING:
    from .session import ExecutionSession


class DynamicLibrary:
    """ORC JIT Dynamic Library (JITDylib).

    Represents a collection of symbols that can be loaded from object files and linked
    against other dynamic libraries. Supports JIT compilation and symbol resolution.

    Examples
    --------
    >>> session = create_session()
    >>> lib = session.create_library()
    >>> lib.add("add.o")
    >>> lib.add("multiply.o")
    >>> add_func = lib.get_function("add")
    >>> result = add_func(1, 2)

    """

    def __init__(self, handle: Any, session: ExecutionSession) -> None:
        """Initialize DynamicLibrary from a handle.

        Parameters
        ----------
        handle : object
            The underlying C++ ORCJITDynamicLibrary object.
        session : ExecutionSession
            The parent execution session (kept alive for the library's lifetime).

        """
        self._handle = handle
        self._session = session  # Keep session alive
        self._add_func = get_global_func("orcjit.DynamicLibraryAdd")
        self._link_func = get_global_func("orcjit.DynamicLibraryLinkAgainst")
        self._to_module_func = get_global_func("orcjit.DynamicLibraryToModule")

    def add(self, object_file: str | Path) -> None:
        """Add an object file to this dynamic library.

        Parameters
        ----------
        object_file : str or Path
            Path to the object file to load.

        Examples
        --------
        >>> lib.add("add.o")
        >>> lib.add(Path("multiply.o"))

        """
        if isinstance(object_file, Path):
            object_file = str(object_file)
        self._add_func(self._handle, object_file)

    def link_against(self, *libraries: DynamicLibrary) -> None:
        """Link this library against other dynamic libraries.

        Sets the search order for symbol resolution. Symbols not found in this library
        will be searched in the linked libraries in the order specified.

        Parameters
        ----------
        *libraries : DynamicLibrary
            One or more dynamic libraries to link against.

        Examples
        --------
        >>> session = create_session()
        >>> lib_utils = session.create_library()
        >>> lib_utils.add("utils.o")
        >>> lib_main = session.create_library()
        >>> lib_main.add("main.o")
        >>> lib_main.link_against(lib_utils)  # main can call utils symbols

        """
        handles = [lib._handle for lib in libraries]
        self._link_func(self._handle, *handles)

    def get_function(self, name: str) -> Function:
        """Get a function from this dynamic library.

        Parameters
        ----------
        name : str
            The name of the function to retrieve.

        Returns
        -------
        callable
            The function object that can be called from Python.

        Examples
        --------
        >>> lib.add("add.o")
        >>> add_func = lib.get_function("add")
        >>> result = add_func(1, 2)

        """
        # Get the module handle and use ModuleGetFunction
        module_handle = self._to_module_func(self._handle)

        func = ModuleGetFunction(module_handle, name, False)
        if func is None:
            raise AttributeError(f"Module has no function '{name}'")
        return func
