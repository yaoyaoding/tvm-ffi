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
"""Auto-build test objects for all available compiler variants.

Uses tvm_ffi.cpp.build to compile C/C++ test sources to object files.
Detects platform and available compilers:
  Linux:   LLVM Clang (default) + GCC
  macOS:   LLVM Clang (default) + Apple Clang
  Windows: LLVM Clang (default) + MSVC + clang-cl

Objects are placed under the test directory (c/, cc/, c-gcc/, etc.) and
skipped if they already exist.  LLVM clang is found via ``$LLVM_PREFIX/bin``
or ``$PATH``; no hardcoded install path is assumed.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import pytest
import tvm_ffi.cpp

TESTS_DIR = Path(__file__).resolve().parent
SOURCES_C = TESTS_DIR / "sources" / "c"
SOURCES_CC = TESTS_DIR / "sources" / "cc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extra_cflags() -> list[str]:
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return ["-mno-outline-atomics"]
    return []


def _build_objects(
    src_dir: Path,
    out_dir: Path,
    *,
    ext_glob: str,
    extra_cflags: list[str],
) -> None:
    """Compile all sources in *src_dir* to object files in *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_dir.glob(ext_glob)):
        dest = out_dir / f"{src.stem}.o"
        if dest.exists():
            continue
        build_dir = out_dir / f".build_{src.stem}"
        obj_path = tvm_ffi.cpp.build(
            name=src.stem,
            sources=[str(src)],
            output=f"{src.stem}.o",
            extra_cflags=extra_cflags,
            build_directory=str(build_dir),
        )
        shutil.copy2(obj_path, dest)


def _build_variant(
    name: str,
    *,
    cc: str | None,
    cxx: str | None,
    extra_cflags: list[str],
    c_outdir: Path,
    cc_outdir: Path | None,
) -> None:
    """Build all test objects for one compiler variant."""
    saved_cc = os.environ.get("CC")
    saved_cxx = os.environ.get("CXX")
    try:
        if cc:
            os.environ["CC"] = cc
        if cxx:
            os.environ["CXX"] = cxx

        if cc:
            _build_objects(SOURCES_C, c_outdir, ext_glob="*.c", extra_cflags=extra_cflags)
        if cxx and cc_outdir:
            _build_objects(SOURCES_CC, cc_outdir, ext_glob="*.cc", extra_cflags=extra_cflags)
    finally:
        if saved_cc is None:
            os.environ.pop("CC", None)
        else:
            os.environ["CC"] = saved_cc
        if saved_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = saved_cxx


# ---------------------------------------------------------------------------
# Platform detection and variant dispatch
# ---------------------------------------------------------------------------


def _find_llvm_clang() -> tuple[str, str] | None:
    """Find LLVM clang/clang++ via LLVM_PREFIX or PATH.

    Returns (clang, clang++) paths, or None if not found.
    """
    # 1. Explicit LLVM_PREFIX
    llvm_prefix = os.environ.get("LLVM_PREFIX")
    if llvm_prefix:
        p = Path(llvm_prefix)
        # conda-style layout: Library/bin on Windows
        bin_dir = p / "Library" / "bin" if (p / "Library" / "bin").exists() else p / "bin"
        cc = bin_dir / ("clang.exe" if platform.system() == "Windows" else "clang")
        if cc.exists():
            cxx = bin_dir / ("clang++.exe" if platform.system() == "Windows" else "clang++")
            return str(cc), str(cxx)

    # 2. Find clang in PATH
    cc = shutil.which("clang")
    if cc:
        cxx = shutil.which("clang++")
        return cc, cxx or cc

    return None


def _build_all_variants() -> None:
    """Detect platform and build every applicable compiler variant."""
    system = platform.system()
    extra = _extra_cflags()

    if system in ("Linux", "Darwin"):
        llvm = _find_llvm_clang()
        if llvm:
            clang, clangxx = llvm
            _build_variant(
                "LLVM Clang",
                cc=clang,
                cxx=clangxx,
                extra_cflags=extra,
                c_outdir=TESTS_DIR / "c",
                cc_outdir=TESTS_DIR / "cc",
            )
        if system == "Linux" and shutil.which("gcc"):
            _build_variant(
                "GCC",
                cc="gcc",
                cxx="g++",
                extra_cflags=extra,
                c_outdir=TESTS_DIR / "c-gcc",
                cc_outdir=TESTS_DIR / "cc-gcc",
            )
        if system == "Darwin" and Path("/usr/bin/clang").exists():
            _build_variant(
                "Apple Clang",
                cc="/usr/bin/clang",
                cxx="/usr/bin/clang++",
                extra_cflags=extra,
                c_outdir=TESTS_DIR / "c-appleclang",
                cc_outdir=TESTS_DIR / "cc-appleclang",
            )

    elif system == "Windows":
        llvm = _find_llvm_clang()
        if llvm:
            clang, _ = llvm
            _build_variant(
                "LLVM Clang",
                cc=clang,
                cxx=None,
                extra_cflags=extra,
                c_outdir=TESTS_DIR / "c",
                cc_outdir=None,
            )
        if shutil.which("cl"):
            _build_variant(
                "MSVC",
                cc="cl",
                cxx=None,
                extra_cflags=["/GS-"],
                c_outdir=TESTS_DIR / "c-msvc",
                cc_outdir=None,
            )
        if shutil.which("clang-cl"):
            _build_variant(
                "clang-cl",
                cc="clang-cl",
                cxx=None,
                extra_cflags=["/GS-"],
                c_outdir=TESTS_DIR / "c-clang-cl",
                cc_outdir=None,
            )


# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _build_test_objects() -> None:
    """Build test objects for all available compiler variants."""
    _build_all_variants()
