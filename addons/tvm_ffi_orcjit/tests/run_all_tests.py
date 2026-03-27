#!/usr/bin/env python3
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
"""Run pytest and quick-start examples.

Single entry point for CI.  All test objects (including multi-compiler variants)
are auto-built by conftest.py's session-scoped fixture when pytest runs.

Usage:
    python run_all_tests.py
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
QUICKSTART_DIR = TESTS_DIR.parent / "examples" / "quick-start"


def _run(cmd: list[str], **kwargs: object) -> None:
    print(f"+ {' '.join(str(a) for a in cmd)}", flush=True)
    subprocess.check_call(cmd, **kwargs)


def main() -> int:
    """Run pytest and quick-start examples."""
    # 1. Run pytest (conftest.py auto-builds all compiler variant objects)
    print(f"\n{'=' * 60}\nRunning pytest\n{'=' * 60}\n", flush=True)
    _run([sys.executable, "-m", "pytest", str(TESTS_DIR), "-v"])

    # 2. Run quick-start examples (objects compiled by run.py via tvm_ffi.cpp.build)
    print(f"\n{'=' * 60}\nRunning quick-start examples\n{'=' * 60}\n", flush=True)
    langs = ["c"]
    if platform.system() != "Windows":
        langs.insert(0, "cpp")
    for lang in langs:
        _run([sys.executable, str(QUICKSTART_DIR / "run.py"), "--lang", lang])

    print(f"\n{'=' * 60}\nAll tests passed\n{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
