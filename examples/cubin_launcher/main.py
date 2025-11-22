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
"""Test script for CUBIN launcher libraries.

This script tests both embedded and dynamic CUBIN loading approaches
using PyTorch tensors with TVM-FFI.
"""

import os
import sys

import torch
from tvm_ffi import load_module


def test_embedded_library():
    """Test the lib_embedded.so library with embedded CUBIN."""
    print("=" * 60)
    print("Testing Embedded CUBIN Library")
    print("=" * 60)

    # Load the library
    lib_path = os.path.join(os.path.dirname(__file__), "build", "lib_embedded.so")
    mod = load_module(lib_path)
    print(f"Loaded library: {lib_path}")

    # Get the functions
    add_one = mod["add_one"]
    mul_two = mod["mul_two"]
    print("Loaded functions: add_one, mul_two")

    # Test add_one kernel
    print("\n[Test 1] add_one kernel")
    n = 1024
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    add_one(x, y)

    # Verify results
    expected = x + 1
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return False

    # Test mul_two kernel
    print("\n[Test 2] mul_two kernel")
    n = 512
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.5
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    mul_two(x, y)

    # Verify results
    expected = x * 2
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return False

    # Test chained execution
    print("\n[Test 3] Chained execution: (x + 1) * 2")
    n = 256
    x = torch.full((n,), 10.0, dtype=torch.float32, device="cuda")
    temp = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Initial value: {x[0].item()}")
    add_one(x, temp)  # temp = x + 1 = 11
    mul_two(temp, y)  # y = temp * 2 = 22

    expected = 22.0
    if torch.allclose(y, torch.tensor(expected, device="cuda")):
        print(f"  [PASS] Result: {y[0].item()}")
    else:
        print(f"  [FAIL] Expected {expected}, got {y[0].item()}")
        return False

    return True


def test_dynamic_library():
    """Test the lib_dynamic.so library with dynamic CUBIN loading."""
    print("\n" + "=" * 60)
    print("Testing Dynamic CUBIN Library")
    print("=" * 60)

    # Load the library
    lib_path = os.path.join(os.path.dirname(__file__), "build", "lib_dynamic.so")
    mod = load_module(lib_path)
    print(f"Loaded library: {lib_path}")

    # Load CUBIN file
    cubin_path = os.path.join(os.path.dirname(__file__), "build", "kernel.cubin")
    load_cubin = mod["load_cubin"]
    load_cubin(cubin_path)
    print(f"Loaded CUBIN from: {cubin_path}")

    # Get the kernel functions
    add_one = mod["add_one"]
    mul_two = mod["mul_two"]
    print("Loaded functions: add_one, mul_two")

    # Test add_one kernel
    print("\n[Test 1] add_one kernel")
    n = 2048
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.1
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    add_one(x, y)

    # Verify results
    expected = x + 1
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return False

    # Test mul_two kernel
    print("\n[Test 2] mul_two kernel")
    n = 1024
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.25
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    mul_two(x, y)

    # Verify results
    expected = x * 2
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return False

    # Test chained execution
    print("\n[Test 3] Chained execution: (x + 1) * 2")
    n = 512
    x = torch.full((n,), 5.0, dtype=torch.float32, device="cuda")
    temp = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Initial value: {x[0].item()}")
    add_one(x, temp)  # temp = x + 1 = 6
    mul_two(temp, y)  # y = temp * 2 = 12

    expected = 12.0
    if torch.allclose(y, torch.tensor(expected, device="cuda")):
        print(f"  [PASS] Result: {y[0].item()}")
    else:
        print(f"  [FAIL] Expected {expected}, got {y[0].item()}")
        return False

    return True


def main():
    """Main test function."""
    print("CUBIN Launcher Library Tests with PyTorch")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    try:
        # Test embedded library
        embedded_success = test_embedded_library()

        # Test dynamic library
        dynamic_success = test_dynamic_library()

        # Final summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Embedded Library: {'[PASS]' if embedded_success else '[FAIL]'}")
        print(f"Dynamic Library:  {'[PASS]' if dynamic_success else '[FAIL]'}")

        if embedded_success and dynamic_success:
            print("\n[PASS] All tests passed!")
            return 0
        else:
            print("\n[FAIL] Some tests failed!")
            return 1

    except Exception as e:
        print(f"\n[ERROR] {e!s}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
