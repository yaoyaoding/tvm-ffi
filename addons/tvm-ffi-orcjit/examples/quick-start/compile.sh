#!/bin/bash
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

# Compile script for quick-start example

set -e

# Colors for output
GREEN='\\033[0;32m'
RED='\\033[0;31m'
NC='\\033[0m' # No Color

echo -e "${GREEN}Compiling add.cc to object file...${NC}"

# Check if tvm-ffi-config is available
if ! command -v tvm-ffi-config &> /dev/null; then
    echo -e "${RED}Error: tvm-ffi-config not found${NC}"
    echo "Make sure apache-tvm-ffi is installed:"
    echo "  pip install -e ../../../"
    exit 1
fi

# Get compilation flags from tvm-ffi-config
echo -e "${GREEN}Getting compilation flags from tvm-ffi-config...${NC}"
CXXFLAGS=$(tvm-ffi-config --cxxflags)
LDFLAGS=$(tvm-ffi-config --ldflags)

# Override C++ standard to C++20 (needed for lambda in unevaluated context)
CXXFLAGS="${CXXFLAGS/-std=c++17/-std=c++20}"

echo "  CXXFLAGS: $CXXFLAGS"
echo "  LDFLAGS: $LDFLAGS"

# Compile to object file
echo -e "${GREEN}Compiling...${NC}"
# shellcheck disable=SC2086
g++ -c add.cc \
    -o add.o \
    $CXXFLAGS \
    -fPIC \
    -O2

echo -e "${GREEN}Successfully compiled add.o${NC}"
echo ""
echo "You can now run: python run.py"
