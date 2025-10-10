from typing import Optional, Sequence
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

class Param:
    """Base class for all parameters"""
    pass


class Var(Param):
    """variables: pointer, integer, floating-point, boolean, etc."""
    def __init__(self, tp: ir.Type):
        self.tp = tp


class Tensor(Param):
    """tensor parameter"""
    def __init__(self, ptr: Var, shape: Sequence[int | Var], strides: Optional[Sequence[Var]] = None):
        self.ptr: Var = ptr
        self.shape: list[int | Var] = list(shape)
        self.strides: Optional[list[Var]] = list(strides) if strides is not None else None


class FunctionAdapter:
    def get_param_spec(self) -> list[Param]:
        """the parameters that the tvm-ffi function expects"""
        raise NotImplementedError()

    def call(self, mapping: dict[Var, ir.Value]) -> None:
        """launch the kernel given the MLIR value of the Var parameters"""
        raise NotImplementedError()
