from typing import Sequence, Optional
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

class MLIRTypeBuilder:
    @property
    def i32_type(self) -> ir.Type:
        return self.signed_type(32)
    
    @property
    def ui32_type(self) -> ir.Type:
        return self.unsigned_type(32)
    
    @property
    def ptr_type(self) -> ir.Type:
        return self.pointer_type()

    def as_attr(self, tp: ir.Type) -> ir.TypeAttr:
        """ Convert the type to a type attribute. """
        return ir.TypeAttr.get(tp)

    def signed_type(self, bits: int) -> ir.Type:
        """ Get the `i<bits>` type. """
        return ir.IntegerType.get_signless(bits)
    
    def unsigned_type(self, bits: int) -> ir.Type:
        """ Get the `ui<bits>` type. """
        return ir.IntegerType.get_unsigned(bits)
    
    def void_type(self) -> ir.Type:
        """ Get the `!llvm.void` type. """
        return ir.Type.parse("!llvm.void")   # did not find a programmatic way to get the void type
    
    def pointer_type(self) -> ir.Type:
        """ Get the `!llvm.ptr` type. """
        return llvm.PointerType.get()
    
    def struct_type(self, *, name: Optional[str] = None, fields: Sequence[ir.Type] = (), packed: bool = False) -> ir.Type:
        """
        Get or create a struct type.

        Parameters
        ----------
        name : Optional[str]
            The name of the struct type. If not provided, a `literal` struct type is created, which is identified by its fields only.
        fields : Sequence[ir.Type]
            The fields of the struct type.
        packed : bool
            Whether to create a packed struct type. If `True`, there is no padding between fields. Otherwise, there might be padding
            between fields to ensure alignment.

        See Also
        --------
        https://mlir.llvm.org/docs/Dialects/LLVM/#structure-types
        """
        if name is None:
            return llvm.StructType.get_literal(fields, packed=packed)
        else:
            return llvm.StructType.new_identified(name, fields, packed=packed)
    
    def identified_struct_type(self, name: str) -> ir.Type:
        """ Get a previously created identified struct type by its name. """
        return llvm.StructType.get_identified(name)
    
    def func_type(self, *, params: Sequence[ir.Type] = (), ret: ir.Type) -> ir.Type:
        """ Get a function type. 

        Parameters
        ----------
        params : Sequence[ir.Type]
            The parameters of the function type.
        result : ir.Type
            The result of the function type.

        See Also
        --------
        https://mlir.llvm.org/docs/Dialects/LLVM/#function-types
        """
        return ir.Type.parse("!llvm.func<{} ({})>".format(str(ret), ", ".join(map(str, params))))   # did not find a programmatic way to get the function type
    
    
if __name__ == "__main__":
    builder = MLIRTypeBuilder()
    with ir.Context(), ir.Location.unknown():
        assert str(builder.signed(32)) == "i32"
        assert str(builder.unsigned(32)) == "ui32", str(builder.unsigned(32))
        assert str(builder.void()) == "!llvm.void"
        assert str(builder.pointer()) == "!llvm.ptr"
        assert str(builder.struct(fields=[builder.signed(32), builder.unsigned(32)])) == "!llvm.struct<(i32, ui32)>"
        assert str(builder.struct(fields=[builder.signed(32), builder.unsigned(32)], packed=True)) == "!llvm.struct<packed (i32, ui32)>"
        assert str(builder.struct(name="MyStruct", fields=[builder.signed(32), builder.unsigned(32)])) == '!llvm.struct<"MyStruct", (i32, ui32)>'
        assert str(builder.identified_struct("MyStruct")) == '!llvm.struct<"MyStruct", (i32, ui32)>'
        print(builder.func(params=[builder.signed(32), builder.unsigned(32)], ret=builder.pointer()))
