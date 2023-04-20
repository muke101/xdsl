from __future__ import annotations

from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import IRDLOperation, irdl_op_definition, Operand
from xdsl.dialects.riscv import Register, RegisterType

from typing import Annotated

from xdsl.traits import Pure


@irdl_op_definition
class TensorPrintOp(IRDLOperation):
    name = "riscv.toy.print"

    rs1: Annotated[Operand, RegisterType]

    def __init__(self, rs1: Operation | SSAValue):
        super().__init__(operands=[rs1], result_types=[])


@irdl_op_definition
class AllocOp(IRDLOperation):
    """
    Allocate a buffer of `count` ints, or `count` * 4 bytes
    """

    name = "riscv.buffer.alloc"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]

    traits = frozenset([Pure()])

    def __init__(self, count_reg: Operation | SSAValue):
        super().__init__(operands=[count_reg], result_types=[RegisterType(Register())])


@irdl_op_definition
class BufferAddOp(IRDLOperation):
    name = "riscv.buffer.add"

    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    rs3: Annotated[Operand, RegisterType]

    def __init__(
        self,
        count_reg: Operation | SSAValue,
        source: Operation | SSAValue,
        destination: Operation | SSAValue,
    ):
        super().__init__(operands=[count_reg, source, destination])


@irdl_op_definition
class BufferMulOp(IRDLOperation):
    name = "riscv.buffer.mul"

    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    rs3: Annotated[Operand, RegisterType]

    def __init__(
        self,
        count_reg: Operation | SSAValue,
        source: Operation | SSAValue,
        destination: Operation | SSAValue,
    ):
        super().__init__(operands=[count_reg, source, destination])


ToyRISCV = Dialect(
    [
        TensorPrintOp,
        AllocOp,
        BufferAddOp,
        BufferMulOp,
    ],
    [],
)
