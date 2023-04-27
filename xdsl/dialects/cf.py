from __future__ import annotations

from typing import List, Union

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import SSAValue, Operation, Block, Dialect
from xdsl.irdl import (
    OperandDef,
    VarOperandDef,
    irdl_op_definition,
    AttrSizedOperandSegments,
    IRDLOperation,
)


@irdl_op_definition
class Branch(IRDLOperation):
    name: str = "cf.br"

    arguments = VarOperandDef()

    @staticmethod
    def get(block: Block, *ops: Union[Operation, SSAValue]) -> Branch:
        return Branch.build(operands=[[op for op in ops]], successors=[block])


@irdl_op_definition
class ConditionalBranch(IRDLOperation):
    name: str = "cf.cond_br"

    cond = OperandDef(IntegerType(1))
    then_arguments = VarOperandDef()
    else_arguments = VarOperandDef()

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(
        cond: Union[Operation, SSAValue],
        then_block: Block,
        then_ops: List[Union[Operation, SSAValue]],
        else_block: Block,
        else_ops: List[Union[Operation, SSAValue]],
    ) -> ConditionalBranch:
        return ConditionalBranch.build(
            operands=[cond, then_ops, else_ops], successors=[then_block, else_block]
        )


Cf = Dialect([Branch, ConditionalBranch], [])
