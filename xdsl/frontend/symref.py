from __future__ import annotations
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    AttributeDef,
    OperandDef,
    ResultDef,
    irdl_op_definition,
    Operation,
    IRDLOperation,
)
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr


@irdl_op_definition
class Declare(IRDLOperation):
    name: str = "symref.declare"
    sym_name = AttributeDef(StringAttr)

    @staticmethod
    def get(sym_name: str | StringAttr) -> Declare:
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        return Declare.build(attributes={"sym_name": sym_name})


@irdl_op_definition
class Fetch(IRDLOperation):
    name: str = "symref.fetch"
    value = ResultDef()
    symbol = AttributeDef(SymbolRefAttr)

    @staticmethod
    def get(symbol: str | SymbolRefAttr, result_type: Attribute) -> Fetch:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        return Fetch.build(attributes={"symbol": symbol}, result_types=[result_type])


@irdl_op_definition
class Update(IRDLOperation):
    name: str = "symref.update"
    value = OperandDef()
    symbol = AttributeDef(SymbolRefAttr)

    @staticmethod
    def get(symbol: str | SymbolRefAttr, value: Operation | SSAValue) -> Update:
        if isinstance(symbol, str):
            symbol = SymbolRefAttr(symbol)
        return Update.build(operands=[value], attributes={"symbol": symbol})


Symref = Dialect([Declare, Fetch, Update], [])
