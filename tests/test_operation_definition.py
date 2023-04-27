from __future__ import annotations

import pytest

from xdsl.dialects.builtin import IntAttr, StringAttr, i32

from xdsl.ir import Attribute, OpResult, Region
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedRegionSegments,
    AttrSizedResultSegments,
    OptAttributeDef,
    OptOperandDef,
    OptRegionDef,
    OptResultDef,
    VarOperandDef,
    VarRegionDef,
    VarResultDef,
    irdl_op_definition,
    OperandDef,
    ResultDef,
    AttributeDef,
    AnyAttr,
    OpDef,
    RegionDef,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException

################################################################################
#                              IRDL definition                                 #
################################################################################


@irdl_op_definition
class OpDefTestOp(IRDLOperation):
    name = "test.op_def_test"

    operand = OperandDef()
    result = ResultDef()
    attr = AttributeDef(Attribute)
    region = RegionDef()

    # Check that we can define methods in operation definitions
    def test(self):
        pass


def test_get_definition():
    """Test retrieval of an IRDL definition from an operation"""
    assert OpDefTestOp.irdl_definition == OpDef(
        "test.op_def_test",
        operands=[("operand", OperandDef(AnyAttr()))],
        results=[("result", ResultDef(AnyAttr()))],
        attributes={"attr": AttributeDef(Attribute)},
        regions=[("region", RegionDef())],
    )


################################################################################
#                                  Verifiers                                   #
################################################################################


@irdl_op_definition
class AttrOp(IRDLOperation):
    name: str = "test.two_var_result_op"
    attr = AttributeDef(StringAttr)


def test_attr_verify():
    op = AttrOp.create(attributes={"attr": IntAttr(1)})
    with pytest.raises(VerifyException) as e:
        op.verify()
    assert e.value.args[0] == "#int<1> should be of base attribute string"


################################################################################
#                                Accessors                                     #
################################################################################


@irdl_op_definition
class RegionOp(IRDLOperation):
    name = "test.region_op"

    irdl_options = [AttrSizedRegionSegments()]

    region = RegionDef()
    opt_region = OptRegionDef()
    var_region = VarRegionDef()


def test_region_accessors():
    """Test accessors for regions."""
    region1 = Region()
    region2 = Region()
    region3 = Region()
    region4 = Region()

    op = RegionOp.build(regions=[region1, [region2], [region3, region4]])
    assert op.region is op.regions[0]
    assert op.opt_region is op.regions[1]
    assert len(op.var_region) == 2
    assert op.var_region[0] is op.regions[2]
    assert op.var_region[1] is op.regions[3]

    region1 = Region()

    op = RegionOp.build(regions=[region1, [], []])
    assert op.opt_region is None
    assert len(op.var_region) == 0


@irdl_op_definition
class OperandOp(IRDLOperation):
    name = "test.operand_op"

    irdl_options = [AttrSizedOperandSegments()]

    operand = OperandDef()
    opt_operand = OptOperandDef()
    var_operand = VarOperandDef()


def test_operand_accessors():
    """Test accessors for operands."""
    operand1 = OpResult(i32, None, None)  # type: ignore
    operand2 = OpResult(i32, None, None)  # type: ignore
    operand3 = OpResult(i32, None, None)  # type: ignore
    operand4 = OpResult(i32, None, None)  # type: ignore

    op = OperandOp.build(operands=[operand1, [operand2], [operand3, operand4]])
    assert op.operand is op.operands[0]
    assert op.opt_operand is op.operands[1]
    assert len(op.var_operand) == 2
    assert op.var_operand[0] is op.operands[2]
    assert op.var_operand[1] is op.operands[3]

    op = OperandOp.build(operands=[operand1, [], []])
    assert op.opt_operand is None
    assert len(op.var_operand) == 0


@irdl_op_definition
class OpResultOp(IRDLOperation):
    name = "test.op_result_op"

    irdl_options = [AttrSizedResultSegments()]

    result = ResultDef()
    opt_result = OptResultDef()
    var_result = VarResultDef()


def test_opresult_accessors():
    """Test accessors for results."""
    op = OpResultOp.build(result_types=[i32, [i32], [i32, i32]])
    assert op.result is op.results[0]
    assert op.opt_result is op.results[1]
    assert len(op.var_result) == 2
    assert op.var_result[0] is op.results[2]
    assert op.var_result[1] is op.results[3]

    op = OpResultOp.build(result_types=[i32, [], []])
    assert op.opt_result is None
    assert len(op.var_result) == 0


@irdl_op_definition
class AttributeOp(IRDLOperation):
    name = "test.attribute_op"

    attr = AttributeDef(StringAttr)
    opt_attr = OptAttributeDef(StringAttr)


def test_attribute_accessors():
    """Test accessors for attributes."""

    op = AttributeOp.create(
        attributes={"attr": StringAttr("test"), "opt_attr": StringAttr("opt_test")}
    )
    assert op.attr is op.attributes["attr"]
    assert op.opt_attr is op.attributes["opt_attr"]

    op = AttributeOp.create(attributes={"attr": StringAttr("test")})
    assert op.opt_attr is None


def test_attribute_setters():
    """Test setters for attributes."""

    op = AttributeOp.create(attributes={"attr": StringAttr("test")})

    op.attr = StringAttr("new_test")
    assert op.attr.data == "new_test"

    op.opt_attr = StringAttr("new_opt_test")
    assert op.opt_attr.data == "new_opt_test"

    op.opt_attr = None
    assert op.opt_attr is None
