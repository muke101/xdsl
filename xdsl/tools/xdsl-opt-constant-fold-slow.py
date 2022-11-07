#!/usr/bin/env python3

import argparse
from typing import cast
from xdsl.elevate import *
from xdsl.immutable_ir import IOp, IResult, ISSAValue, from_op, get_immutable_copy
from xdsl.immutable_utils import GarbageCollect, GarbageCollect_new
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.pattern_rewriter import RewriteOnceWalker, RewritePattern, PatternRewriter, GreedyRewritePatternApplier, PatternRewriteWalker

from xdsl.dialects.builtin import ModuleOp, IntegerAttr
from xdsl.dialects.arith import Constant, Addi
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
from xdsl.ir import Operation, MLContext
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)
from dataclasses import dataclass
# from memory_profiler import profile
import sys

to_keep_in_memory = list[Any]()


def flush_memory(): 
    global to_keep_in_memory
    to_keep_in_memory = list[Any]()

@dataclass
class ConstantFoldAddRewriter(RewritePattern):

    num_changed_values: int = 0

    def is_integer_literal(self, op: Operation) -> bool:
        return isinstance(op, Constant) and isinstance(op.value, IntegerAttr)

    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, expr: Addi, rewriter: PatternRewriter) -> None:
        if self.is_integer_literal(expr.lhs.op) and self.is_integer_literal(
                expr.rhs.op):
            lhs_value = expr.lhs.op.value.parameters[0].data
            rhs_value = expr.rhs.op.value.parameters[0].data
            result_value = lhs_value + rhs_value

            new_constant = Constant.from_int_constant(result_value,
                                                      expr.results[0].typ)
            rewriter.replace_op(expr, [new_constant])
            self.num_changed_values += 1
        return


@dataclass
class ConstantFoldAndIRewriter(RewritePattern):

    num_changed_values: int = 0

    def is_integer_literal(self, op: Operation) -> bool:
        return isinstance(op, Constant) and isinstance(op.value, IntegerAttr)

    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, expr: AndI, rewriter: PatternRewriter) -> None:
        if self.is_integer_literal(expr.lhs.op) and self.is_integer_literal(
                expr.rhs.op):
            lhs_value: OpResult = expr.lhs.op.value.parameters[0].data
            rhs_value: OpResult = expr.rhs.op.value.parameters[0].data
            result_value = lhs_value & rhs_value

            new_constant = Constant.from_int_constant(result_value,
                                                      expr.results[0].typ)
            # also erases expr
            rewriter.replace_op(expr, [new_constant])
            if len(expr.lhs.uses) == 0:
                rewriter.erase_op(expr.lhs.op)
            if len(expr.rhs.uses) == 0:
                rewriter.erase_op(expr.rhs.op)
            self.num_changed_values += 1
        return

@dataclass
class InlineIfRewriter(RewritePattern):

    num_changed_values: int = 0

    def is_integer_literal(self, op: Operation) -> bool:
        return isinstance(op, Constant) and isinstance(op.value, IntegerAttr)

    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, expr: scf.If, rewriter: PatternRewriter) -> None:
        if self.is_integer_literal(expr.cond.op) and expr.cond.op.value.parameters[0].data == 1:
            if isinstance((yield_op := expr.true_region.blocks[0].ops[-1]), scf.Yield):
                rewriter.replace_op(expr, expr.true_region.blocks[0].ops[:-1], new_results=[yield_op.operands[0]])
                self.num_changed_values += len(expr.true_region.blocks[0].ops)
        return

@dataclass
class RemoveUnusedRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        for r in op.results:
            if len(r.uses) == 0:
                rewriter.erase_matched_op()


# Assign rvalue to lvalue, and delete rvalue contents.
def assign_module_and_delete(lvalue: ModuleOp, rvalue: ModuleOp, keep_in_memory: bool=False):
    lregion = lvalue.regions[0]
    rregion = rvalue.regions[0]
    lvalue.regions[0] = rregion
    rregion.parent = lvalue
    lregion.parent = None
    if keep_in_memory:
        global to_keep_in_memory
        to_keep_in_memory.append(lregion)

def save_module_for_backtracking(module: IOp):
    global to_keep_in_memory
    to_keep_in_memory.append(module)


def constant_folding_clone(ctx: MLContext, module: ModuleOp, keep_in_memory: bool) -> None:
    rewriter = ConstantFoldAddRewriter()
    walker = RewriteOnceWalker(rewriter)
    cleanup_walker = RewriteOnceWalker(RemoveUnusedRewriter())

    module_copy = module.clone()
    if not walker.rewrite_module(module_copy):
        return

    assign_module_and_delete(module, module_copy, keep_in_memory=keep_in_memory)
    module_copy = module.clone()

    while walker.rewrite_module(module_copy):
        assign_module_and_delete(module, module_copy, keep_in_memory=keep_in_memory)
        module_copy = module.clone()
        cleanup_walker.rewrite_module(module)
        cleanup_walker.rewrite_module(module)

    # flush_memory()


def constant_folding_fast(ctx: MLContext, module: ModuleOp) -> None:
    walker = RewriteOnceWalker(ConstantFoldAddRewriter())
    cleanup_walker = RewriteOnceWalker(RemoveUnusedRewriter())

    while walker.rewrite_module(module):
        cleanup_walker.rewrite_module(module)
        cleanup_walker.rewrite_module(module)
        pass
def bool_nest_cloning(ctx: MLContext, module: ModuleOp, keep_in_memory: bool) -> None:
    fold_and_rewriter = ConstantFoldAndIRewriter()
    fold_and_once = RewriteOnceWalker(fold_and_rewriter)
    inline_if_rewriter = InlineIfRewriter()
    inline_if_once = RewriteOnceWalker(inline_if_rewriter)
    cleanup_walker = PatternRewriteWalker(RemoveUnusedRewriter())

    module_copy = module.clone()

    # currently we don't clone before doing cleanup!
    old_versions: list[ModuleOp] = [module]
    while fold_and_once.rewrite_module(module_copy):
        assign_module_and_delete(module, module_copy, keep_in_memory=keep_in_memory)
        module_copy = module.clone()
        old_versions.append(module)
        # cleanup_walker.rewrite_module(module_copy)
        # assign_module_and_delete(module, module_copy, keep_in_memory=keep_in_memory)

    while inline_if_once.rewrite_module(module_copy):
        assign_module_and_delete(module, module_copy, keep_in_memory=keep_in_memory)
        module_copy = module.clone()
        old_versions.append(module)
        # cleanup_walker.rewrite_module(module_copy)

    assign_module_and_delete(module, module_copy, keep_in_memory=keep_in_memory)
    # global to_keep_in_memory
    # print("Keeping in memory: ", len(to_keep_in_memory))
    flush_memory()

@dataclass(frozen=True)
class FoldConstantAdd(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(
                op_type=arith.Addi,
                operands=[IResult(op=IOp(op_type=arith.Constant,
                                        attributes={"value": IntegerAttr() as attr1}) as c1),
                          IResult(op=IOp(op_type=arith.Constant,
                                        attributes={"value": IntegerAttr() as attr2}))]):
                result = from_op(c1,
                        attributes={
                            "value":
                            IntegerAttr.from_params(
                                attr1.value.data + attr2.value.data,
                                attr1.typ)
                        })
                return success(result)
            case _:
                return failure(self)


@dataclass(frozen=True)
class FoldConstantAnd(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(
                op_type=arith.AndI,
                operands=[IResult(op=IOp(op_type=arith.Constant,
                                        attributes={"value": IntegerAttr() as attr1}) as c1),
                          IResult(op=IOp(op_type=arith.Constant,
                                        attributes={"value": IntegerAttr() as attr2}))]):

                result = from_op(c1,
                        attributes={
                            "value":
                            IntegerAttr.from_params(
                                attr1.value.data & attr2.value.data,
                                attr1.typ)
                        })
                return success(result)
            case _:
                return failure(self)


@dataclass(frozen=True)
class InlineIf(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=scf.If,
                        operands=[IResult(op=IOp(op_type=arith.Constant, attributes={"value": IntegerAttr(value=IntAttr(data=1))}))],
                        region=IRegion(ops=ops)):
                        return success(ops[:-1] if len(ops) > 0 and (ops[-1].op_type==scf.Yield) else ops)
            case _:
                return failure(self)


@dataclass(frozen=True)
class FakeDCE(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=ModuleOp):
                new_region = IRegion([IBlock([], op.region.ops[2:])])
                result = from_op(op, regions=[new_region])
                return success(result)
            case _:
                return failure(self)


def constant_folding_composable2(ctx: MLContext, module: ModuleOp) -> None:
    sys.setrecursionlimit(1000000)
    imodule: IOp = get_immutable_copy(module)
    
    # Applying small strategies after another
    # strategy = try_(topToBottom(FoldConstantAdd()))
    # new_imodule = imodule
    # for _ in range(1000):
    #     new_imodule = (strategy ^ topToBottom(FakeDCE())).apply(new_imodule).result_op
    #     new_imodule = (strategy ^ try_(topToBottom(FoldConstantAdd()))).apply(new_imodule).result_op
        
    # Building one giant strategy and applying it once
    strategy = try_(topToBottom(FoldConstantAdd()))
    
    for _ in range(300):
        strategy =  strategy ^ (topToBottom(FakeDCE())) ^ try_(topToBottom(FoldConstantAdd())) 

    new_imodule = strategy.apply(imodule).result_op
    
    new_module = new_imodule.get_mutable_copy()

    # TODO: find a way to not free the memory of the intermediate modules we might need for backtracking

    assign_module_and_delete(module, cast(ModuleOp, new_module))

def constant_folding_composable(ctx: MLContext, module: ModuleOp) -> None:
    sys.setrecursionlimit(10000)
    imodule: IOp = get_immutable_copy(module)
    new_imodule = everywhere(FoldConstantAdd()).apply(imodule).result_op
    new_imodule = topToBottom(GarbageCollect_new()).apply(new_imodule).result_op

    new_module = new_imodule.get_mutable_copy()
    assign_module_and_delete(module, cast(ModuleOp, new_module))

def bool_nest_composable(ctx: MLContext, module: ModuleOp, keep_in_memory: bool) -> None:
    sys.setrecursionlimit(1000000)
    imodule: IOp = get_immutable_copy(module)
    # Individual applications of small strategies
    if keep_in_memory:
            save_module_for_backtracking(imodule)
    new_imodule = imodule
    while (rr := topToBottom(FoldConstantAnd()).apply(new_imodule)).isSuccess():
        new_imodule = rr.result_op
        if keep_in_memory:
            save_module_for_backtracking(new_imodule)
    while (rr := topToBottom(InlineIf()).apply(new_imodule)).isSuccess():
        new_imodule = rr.result_op
        if keep_in_memory:
            save_module_for_backtracking(new_imodule)

    # Build a large strategy and apply it once
    # strategy = everywhere(FoldConstantAnd()) ^ everywhere(InlineIf())
    # new_imodule = strategy.apply(imodule).result_op
    
    new_module = new_imodule.get_mutable_copy()
    # global to_keep_in_memory
    # print("Keeping in memory: ", len(to_keep_in_memory))
    assign_module_and_delete(module, cast(ModuleOp, new_module))
    flush_memory()

class OptMain(xDSLOptMain):

    def register_all_dialects(self):
        super().register_all_dialects()

    def register_all_passes(self):
        super().available_passes['constant-fold-clone'] = lambda ctx, module: constant_folding_clone(ctx, module, self.args.keep_copied_module)
        super().available_passes['bool-nest-clone'] = lambda ctx, module: bool_nest_cloning(ctx, module, self.args.keep_copied_module)
        super().available_passes['constant-fold-fast'] = constant_folding_fast
        super().available_passes[
            'constant-fold-composable'] = constant_folding_composable
        super().available_passes[
            'constant-fold-composable2'] = constant_folding_composable2
        super().available_passes[
            'bool-nest-composable'] = lambda ctx, module: bool_nest_composable(ctx, module, self.args.keep_copied_module)
        super().register_all_passes()

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)
        arg_parser.add_argument("--keep-copied-module",
                        default=True,
                        action='store_true')


def __main__(args: Optional[argparse.Namespace] = None):
    xdsl_main = OptMain()
    if args:
        xdsl_main.args = args

    xdsl_main.run()


if __name__ == "__main__":
    # fixed_args = argparse.Namespace(input_file='experiments/program_10.xdsl',
    #                     target='xdsl',
    #                     frontend=None,
    #                     disable_verify=False,
    #                     output_file=None,
    #                     passes='constant-fold-composable',
    #                     print_between_passes=False,
    #                     verify_diagnostics=False,
    #                     use_mlir_bindings=False,
    #                     allow_unregistered_ops=False)

    __main__()