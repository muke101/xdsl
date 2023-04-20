from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)

from xdsl.dialects import riscv
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import riscv_buffer


class LowerBufferAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: riscv_buffer.BufferAddOp, rewriter: PatternRewriter
    ):
        rewriter.replace_matched_op(
            riscv.CustomEmulatorInstructionOp("buffer.add", op.operands, ())
        )


class LowerBufferMulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: riscv_buffer.BufferMulOp, rewriter: PatternRewriter
    ):
        rewriter.replace_matched_op(
            riscv.CustomEmulatorInstructionOp("buffer.mul", op.operands, ())
        )


class LowerRISCVBuffer(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerBufferAddOp()).rewrite_module(op)
        PatternRewriteWalker(LowerBufferMulOp()).rewrite_module(op)
        dce(op)
