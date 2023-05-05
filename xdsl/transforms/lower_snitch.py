from dataclasses import dataclass
from xdsl.passes import ModulePass

from xdsl.ir import MLContext

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.dialects import snitch, builtin


class LowerSsrSetupShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrSetupShape, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op([], [])


class LowerSsrEnable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrEnable, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op([], [])


class LowerSsrSetupRepetition(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetupRepetition, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op([], [])


class LowerSsrRead(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrRead, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op([], [])


class LowerSsrWrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrWrite, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op([], [])


class LowerSsrDisable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrDisable, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op([], [])


@dataclass
class LowerSnitchPass(ModulePass):
    name = "lower-snitch"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerSsrSetupShape(),
                    LowerSsrSetupRepetition(),
                    LowerSsrRead(),
                    LowerSsrWrite(),
                    LowerSsrEnable(),
                    LowerSsrDisable(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
