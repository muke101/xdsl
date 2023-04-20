from collections import Counter
from dataclasses import dataclass

from xdsl.ir import MLContext, Operation, Block, Region, SSAValue
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.dialects import riscv, riscv_ssa, riscv_func
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy as td
from ..dialects import riscv_buffer as rbd
from ..dialects import vector as tvd


class AddSections(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
        # bss stands for block starting symbol
        heap_section = riscv_ssa.SectionOp(
            ".bss",
            Region(
                Block(
                    [
                        riscv.LabelOp("heap"),
                        riscv.DirectiveOp(".space", f"{1024}"),  # 1kb
                    ]
                )
            ),
        )
        data_section = riscv_ssa.SectionOp(".data", Region(Block()))
        text_section = riscv_ssa.SectionOp(
            ".text", rewriter.move_region_contents_to_new_regions(op.regions[0])
        )

        op.body.add_block(Block([heap_section, data_section, text_section]))


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.FuncOp, rewriter: PatternRewriter):
        name = op.sym_name.data

        # TODO: add support for user defined functions
        assert name == "main", "Only support lowering main function for now"

        region = op.regions[0]

        # insert a heap pointer at the start of every function
        # TODO: replace with insert_op_at_start
        first_op = region.blocks[0].first_op
        assert first_op is not None
        rewriter.insert_op_before(riscv.LiOp("heap"), first_op)

        # create riscv func op with same ops
        riscv_op = riscv_func.FuncOp(
            name, rewriter.move_region_contents_to_new_regions(region)
        )

        rewriter.replace_matched_op(riscv_op)


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ReturnOp, rewriter: PatternRewriter):
        # TODO: add support for optional argument
        assert op.input is None, "Only support return with no arguments for now"

        rewriter.replace_matched_op(riscv_func.ReturnOp())


class DataSectionRewritePattern(RewritePattern):
    _data_section: riscv_ssa.SectionOp | None = None
    _counter: Counter[str] = Counter()

    def data_section(self, op: Operation) -> riscv_ssa.SectionOp:
        """
        Relies on the data secition being inserted earlier by AddDataSection
        """
        if self._data_section is None:
            module_op = op.get_toplevel_object()
            assert isinstance(
                module_op, ModuleOp
            ), f"The top level object of {str(op)} must be a ModuleOp"

            for op in module_op.body.blocks[0].ops:
                if not isinstance(op, riscv_ssa.SectionOp):
                    continue
                if op.directive.data != ".data":
                    continue
                self._data_section = op

            assert self._data_section is not None

        return self._data_section

    def label(self, func_name: str, kind: str) -> str:
        key = f"{func_name}.{kind}"
        count = self._counter[key]
        self._counter[key] += 1
        return f"{key}.{count}"

    def func_name_of_op(self, op: Operation) -> str:
        region = op.parent_region()
        assert region is not None
        func_op = region.parent_op()
        assert isinstance(func_op, riscv_func.FuncOp)
        return func_op.func_name.data

    def add_data(self, op: Operation, label: str, data: list[int]):
        encoded_data = ", ".join(hex(el) for el in data)
        self.data_section(op).regions[0].blocks[0].add_ops(
            [riscv.LabelOp(label), riscv.DirectiveOp(".word", encoded_data)]
        )


class LowerVectorConstantOp(DataSectionRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tvd.VectorConstantOp, rewriter: PatternRewriter):
        """
        Vectors are represented in memory as an n+1 array of int32, where the first
        entry is the count of the vector
        """
        data = op.get_data()
        label = self.label(self.func_name_of_op(op), op.label.data)

        self.add_data(op, label, [len(data), *data])
        rewriter.replace_matched_op(riscv.LiOp(label))


class LowerPrintOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.PrintOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(rbd.TensorPrintOp(op.input))


class LowerVectorAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tvd.VectorAddOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            [
                count := riscv.LwOp(op.lhs, 0, comment="Get input count"),
                storage_count := riscv.AddiOp(
                    count, 1, comment="Input storage int32 count"
                ),
                vector := rbd.AllocOp(storage_count),
                riscv.SwOp(count, vector, 0, comment="Set result count"),
                lhs := riscv.AddiOp(op.lhs, 4, comment="lhs storage"),
                rhs := riscv.AddiOp(op.rhs, 4, comment="rhs storage"),
                dest := riscv.AddiOp(vector, 4, comment="destination storage"),
                rbd.BufferAddOp(count, lhs, dest),
                rbd.BufferAddOp(count, rhs, dest),
            ],
            [vector.rd],
        )


class LowerVectorMulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tvd.VectorMulOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            [
                count := riscv.LwOp(op.lhs, 0, comment="Get input count"),
                storage_count := riscv.AddiOp(
                    count, 1, comment="Input storage int32 count"
                ),
                vector := rbd.AllocOp(storage_count),
                riscv.SwOp(count, vector, 0, comment="Set result count"),
                lhs := riscv.AddiOp(op.lhs, 4, comment="lhs storage"),
                rhs := riscv.AddiOp(op.rhs, 4, comment="rhs storage"),
                dest := riscv.AddiOp(vector, 4, comment="destination storage"),
                rbd.BufferAddOp(count, lhs, dest),
                rbd.BufferMulOp(count, rhs, dest),
            ],
            [vector.rd],
        )


class LowerTensorMakeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tvd.TensorMakeOp, rewriter: PatternRewriter):
        shape = op.shape
        data = op.data

        tensor_storage_len_op = riscv.LiOp(2, comment="Tensor storage")
        tensor_op = rbd.AllocOp(tensor_storage_len_op)
        tensor_set_shape_op = riscv.SwOp(
            shape, tensor_op, 0, comment="Set tensor shape"
        )
        tensor_set_data_op = riscv.SwOp(data, tensor_op, 4, comment="Set tensor data")

        rewriter.replace_matched_op(
            [
                tensor_storage_len_op,
                tensor_op,
                tensor_set_shape_op,
                riscv.LwOp(tensor_op, 0),
                tensor_set_data_op,
            ],
            [tensor_op.rd],
        )


class LowerTensorShapeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tvd.TensorShapeOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            riscv.LwOp(op.tensor, 0, comment="Get tensor shape")
        )


class LowerTensorDataOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tvd.TensorDataOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(riscv.LwOp(op.tensor, 4, comment="Get tensor data"))


class LowerTensorPrintOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rbd.TensorPrintOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            riscv.CustomEmulatorInstructionOp("toy.print", op.operands, [])
        )


@dataclass
class LowerAllocOp(RewritePattern):
    use_accelerator: bool = True

    def heap_address(self, op: Operation) -> SSAValue:
        block = op.parent_block()
        assert block is not None
        heap_op = block.first_op
        assert heap_op is not None
        # TODO: check that this is indeed the heap op
        # assert isinstance(heap_op, rd.LiOp)
        # and isinstance(heap_op.immediate, rd.LabelAttr)
        return heap_op.results[0]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rbd.AllocOp, rewriter: PatternRewriter):
        if self.use_accelerator:
            rewriter.replace_matched_op(
                riscv.CustomEmulatorInstructionOp(
                    "buffer.alloc", op.operands, [riscv.RegisterType(riscv.Register())]
                )
            )
            return

        heap_ptr = self.heap_address(op)

        rewriter.replace_matched_op(
            [
                four := riscv.LiOp(4, comment="4 bytes per int"),
                count := riscv.MULOp(op.rs1, four, comment="Alloc count bytes"),
                old_heap_count := riscv.LwOp(heap_ptr, 0, comment="Old heap count"),
                new_heap_count := riscv.AddOp(
                    old_heap_count, count, comment="New heap count"
                ),
                riscv.SwOp(new_heap_count, heap_ptr, 0, comment="Update heap"),
                heap_storage_start := riscv.AddiOp(
                    heap_ptr, 4, comment="Heap storage start"
                ),
                result := riscv.AddOp(
                    heap_storage_start, old_heap_count, comment="Allocated memory"
                ),
            ],
            [result.rd],
        )


class LowerVector(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(AddSections()).rewrite_module(op)
        PatternRewriteWalker(LowerFuncOp()).rewrite_module(op)
        PatternRewriteWalker(LowerReturnOp()).rewrite_module(op)
        PatternRewriteWalker(LowerPrintOp()).rewrite_module(op)

        PatternRewriteWalker(LowerVectorConstantOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorMakeOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorShapeOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorDataOp()).rewrite_module(op)
        PatternRewriteWalker(LowerVectorAddOp()).rewrite_module(op)
        PatternRewriteWalker(LowerVectorMulOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorPrintOp()).rewrite_module(op)

        PatternRewriteWalker(LowerAllocOp()).rewrite_module(op)

        dce(op)
