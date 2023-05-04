# RUN: python %s | mlir-opt --mlir-print-op-generic | filecheck %s

from xdsl.dialects.builtin import ModuleOp

from xdsl.dialects import memref
from xdsl.dialects.builtin import f32
from xdsl.printer import Printer
from xdsl.builder import Builder


@ModuleOp
@Builder.implicit_region
def module():
    input = memref.Alloc.get(f32, 0, [2004, 2004])

    memref.Subview.from_static_parameters(
        input, f32, [2004, 2004], [2, 2], [2000, 1000], [1, 1]
    )


p = Printer()
p.print_op(module)
# CHECK:
