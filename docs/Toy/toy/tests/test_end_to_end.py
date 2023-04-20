from io import StringIO
from xdsl.builder import Builder

from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FunctionType,
    ModuleOp,
    f64,
    i32,
)

from xdsl.printer import Printer

from ..rewrites.lower_toy import LowerToy
from ..rewrites.optimise_toy import OptimiseToy
from ..rewrites.optimise_vector import OptimiseVector

from ..compiler import (
    compile,
    parse_toy,
    context,
)
from ..emulator.emulator_iop import run_riscv
from ..emulator.toy_accelerator import ToyAccelerator

from ..dialects import toy, vector

# from ..dialects import toy, riscv, riscv_ssa, riscv_buffer, vector

ctx = context()

toy_program = """
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<3, 2> = [1, 2, 3, 4, 5, 6];

  # There is a built-in print instruction to display the contents of the tensor
  print(b);

  # Reshapes are implicit on assignment
  var c<2, 3> = b;

  # There are + and * operators for pointwise addition and multiplication
  var d = a + c;

  print(d);
}
"""


def desc(op: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream).print(op)
    return stream.getvalue()


@ModuleOp
@Builder.implicit_region
def toy_0():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b_0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]).res
        b = toy.ReshapeOp(b_0, [3, 2]).res
        toy.PrintOp(b)
        c = toy.ReshapeOp(b, [2, 3]).res
        d = toy.AddOp(a, c).res
        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def toy_1():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [3, 2]).res
        toy.PrintOp(b)
        c = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        d = toy.AddOp(a, c).res
        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def vir_0():
    main_type = FunctionType.from_lists([], [])

    def vector_i32(elements: list[int]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, i32)

    def vector_f64(elements: list[float]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, f64)

    def tensor_type(shape: list[int]) -> toy.TensorTypeF64:
        return toy.TensorTypeF64.from_type_and_list(f64, shape)

    @Builder.implicit_region
    def main() -> None:
        a_shape = vector.VectorConstantOp(vector_i32([2, 3]), "tensor_shape").res
        a_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        a = vector.TensorMakeOp(a_shape, a_data, tensor_type([2, 3])).tensor
        b_shape = vector.VectorConstantOp(vector_i32([3, 2]), "tensor_shape").res
        b_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        b = vector.TensorMakeOp(b_shape, b_data, tensor_type([3, 2])).tensor
        toy.PrintOp(b)

        c_shape = vector.VectorConstantOp(vector_i32([2, 3]), "tensor_shape").res
        c_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        c = vector.TensorMakeOp(c_shape, c_data, tensor_type([2, 3])).tensor

        d_shape = vector.TensorShapeOp(a).data
        lhs = vector.TensorDataOp(a).data
        rhs = vector.TensorDataOp(c).data
        d_data = vector.VectorAddOp(lhs, rhs).res
        d = vector.TensorMakeOp(d_shape, d_data, tensor_type([2, 3])).tensor

        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def vir_1():
    main_type = FunctionType.from_lists([], [])

    def vector_i32(elements: list[int]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, i32)

    def vector_f64(elements: list[float]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, f64)

    def tensor_type(shape: list[int]) -> toy.TensorTypeF64:
        return toy.TensorTypeF64.from_type_and_list(f64, shape)

    @Builder.implicit_region
    def main() -> None:
        a_shape = vector.VectorConstantOp(vector_i32([2, 3]), "tensor_shape").res
        a_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res

        b_shape = vector.VectorConstantOp(vector_i32([3, 2]), "tensor_shape").res
        b_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        b = vector.TensorMakeOp(b_shape, b_data, tensor_type([3, 2])).tensor
        toy.PrintOp(b)

        c_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res

        d_data = vector.VectorAddOp(a_data, c_data).res
        d = vector.TensorMakeOp(a_shape, d_data, tensor_type([2, 3])).tensor

        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


risc_0 = ModuleOp(ops=[])  # lower_to_riscv(vir_1)


riscv_code = """.bss
heap:
.space 1024
.data
main.tensor_shape.0:
.word 0x2, 0x2, 0x3
main.tensor_data.0:
.word 0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
main.tensor_shape.1:
.word 0x2, 0x3, 0x2
main.tensor_data.1:
.word 0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
main.tensor_data.2:
.word 0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
.text
main:
    li j0, heap
    li j1, main.tensor_shape.0
    li j2, main.tensor_data.0
    li j3, main.tensor_shape.1
    li j4, main.tensor_data.1
    li j5, 2                                     # Tensor storage
    buffer.alloc j6, j5
    sw j3, j6, 0                                 # Set tensor shape
    lw j7, j6, 0
    sw j4, j6, 4                                 # Set tensor data
    toy.print j6
    li j8, main.tensor_data.2
    lw j9, j2, 0                                 # Get input count
    addi j10, j9, 1                              # Input storage int32 count
    buffer.alloc j11, j10
    sw j9, j11, 0                                # Set result count
    addi j12, j2, 4                              # lhs storage
    addi j13, j8, 4                              # rhs storage
    addi j14, j11, 4                             # destination storage
    buffer.add j9, j12, j14
    buffer.add j9, j13, j14
    li j15, 2                                    # Tensor storage
    buffer.alloc j16, j15
    sw j1, j16, 0                                # Set tensor shape
    lw j17, j16, 0
    sw j11, j16, 4                               # Set tensor data
    toy.print j16
    li a7, 93
    ecall
"""


def test_compile():
    code = compile(toy_program)
    assert code == riscv_code


def test_parse_toy():
    assert desc(toy_0) == desc(parse_toy(toy_program))
    assert toy_0.is_structurally_equivalent(parse_toy(toy_program))


def test_optimise_toy():
    assert desc(toy_1) == desc(OptimiseToy().apply_to_clone(ctx, toy_0))
    assert toy_1.is_structurally_equivalent(OptimiseToy().apply_to_clone(ctx, toy_0))


def test_lower_from_toy():
    assert desc(vir_0) == desc(LowerToy().apply_to_clone(ctx, toy_1))
    assert vir_0.is_structurally_equivalent(LowerToy().apply_to_clone(ctx, toy_1))


def test_optimise_vir():
    assert desc(vir_1) == desc(OptimiseVector().apply_to_clone(ctx, vir_0))
    assert vir_1.is_structurally_equivalent(OptimiseVector().apply_to_clone(ctx, vir_0))


# TODO: when the riscv dialects stabilise a bit

# def test_lower_to_riscv():
#     assert desc(risc_0) == desc(lower_to_riscv(vir_1))
#     assert risc_0.is_structurally_equivalent(lower_to_riscv(vir_1))


# def test_print_riscv_ssa():
#     assert riscv_code == print_riscv_ssa(risc_0)


def test_execution():
    stream = StringIO()
    ToyAccelerator.stream = stream
    run_riscv(riscv_code, extensions=[ToyAccelerator], unlimited_regs=True, verbosity=1)
    assert "[[2, 4, 6], [8, 10, 12]]" in stream.getvalue()
