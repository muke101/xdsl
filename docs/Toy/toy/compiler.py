from pathlib import Path


from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.riscv_asm_writer import riscv_code
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.lower_riscv_ssa import LowerRISCVSSA
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc

from .frontend.ir_gen import IRGen
from .frontend.parser import Parser

from .rewrites.lower_toy import LowerToy
from .rewrites.optimise_toy import OptimiseToy
from .rewrites.lower_vector import LowerVector
from .rewrites.optimise_vector import OptimiseVector
from .rewrites.lower_riscv_buffer import LowerRISCVBuffer

from .emulator.emulator_iop import run_riscv
from .emulator.toy_accelerator import ToyAccelerator

from .dialects import riscv_debug, toy, vector, riscv_buffer
from xdsl.dialects import riscv, riscv_ssa, riscv_func


def parse_toy(program: str, ctx: MLContext | None = None) -> ModuleOp:
    mlir_gen = IRGen()
    module_ast = Parser(Path("in_memory"), program).parseModule()
    module_op = mlir_gen.ir_gen_module(module_ast)
    return module_op


def context() -> MLContext:
    ctx = MLContext()
    ctx.register_dialect(toy.Toy)
    ctx.register_dialect(vector.Vector)
    ctx.register_dialect(riscv_buffer.ToyRISCV)
    ctx.register_dialect(riscv_debug.RISCV_DEBUG)
    ctx.register_dialect(riscv.RISCV)
    ctx.register_dialect(riscv_ssa.RISCV_SSA)
    ctx.register_dialect(riscv_func.RISCV_FUNC)
    return ctx


def compile(program: str) -> str:
    ctx = context()

    op = parse_toy(program)

    OptimiseToy().apply(ctx, op)
    LowerToy().apply(ctx, op)
    OptimiseVector().apply(ctx, op)
    LowerVector().apply(ctx, op)
    LowerRISCVBuffer().apply(ctx, op)
    LowerRISCVFunc().apply(ctx, op)
    LowerRISCVSSA().apply(ctx, op)
    RISCVRegisterAllocation().apply(ctx, op)
    code = riscv_code(op)

    return code


def emulate_riscv(program: str):
    run_riscv(program, extensions=[ToyAccelerator], unlimited_regs=True, verbosity=1)
