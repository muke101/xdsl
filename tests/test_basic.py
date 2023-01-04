from xdsl.ir import Block, Region, SSAValue
from xdsl.dialects.arith import Constant, Mulf, Addf
from xdsl.dialects.affine import For
from xdsl.dialects.builtin import i32, f32, IntegerType, IntAttr
from xdsl.dialects.memref import Alloc, Load, Store, MemRefType, Alloca


from xdsl.printer import Printer

def test_combo():
    cst = Constant.from_float_and_width(0.0, f32)
    N = Constant.from_int_and_width(1, i32)
    M = Constant.from_int_and_width(1, i32)
    K = Constant.from_int_and_width(1, i32)

    my_i32 = IntegerType.from_width(32)

    lhs = Alloc.get(my_i32, 32, [2, 2])
    rhs = Alloc.get(my_i32, 32, [2, 2])
    out = Alloc.get(my_i32, 32, [2, 2])

    k = SSAValue(typ=i32)
    j = SSAValue(typ=i32)

    load3 = Load.get(lhs, [k, j])
    load4 = Load.get(rhs, [k, j])

    mul5 = Mulf(load3, load4)

    block0 = Block.from_ops([mul5])
    region0 = Region.from_block_list([block0])
    # import pdb;pdb.set_trace()





def test_matmul():
    """
    Test adapted from:
    https://github.com/wehu/c-mlir#readme
    """

    cst = Constant.from_float_and_width(0.0, f32)
    N = Constant.from_int_and_width(1, i32)
    M = Constant.from_int_and_width(2, i32)
    K = Constant.from_int_and_width(3, i32)

    my_i32 = IntegerType.from_width(32)

    lhs = Alloca.get(my_i32, 64, [K, N])
    rhs = Alloca.get(my_i32, 64, [N, M])
    output = Alloca.get(my_i32, 64, [K, M])

    k = SSAValue(typ=i32)
    j = SSAValue(typ=i32)
    i = SSAValue(typ=i32)

    store0 = Store.get(cst, output, [i, j])

    load3 = Load.get(lhs, [i, j])
    load4 = Load.get(rhs, [i, k])
    load5 = Load.get(output, [k, j])
    
    mul6 = Mulf(load4, load5)
    add7 = Addf(load3, mul6)
    inm = SSAValue.get(add7)
    store8 = Store.get(inm, output, [i, j])


    block0 = Block.from_ops([load3, load4, load5, mul6, add7, store8])
    region0 = Region.from_block_list([block0])

    
    for0 = For.from_region([], 0, 1, region0)

    import pdb;pdb.set_trace()

    printer = Printer()
    
    for1 = For.from_region([j], 0, 1, region0)
    for2 = For.from_region([k], 0, 1, region0)

    region0 = Region.from_block_list([block0])


    assert for0.attributes['lower_bound'].value == IntAttr(data=0)
    assert for0.attributes['upper_bound'].value == IntAttr(data=1)
    assert for0.attributes['step'].value == IntAttr(data=1)


