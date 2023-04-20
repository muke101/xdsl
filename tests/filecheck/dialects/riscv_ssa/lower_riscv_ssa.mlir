// RUN: xdsl-opt -p lower-riscv-ssa %s | filecheck %s

"builtin.module"() ({
    "riscv_ssa.section"() ({
        "riscv.label"() {"label" = #riscv.label<"main">} : () -> ()
        "riscv_ssa.syscall"() {"syscall_num" = 93 : i32} : () -> ()
    }) {"directive" = "text"} : () -> ()
}) : () -> ()


// CHECK: "builtin.module"() ({
// CHECK-NEXT:     "riscv.directive"() {"directive" = "text"} : () -> ()
// CHECK-NEXT:     "riscv.label"() {"label" = #riscv.label<"main">} : () -> ()
// CHECK-NEXT:     %0 = "riscv.li"() {"immediate" = 93 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:     "riscv.ecall"() : () -> ()
// CHECK-NEXT: }) : () -> ()
