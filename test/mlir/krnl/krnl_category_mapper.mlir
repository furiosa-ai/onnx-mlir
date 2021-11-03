// RUN: onnx-mlir-opt --convert-krnl-to-affine --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

  func private @test_category_mapper_string_to_int64(%arg0: memref<2x2x!krnl.string>) -> memref<2x2xi64> {
    %c0_i32 = arith.constant 0 : i32
    %c-1_i64 = arith.constant -1 : i64
    %0 = memref.alloc() {alignment = 16 : i64} : memref<2x2xi64>
    %1 = "krnl.global"() {name = "G", shape = [3], value = [1 : i32, 0 : i32, -3 : i32]} : () -> memref<3xi32>
    %2 = "krnl.global"() {name = "V", shape = [3], value = [1 : i32, 2 : i32, 0 : i32]} : () -> memref<3xi32>
    %3 = "krnl.global"() {name = "cat_int64s", shape = [3], value = [1, 2, 3]} : () -> memref<3xi64>
    %4 = "krnl.global"() {name = "cat_strings", shape = [3], value = ["cat", "dog", "cow"]} : () -> memref<3x!krnl.string>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg1 = 0 to 2, %5#1 -> %arg2 = 0 to 2) {
      %6 = krnl.load %arg0[%arg1, %arg2] : memref<2x2x!krnl.string>
      %7 = "krnl.find_index"(%6, %1, %2) : (!krnl.string, memref<3xi32>, memref<3xi32>) -> index
      %8 = krnl.load %4[%7] : memref<3x!krnl.string>
      %9 = "krnl.strlen"(%8) : (!krnl.string) -> i64
      %10 = "krnl.strncmp"(%6, %8, %9) : (!krnl.string, !krnl.string, i64) -> i32
      %11 = arith.cmpi eq, %10, %c0_i32 : i32
      scf.if %11 {
        %12 = krnl.load %3[%7] : memref<3xi64>
        krnl.store %12, %0[%arg1, %arg2] : memref<2x2xi64>
      } else {
        krnl.store %c-1_i64, %0[%arg1, %arg2] : memref<2x2xi64>
      }
    }
    return %0 : memref<2x2xi64>
  }
