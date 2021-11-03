/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlToLLVM.hpp - Lowering from KRNL+Affine+Std to LLVM -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

namespace mlir {

class MLIRContext;
class RewritePatternSet;

void checkConstantOutputs(
    ModuleOp &module, SmallVectorImpl<bool> &constantOutputs);

void populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    MLIRContext *ctx, LLVMTypeConverter &typeConverter,
    ArrayRef<bool> constantOutputs);

} // namespace mlir
