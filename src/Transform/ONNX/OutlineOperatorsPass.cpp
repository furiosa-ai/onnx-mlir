/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OutlineOperatorsPass.cpp - Operator Outlining ---------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass to outline each operator for debugging purposes
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include <iostream>

using BlockListType = llvm::iplist<mlir::Block>;
using namespace mlir;

namespace {



/*!
 *  Pass that puts each operator in a separate function called from the
 *  main graph
 *  
 */
class OutlineOperatorsPass : public mlir::PassWrapper<OutlineOperatorsPass,
                               OperationPass<mlir::ModuleOp>> {
private:

public:
  OutlineOperatorsPass() {}

  std::string getOpName(Operation *op) {
    auto symbolAttr =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return std::string(symbolAttr.getValue());
    return (op->getName().getStringRef().str());
  }

class OutlinePattern : public RewritePattern {
public:
  //OutlinePattern(PatternBenefit benefit, MLIRContext *context)
  //    : RewritePattern(OnnxGemmOp::getOperationName(), benefit, context) {}

  OutlinePattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  OutlinePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), PatternBenefit(1), context) {}

};

/// Populate the pattern list.
void collectOutlinePatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<OutlinePattern>(/*benefit=*/1, ctx);
}

/// Define a custom PatternRewriter for use by the driver.
class OutlinePatternRewriter : public PatternRewriter {
public:
  OutlinePatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) {
    std::cout << "**** inside match and rewrite";
    return success();                              
  }
};

/// Apply the custom driver to `op`.
void applyOutlinePatternDriver(Operation *op,
                          RewritePatternSet &patterns) {
  // Initialize the custom PatternRewriter.
  OutlinePatternRewriter rewriter(op->getContext());
  
  std::cout << "   entered pattern driver" << std::endl;
  // Create the applicator and apply our cost model.
  PatternApplicator applicator(FrozenRewritePatternSet(std::move(patterns)));
  //applicator.applyCostModel([](const Pattern &pattern) {
  //  return pattern.getBenefit();
  //});
  applicator.applyDefaultCostModel();

  std::cout << "   built cost model " << std::endl;
  // Try to match and apply a pattern.
  LogicalResult result = applicator.matchAndRewrite(op, rewriter, [](const Pattern &pat) -> bool {
      std::cout << "matching ..." <<std::endl;
      return true;
  });

  std::cout << "   applied pattern " << std::endl;

  if (failed(result)) {
    // ... No patterns were applied.
  }
  // ... A pattern was successfully applied.
}
  static void genOutlinedFunction(PatternRewriter &rewriter,
      MLIRContext *context, std::string funcName,
      Location loc) {
    /*    
    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    llvm::SmallVector<Type, 1> outputsType{opaquePtrTy};

    auto funcType = rewriter.getFunctionType(llvm::None, outputsType);
    llvm::SmallVector<NamedAttribute, 1> attrs;
    auto funcOp = rewriter.create<FuncOp>(
        UnknownLoc::get(context), funcName, funcType, attrs);

    auto entryBlock = funcOp.addEntryBlock();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    auto sigAddr = rewriter.create<LLVM::AddressOfOp>(loc, sigvar);
    auto sigVoidPtr =
        rewriter.create<LLVM::BitcastOp>(loc, opaquePtrTy, sigAddr);
    llvm::SmallVector<Value, 1> results = {sigVoidPtr};
    rewriter.create<ReturnOp>(UnknownLoc::get(context), results);
    */

  }

  void processOp(Operation *op) {
      //auto onnxtype = ONNXOpsDialect.getTypeID();
      auto opName = getOpName(op);
      std::cout << "Operation is " << opName << std::endl;
      if (op->getDialect()->getNamespace() == "onnx") {
          std::cout << "   is an onnx op" << std::endl;
          if (opName == "onnx.Gemm") {
          std::cout << "   --- outline Gemm" << std::endl;
          }
      }
      for (Region &region : op->getRegions())
         processRegion(region);
  }

 void processRegion(Region &region) {
     std::cout << "   -- entering region" << std::endl;
     for (mlir::Block &block : region.getBlocks())
        processBlock(block);
  } 

  void processBlock(mlir::Block &block) {
    std::cout << "   -- entering block" << std::endl;
    for (mlir::Operation &op : block.getOperations())
        processOp(&op);
  }


  void runOnOperation() override {
    processOp(getOperation());
    RewritePatternSet patterns(&getContext());
    collectOutlinePatterns(patterns,&getContext());
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "       apply patterns " << std::endl;
    applyOutlinePatternDriver(getOperation(),patterns);
    //patterns.insert<OutlinePattern>(&getContext());
    //LogicalResult res =
    //    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    //assert((succeeded(res) || failed(res)) && "remove unused var warning");

   }

};
} // end anonymous namespace

/*!
 * Create an Outline Operators pass.
 */
std::unique_ptr<mlir::Pass> mlir::createOutlineOperatorsPass() {
  return std::make_unique<OutlineOperatorsPass>();
}
