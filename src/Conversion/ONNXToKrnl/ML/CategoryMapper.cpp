/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ CategoryMapper.cpp - Lowering CategoryMapper Op ---------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CategoryMapper Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Conversion/ONNXToKrnl/PerfectHash.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using llvm::dbgs;

#define DEBUG_TYPE "category_mapper_onnx_to_krnl"

struct ONNXCategoryMapperOpLowering : public ConversionPattern {
  using PerfectHashTable = struct {
    Value G;
    Value V;
    Value len;
  };

  ONNXCategoryMapperOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXCategoryMapperOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto categoryMapperOp = cast<ONNXCategoryMapperOp>(op);
    ONNXCategoryMapperOpAdaptor operandAdaptor(operands);

    ONNXCategoryMapperOpShapeHelper shapeHelper(&categoryMapperOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    LogicalResult shapeComputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapeComputed;
    assert(succeeded(shapeComputed) && "Could not compute output shape");

    // Operands and attributes.
    Location loc = categoryMapperOp.getLoc();
    Value X = operandAdaptor.X();
    ArrayAttr cats_int64sAttr = categoryMapperOp.cats_int64sAttr();
    ArrayAttr cats_stringsAttr = categoryMapperOp.cats_stringsAttr();

    DenseElementsAttr cats_int64s = mlir::DenseElementsAttr::get(
        RankedTensorType::get(
            cats_int64sAttr.size(), rewriter.getIntegerType(64)),
        cats_int64sAttr.getValue());
    DenseElementsAttr cats_strings = mlir::DenseElementsAttr::get(
        RankedTensorType::get(
            cats_stringsAttr.size(), StringType::get(rewriter.getContext())),
        cats_stringsAttr.getValue());

    IntegerAttr default_int64 = categoryMapperOp.default_int64Attr();
    DenseElementsAttr default_string =
        (categoryMapperOp.default_stringAttr())
            ? mlir::DenseElementsAttr::get(
                  RankedTensorType::get(
                      {}, StringType::get(rewriter.getContext())),
                  categoryMapperOp.default_stringAttr().getValue())
            : nullptr;

    // Basic information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();
    ShapedType inputType = X.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    KrnlBuilder createKrnl(rewriter, op->getLoc());
    MathBuilder createMath(createKrnl);

    // Generate a perfect hash table. The hash table will be used to lookup the
    // index of the input values.
    PerfectHashTable perfectHashTable =
        createPerfectHashTable(cats_int64s, cats_strings, cats_int64sAttr,
            cats_stringsAttr, elementType, createKrnl);

    // Create loop invariant values.
    Value constantForCatsInt64s = createKrnl.constant(
        convertToMemRefType(cats_int64s.getType()), "cats_int64s", cats_int64s);

    llvm::errs() << "constantForCatsInt64s: " << constantForCatsInt64s << "\n";
    Value constantForCatsStrings =
        createKrnl.constant(convertToMemRefType(cats_strings.getType()),
            "cats_strings", cats_strings);

    Value defaultInt64 = (default_int64)
                             ? createMath.constant(rewriter.getIntegerType(64),
                                   default_int64.getSInt())
                             : nullptr;
    Value defaultString =
        (default_string)
            ? createKrnl.constant(
                  MemRefType::get({}, StringType::get(rewriter.getContext())),
                  "default_string", default_string)
            : nullptr;

    // Lookup the index in the perfect hash table corresponding to
    // each input value.
    BuildKrnlLoop inputLoops(rewriter, loc, rank);
    inputLoops.createDefineAndIterateOp(X);
    rewriter.setInsertionPointToStart(inputLoops.getIterateBlock());
    {
      // Get loop indices.
      SmallVector<IndexExpr, 4> IVs;
      for (decltype(rank) i = 0; i < rank; ++i) {
        Value iv = inputLoops.getInductionVar(i);
        IVs.emplace_back(DimIndexExpr(iv));
      }

      // Determine the index of 'inputElem' in the perfect hash table
      // 'pHash'. Note: the index might not be valid (this happens
      // when the 'inputElem' is not present in the perfect hash
      // table).
      Value inputElem = createKrnl.loadIE(X, IVs);
      Value index, isIndexValid;
      std::tie(index, isIndexValid) =
          findIndex(inputElem, elementType, perfectHashTable,
              constantForCatsInt64s, constantForCatsStrings, createKrnl);

      // Store the final result.
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(
          loc, isIndexValid, /*withElseRegion=*/true);
      storeResult(index, elementType, ifOp, constantForCatsInt64s,
          constantForCatsStrings, defaultInt64, defaultString, alloc, IVs,
          createKrnl, rewriter);
    }

    rewriter.replaceOp(op, alloc);

    LLVM_DEBUG({
      FuncOp function = getContainingFunction(op);
      assert(function && "Could not find parent function");
      dbgs() << "function: " << function << "\n";
    });

    return success();
  }

private:
  Attribute getElemAttr(ArrayAttr arr, int32_t idx) const {
    return arr.getValue()[idx];
  }

  // Generate a perfect hash table for the input dictionary.
  // Depending on the runtime type 'elementType' (the type of the element of
  // the input tensor) this function created a perfect hash table for:
  //  - cats_int64s if elementType is a int64_t
  //  - cats_strings if elementType is a string
  PerfectHashTable createPerfectHashTable(DenseElementsAttr cats_int64s,
      DenseElementsAttr cats_strings, ArrayAttr cats_int64s_ArrayAttr,
      ArrayAttr cats_strings_ArrayAttr, Type elementType,
      const KrnlBuilder &createKrnl) const {
    MathBuilder createMath(const_cast<KrnlBuilder &>(createKrnl));
    OpBuilder builder = createKrnl.getBuilder();
    PerfectHashTable res;

    // Create constants to hold the arrays 'G' and 'V'.
    auto createConstants = [&](const SmallVector<int32_t> &G,
                               const SmallVector<int32_t> &V) {
      assert(V.size() == G.size() && "V and G should have the same size");

      MemRefType type = MemRefType::get(
          {static_cast<int64_t>(V.size())}, builder.getIntegerType(32));
      res.G = createKrnl.constant(type, "G", builder.getI32VectorAttr(G));
      res.V = createKrnl.constant(type, "V", builder.getI32VectorAttr(V));
      res.len = createMath.constant(builder.getIntegerType(32), G.size());
      return res;
    };

    TypeSwitch<Type>(elementType)
        .Case<IntegerType>([&](IntegerType type) {
          // Populate the dictionary.
          assert(type.getWidth() == 64 && "Unexpected integer type");
          std::map<int64_t, int32_t> dict;
          int32_t size = cats_int64s.size();
          for (int32_t idx = 0; idx < size; ++idx) {
            Attribute elemAttr = getElemAttr(cats_int64s_ArrayAttr, idx);
            int64_t key = elemAttr.cast<IntegerAttr>().getInt();
            dict[key] = idx;
          }

          // Create the perfect hash (i.e. G and V), store them into constants.
          PerfectHash<int64_t, int32_t> pHash(dict);
          res = createConstants(pHash.getG(), pHash.getV());
        })
        .Case<StringType>([&](StringType type) {
          // Populate the dictionary.
          std::map<StringRef, int32_t> dict;
          int32_t size = cats_strings.size();
          for (int32_t idx = 0; idx < size; ++idx) {
            Attribute elemAttr = getElemAttr(cats_strings_ArrayAttr, idx);
            StringRef key = elemAttr.cast<StringAttr>().getValue();
            dict[key] = idx;
          }

          // Create the perfect hash (i.e. G and V), store them into constants.
          PerfectHash<StringRef, int32_t> pHash(dict);
          res = createConstants(pHash.getG(), pHash.getV());
        })
        .Default([&](Type type) { llvm_unreachable("Illegal KeyTy"); });

    return res;
  }

  // Determine the index of 'inputElem' in the perfect hash table 'pHash'.
  // Return the index and a true/false value depending on whether the index is
  // valid or not.
  std::tuple<Value, Value> findIndex(Value inputElem, Type elementType,
      const PerfectHashTable &pHash, Value constantForCatsInt64s,
      Value constantForCatsStrings, const KrnlBuilder &createKrnl) const {
    MathBuilder createMath(const_cast<KrnlBuilder &>(createKrnl));
    OpBuilder builder = createKrnl.getBuilder();
    Value index = createKrnl.findIndex(inputElem, pHash.G, pHash.V, pHash.len);

    std::tuple<Value, Value> res;
    TypeSwitch<Type>(elementType)
        .Case<IntegerType>([&](IntegerType type) {
          // Determine whether the index returned is valid.
          // The index is valid if 'inputElem' compares equal to the string in
          // 'constantForCatsInt64s'.
          Value compareVal = createKrnl.load(constantForCatsInt64s, {index});
          Value isIndexValid = createMath.eq(inputElem, compareVal);
          res = std::make_tuple(index, isIndexValid);
        })
        .Case<StringType>([&](StringType type) {
          // Determine whether the index returned is valid.
          // The index is valid if 'inputElem' compares equal to the string in
          // 'constantForCatsStrings'.
          Value compareVal = createKrnl.load(constantForCatsStrings, {index});
          Value strlenRes = createKrnl.strlen(compareVal);
          Value strncmpRes =
              createKrnl.strncmp(inputElem, compareVal, strlenRes);
          Value zeroVal = createMath.constant(builder.getIntegerType(32), 0);
          Value isIndexValid = createMath.eq(strncmpRes, zeroVal);
          res = std::make_tuple(index, isIndexValid);
        })
        .Default([&](Type type) { llvm_unreachable("Illegal KeyTy"); });

    return res;
  }

  // Store the result in the 'alloc' buffer.
  // Given the 'index' of the input element and an 'ifOp' operation this
  // function generates code in the 'then' and 'else' basic blocks to
  // determines whether the index is valid. If the index is valid
  void storeResult(Value index, Type elementType, scf::IfOp ifOp,
      Value constantForCatsInt64s, Value constantForCatsStrings,
      Value defaultInt64, Value defaultString, Value alloc,
      SmallVector<IndexExpr, 4> IVs, const KrnlBuilder &createKrnl,
      ConversionPatternRewriter &rewriter) const {
    TypeSwitch<Type>(elementType)
        .Case<IntegerType>([&](IntegerType type) {
          // index is valid: retrieve the value from 'cat_strings'.
          rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
          Value loadData = createKrnl.load(constantForCatsStrings, {index});
          createKrnl.storeIE(loadData, alloc, IVs);
          // index is not valid: store the default value.
          rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
          Value loadDefault = createKrnl.load(defaultString);
          createKrnl.storeIE(loadDefault, alloc, IVs);
        })
        .Case<StringType>([&](StringType type) {
          // index is valid: retrieve the value from 'cat_int64s'.
          rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
          Value loadData = createKrnl.load(constantForCatsInt64s, {index});
          createKrnl.storeIE(loadData, alloc, IVs);

          // index is not valid: store the default value.
          rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
          createKrnl.storeIE(defaultInt64, alloc, IVs);
        })
        .Default([&](Type type) { llvm_unreachable("Illegal KeyTy"); });
  }
};

void populateLoweringONNXCategoryMapperOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCategoryMapperOpLowering>(ctx);
}
