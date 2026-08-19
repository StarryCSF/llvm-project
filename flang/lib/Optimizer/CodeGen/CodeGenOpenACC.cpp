//===-- CodeGenOpenACC.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGenOpenACC.h"

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace fir;

#define DEBUG_TYPE "flang-codegen-openacc"

//===----------------------------------------------------------------------===//
// Dynamic legality configuration
//===----------------------------------------------------------------------===//

void mlir::configureOpenACCToLLVMConversionLegality(
    mlir::ConversionTarget &target, const LLVMTypeConverter &typeConverter) {
  // Legacy LLVM IR translator (OpenACCToLLVMIRTranslation.cpp) handles only
  // data/region operations and their helpers. Narrow legality to match.
  target.addDynamicallyLegalOp<
      mlir::acc::DataOp, mlir::acc::EnterDataOp, mlir::acc::ExitDataOp,
      mlir::acc::UpdateOp, mlir::acc::TerminatorOp, mlir::acc::CreateOp,
      mlir::acc::CopyinOp, mlir::acc::CopyoutOp, mlir::acc::DeleteOp,
      mlir::acc::UpdateDeviceOp, mlir::acc::GetDevicePtrOp,
      mlir::acc::InitOp, mlir::acc::ShutdownOp,
      mlir::acc::PresentOp, mlir::acc::OnDeviceOp,
      mlir::acc::UpdateHostOp, mlir::acc::DataBoundsOp, mlir::acc::ParallelOp,
      mlir::acc::SerialOp, mlir::acc::KernelsOp, mlir::acc::LoopOp,
      mlir::acc::AtomicUpdateOp, mlir::acc::AtomicCaptureOp,
      mlir::acc::AtomicReadOp, mlir::acc::AtomicWriteOp,
      mlir::acc::ReductionRecipeOp, mlir::acc::ReductionInitOp,
      mlir::acc::ReductionCombineOp, mlir::acc::ReductionCombineRegionOp,
      mlir::acc::ReductionAccumulateOp, mlir::acc::ReductionAccumulateArrayOp,
      mlir::acc::YieldOp, mlir::acc::FirstprivateMapInitialOp,
      mlir::acc::PrivatizeOp, mlir::acc::UnwrapPrivateOp,
      mlir::acc::PrivateLocalOp, mlir::acc::AttachOp, mlir::acc::DetachOp,
      mlir::acc::SetOp, mlir::acc::WaitOp>(
      [&](mlir::Operation *op) {
        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes()) &&
               llvm::all_of(op->getRegions(),
                            [&](mlir::Region &region) {
                              return typeConverter.isLegal(&region);
                            }) &&
               llvm::all_of(op->getAttrs(), [&](mlir::NamedAttribute attr) {
                 auto typeAttr =
                     mlir::dyn_cast<mlir::TypeAttr>(attr.getValue());
                 return !typeAttr || typeConverter.isLegal(typeAttr.getValue());
               });
      });
}

//===----------------------------------------------------------------------===//
// FIROpenACC Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Base class for OpenACC FIR conversion patterns. Provides access to the
/// FIR-specific LLVMTypeConverter.
template <typename OpType>
class FIROpenACCOpConversion : public mlir::ConvertOpToLLVMPattern<OpType> {
public:
  explicit FIROpenACCOpConversion(const fir::LLVMTypeConverter &lowering)
      : mlir::ConvertOpToLLVMPattern<OpType>(lowering) {}

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }
};

//===----------------------------------------------------------------------===//
// Generic OpenACC to LLVM Conversion Pattern
//===----------------------------------------------------------------------===//

/// Generic conversion pattern for OpenACC operations. Converts result types,
/// type attributes, inlines regions, and converts region block argument types.
/// Operand type conversions are handled by the conversion infrastructure
/// through the dynamic legality check and type converter materialization.
template <typename T>
struct OpenACCOpConversion : public FIROpenACCOpConversion<T> {
  using FIROpenACCOpConversion<T>::FIROpenACCOpConversion;

  mlir::LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const fir::LLVMTypeConverter *converter =
        static_cast<const fir::LLVMTypeConverter *>(this->getTypeConverter());

    // Convert result types.
    llvm::SmallVector<mlir::Type> resTypes;
    if (mlir::failed(converter->convertTypes(op->getResultTypes(), resTypes)))
      return mlir::failure();

    // Convert type attributes. Use convertBoxTypeAsStruct for descriptor
    // types so that varType represents the descriptor struct rather than
    // an opaque pointer (matching OpenMP MapInfoOp behavior).
    llvm::SmallVector<mlir::NamedAttribute> convertedAttrs;
    for (mlir::NamedAttribute attr : op->getAttrs()) {
      if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr.getValue())) {
        mlir::Type origType = typeAttr.getValue();
        mlir::Type convertedType;
        if (fir::isTypeWithDescriptor(origType))
          convertedType = converter->convertBoxTypeAsStruct(
              mlir::cast<fir::BaseBoxType>(fir::unwrapRefType(origType)));
        else
          convertedType = converter->convertType(origType);
        if (!convertedType)
          return mlir::failure();
        convertedAttrs.emplace_back(attr.getName(),
                                    mlir::TypeAttr::get(convertedType));
      } else {
        convertedAttrs.push_back(attr);
      }
    }

    llvm::SmallVector<mlir::Value> convertedOperands;
    convertedOperands.reserve(op->getNumOperands());
    for (auto [originalOperand, convertedOperand] :
         llvm::zip_equal(op->getOperands(), adaptor.getOperands())) {
      if (!originalOperand)
        return mlir::failure();
      convertedOperands.push_back(convertedOperand);
    }

    auto newOp = T::create(rewriter, op.getLoc(), resTypes, convertedOperands,
                           convertedAttrs);

    for (auto [originalRegion, convertedRegion] :
         llvm::zip_equal(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(originalRegion, convertedRegion,
                                  convertedRegion.end());
      if (mlir::failed(
              rewriter.convertRegionTypes(&convertedRegion, *converter)))
        return mlir::failure();
    }

    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

/// Template helper to register conversion patterns for multiple OpenACC
/// operation types at once.
template <typename... Ts>
static inline mlir::RewritePatternSet &
addOpenACCOpConversions(LLVMTypeConverter &converter,
                        mlir::RewritePatternSet &patterns) {
  return patterns.add<OpenACCOpConversion<Ts>...>(converter);
}

//===----------------------------------------------------------------------===//
// Device management operation conversions (init/shutdown/set/wait)
//===----------------------------------------------------------------------===//

/// Conversion pattern for acc.init operation.
struct InitOpConversion : public FIROpenACCOpConversion<mlir::acc::InitOp> {
  using FIROpenACCOpConversion::FIROpenACCOpConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::acc::InitOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.shutdown operation.
struct ShutdownOpConversion : public FIROpenACCOpConversion<mlir::acc::ShutdownOp> {
  using FIROpenACCOpConversion::FIROpenACCOpConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::acc::ShutdownOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.set operation.
struct SetOpConversion : public FIROpenACCOpConversion<mlir::acc::SetOp> {
  using FIROpenACCOpConversion::FIROpenACCOpConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::acc::SetOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.wait operation.
struct WaitOpConversion : public FIROpenACCOpConversion<mlir::acc::WaitOp> {
  using FIROpenACCOpConversion::FIROpenACCOpConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::acc::WaitOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Convert values yielded from OpenACC regions.
struct YieldOpConversion : public FIROpenACCOpConversion<mlir::acc::YieldOp> {
  using FIROpenACCOpConversion::FIROpenACCOpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::YieldOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(
        curOp, [&]() { curOp->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateOpenACCToLLVMConversionPatterns(
    fir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  // Keep acc.bounds result type as-is. The legacy translator does not
  // handle acc.bounds, but the type may appear in block arguments inside
  // recipe regions after recipe-materialization has been run. Retaining
  // the identity conversion prevents type conversion crashes on those
  // already-legalized paths without making acc.bounds a supported op.
  converter.addConversion(
      [&](mlir::acc::DataBoundsType type) -> mlir::Type { return type; });

  addOpenACCOpConversions<mlir::acc::DataOp, mlir::acc::EnterDataOp,
                          mlir::acc::ExitDataOp, mlir::acc::UpdateOp,
                          mlir::acc::CreateOp, mlir::acc::CopyinOp,
                          mlir::acc::CopyoutOp, mlir::acc::DeleteOp,
                          mlir::acc::PresentOp, mlir::acc::AtomicUpdateOp,
                          mlir::acc::DataBoundsOp, mlir::acc::ParallelOp,
                          mlir::acc::OnDeviceOp, mlir::acc::YieldOp,
                          mlir::acc::UpdateHostOp, mlir::acc::ReductionAccumulateOp,
                          mlir::acc::AtomicCaptureOp, mlir::acc::AtomicReadOp,
                          mlir::acc::AtomicWriteOp, mlir::acc::ReductionRecipeOp,
                          mlir::acc::ReductionInitOp, mlir::acc::ReductionCombineOp,
                          mlir::acc::ReductionCombineRegionOp, mlir::acc::KernelsOp,
                          mlir::acc::ReductionAccumulateArrayOp, mlir::acc::LoopOp,
                          mlir::acc::FirstprivateMapInitialOp, mlir::acc::PrivatizeOp,
                          mlir::acc::UnwrapPrivateOp, mlir::acc::PrivateLocalOp,
                          mlir::acc::UpdateDeviceOp, mlir::acc::GetDevicePtrOp,
                          mlir::acc::AttachOp, mlir::acc::DetachOp>(
      converter, patterns);
}

void fir::populateOpenACCFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<InitOpConversion>(converter);
  patterns.add<ShutdownOpConversion>(converter);
  patterns.add<SetOpConversion>(converter);
  patterns.add<WaitOpConversion>(converter);
  patterns.add<YieldOpConversion>(converter);
}
