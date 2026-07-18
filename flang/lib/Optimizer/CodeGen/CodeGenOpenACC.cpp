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

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace fir;

#define DEBUG_TYPE "flang-codegen-openacc"

namespace {

/// A pattern that converts the operand and result types of an OpenACC
/// operation to the LLVM dialect. The body of the region (if any) is not
/// modified and is expected to either be processed by the conversion
/// infrastructure or already contain ops compatible with LLVM dialect types.
template <typename OpType>
class OpenACCFIROpConversion : public mlir::ConvertOpToLLVMPattern<OpType> {
public:
  explicit OpenACCFIROpConversion(const fir::LLVMTypeConverter &lowering)
      : mlir::ConvertOpToLLVMPattern<OpType>(lowering) {}

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }
};

/// Generic conversion pattern for OpenACC data clause operations.
/// Converts varPtr from FIR type (e.g., !fir.ref<!fir.array<...>>) to LLVM ptr.
template <typename OpType>
struct DataClauseOpConversion : public OpenACCFIROpConversion<OpType> {
  using OpenACCFIROpConversion<OpType>::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(OpType curOp, typename OpType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = this->getTypeConverter();
    llvm::SmallVector<mlir::Type> resTypes;
    if (failed(converter->convertTypes(curOp->getResultTypes(), resTypes)))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<OpType>(
        curOp, resTypes, adaptor.getOperands(), curOp->getAttrs());
    return mlir::success();
  }
};

/// Conversion pattern for acc.enter_data operation.
/// Converts dataClauseOperands from FIR types to LLVM types.
struct EnterDataOpConversion
    : public OpenACCFIROpConversion<mlir::acc::EnterDataOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::EnterDataOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.exit_data operation.
struct ExitDataOpConversion
    : public OpenACCFIROpConversion<mlir::acc::ExitDataOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::ExitDataOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.update operation.
struct UpdateOpConversion : public OpenACCFIROpConversion<mlir::acc::UpdateOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::UpdateOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.data operation.
/// Converts dataClauseOperands from FIR types to LLVM types and handles region.
struct DataOpConversion : public OpenACCFIROpConversion<mlir::acc::DataOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::DataOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
      // Convert region block argument types
      if (!curOp.getRegion().empty()) {
        mlir::Block &block = curOp.getRegion().front();
        for (unsigned i = 0; i < block.getNumArguments(); ++i) {
          mlir::BlockArgument arg = block.getArgument(i);
          mlir::Type convertedType = this->getTypeConverter()->convertType(arg.getType());
          if (!convertedType)
            return;
          block.getArgument(i).setType(convertedType);
        }
      }
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.host_data operation.
/// Converts dataClauseOperands and region block argument types.
struct HostDataOpConversion
    : public OpenACCFIROpConversion<mlir::acc::HostDataOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::HostDataOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
      // Convert region block argument types
      if (!curOp.getRegion().empty()) {
        mlir::Block &block = curOp.getRegion().front();
        for (unsigned i = 0; i < block.getNumArguments(); ++i) {
          mlir::BlockArgument arg = block.getArgument(i);
          mlir::Type convertedType = this->getTypeConverter()->convertType(arg.getType());
          if (!convertedType)
            return;
          block.getArgument(i).setType(convertedType);
        }
      }
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.parallel operation.
/// Converts dataClauseOperands and region block argument types.
struct ParallelOpConversion
    : public OpenACCFIROpConversion<mlir::acc::ParallelOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::ParallelOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
      // Convert region block argument types
      if (!curOp.getRegion().empty()) {
        mlir::Block &block = curOp.getRegion().front();
        for (unsigned i = 0; i < block.getNumArguments(); ++i) {
          mlir::BlockArgument arg = block.getArgument(i);
          mlir::Type convertedType = this->getTypeConverter()->convertType(arg.getType());
          if (!convertedType)
            return;
          block.getArgument(i).setType(convertedType);
        }
      }
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.serial operation.
/// Converts dataClauseOperands and region block argument types.
struct SerialOpConversion
    : public OpenACCFIROpConversion<mlir::acc::SerialOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::SerialOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
      // Convert region block argument types
      if (!curOp.getRegion().empty()) {
        mlir::Block &block = curOp.getRegion().front();
        for (unsigned i = 0; i < block.getNumArguments(); ++i) {
          mlir::BlockArgument arg = block.getArgument(i);
          mlir::Type convertedType = this->getTypeConverter()->convertType(arg.getType());
          if (!convertedType)
            return;
          block.getArgument(i).setType(convertedType);
        }
      }
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.loop operation.
/// Converts dataClauseOperands and region block argument types.
struct LoopOpConversion : public OpenACCFIROpConversion<mlir::acc::LoopOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::LoopOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
      // Convert region block argument types
      if (!curOp.getRegion().empty()) {
        mlir::Block &block = curOp.getRegion().front();
        for (unsigned i = 0; i < block.getNumArguments(); ++i) {
          mlir::BlockArgument arg = block.getArgument(i);
          mlir::Type convertedType = this->getTypeConverter()->convertType(arg.getType());
          if (!convertedType)
            return;
          block.getArgument(i).setType(convertedType);
        }
      }
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.declare_enter operation.
/// Converts dataClauseOperands from FIR types to LLVM types.
struct DeclareEnterOpConversion
    : public OpenACCFIROpConversion<mlir::acc::DeclareEnterOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::DeclareEnterOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.declare_exit operation.
/// Converts dataClauseOperands from FIR types to LLVM types.
struct DeclareExitOpConversion
    : public OpenACCFIROpConversion<mlir::acc::DeclareExitOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::DeclareExitOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // adaptor.getOperands() already contains converted operands
    // (token stays as-is, dataClauseOperands converted to LLVM types)
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.init operation.
struct InitOpConversion : public OpenACCFIROpConversion<mlir::acc::InitOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::InitOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.shutdown operation.
struct ShutdownOpConversion
    : public OpenACCFIROpConversion<mlir::acc::ShutdownOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::ShutdownOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.set operation.
struct SetOpConversion : public OpenACCFIROpConversion<mlir::acc::SetOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::SetOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

/// Conversion pattern for acc.wait operation.
struct WaitOpConversion : public OpenACCFIROpConversion<mlir::acc::WaitOp> {
  using OpenACCFIROpConversion::OpenACCFIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(mlir::acc::WaitOp curOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(curOp, [&]() {
      curOp->setOperands(adaptor.getOperands());
    });
    return mlir::success();
  }
};

} // namespace

void fir::configureOpenACCToLLVMConversionLegality(
    mlir::ConversionTarget &target, const LLVMTypeConverter &typeConverter) {
  // OpenACC data clause operations are legal when their operand and result
  // types are LLVM types.
  target.addDynamicallyLegalOp<
      mlir::acc::CopyinOp, mlir::acc::CopyoutOp, mlir::acc::CreateOp,
      mlir::acc::DeleteOp, mlir::acc::PresentOp, mlir::acc::NoCreateOp,
      mlir::acc::AttachOp, mlir::acc::DetachOp, mlir::acc::DevicePtrOp,
      mlir::acc::GetDevicePtrOp, mlir::acc::UpdateDeviceOp,
      mlir::acc::UpdateHostOp, mlir::acc::UseDeviceOp,
      mlir::acc::EnterDataOp, mlir::acc::ExitDataOp, mlir::acc::DataOp,
      mlir::acc::UpdateOp, mlir::acc::HostDataOp,
      mlir::acc::TerminatorOp, mlir::acc::YieldOp,
      mlir::acc::DeclareDeviceResidentOp, mlir::acc::DeclareLinkOp,
      mlir::acc::InitOp, mlir::acc::ShutdownOp,
      mlir::acc::SetOp, mlir::acc::WaitOp>(
      [&](mlir::Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes()) &&
           llvm::all_of(op->getRegions(), [&](mlir::Region &region) {
             return typeConverter.isLegal(&region);
           });
  });

  // OpenACC compute construct operations (parallel/serial/loop) are legal when
  // their operand, result, and region types are LLVM types.
  target.addDynamicallyLegalOp<
      mlir::acc::ParallelOp, mlir::acc::SerialOp, mlir::acc::LoopOp>(
      [&](mlir::Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes()) &&
           llvm::all_of(op->getRegions(), [&](mlir::Region &region) {
             return typeConverter.isLegal(&region);
           });
  });

  // DeclareEnterOp/DeclareExitOp have acc.declare_token result/operand
  // which is an OpenACC dialect type and should remain legal.
  target.addDynamicallyLegalOp<mlir::acc::DeclareEnterOp, mlir::acc::DeclareExitOp>(
      [&](mlir::Operation *op) {
    // Check that non-token operands are LLVM types
    for (mlir::Value operand : op->getOperands()) {
      mlir::Type ty = operand.getType();
      // Skip acc.declare_token type
      if (mlir::isa<mlir::acc::DeclareTokenType>(ty))
        continue;
      if (!typeConverter.isLegal(ty))
        return false;
    }
    // Check that non-token results are LLVM types
    for (mlir::Value result : op->getResults()) {
      mlir::Type ty = result.getType();
      if (mlir::isa<mlir::acc::DeclareTokenType>(ty))
        continue;
      if (!typeConverter.isLegal(ty))
        return false;
    }
    return llvm::all_of(op->getRegions(), [&](mlir::Region &region) {
      return typeConverter.isLegal(&region);
    });
  });
}

void fir::populateOpenACCFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<DataClauseOpConversion<mlir::acc::CopyinOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::CopyoutOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::CreateOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::DeleteOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::PresentOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::NoCreateOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::AttachOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::DetachOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::DevicePtrOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::GetDevicePtrOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::UpdateDeviceOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::UpdateHostOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::UseDeviceOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::DeclareDeviceResidentOp>>(converter);
  patterns.add<DataClauseOpConversion<mlir::acc::DeclareLinkOp>>(converter);
  patterns.add<DeclareEnterOpConversion>(converter);
  patterns.add<DeclareExitOpConversion>(converter);
  patterns.add<EnterDataOpConversion>(converter);
  patterns.add<ExitDataOpConversion>(converter);
  patterns.add<UpdateOpConversion>(converter);
  patterns.add<DataOpConversion>(converter);
  patterns.add<HostDataOpConversion>(converter);
  patterns.add<InitOpConversion>(converter);
  patterns.add<ShutdownOpConversion>(converter);
  patterns.add<SetOpConversion>(converter);
  patterns.add<WaitOpConversion>(converter);
  patterns.add<ParallelOpConversion>(converter);
  patterns.add<SerialOpConversion>(converter);
  patterns.add<LoopOpConversion>(converter);
}