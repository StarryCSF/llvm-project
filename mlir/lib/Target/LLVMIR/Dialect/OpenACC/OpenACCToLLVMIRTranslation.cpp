//===- OpenACCToLLVMIRTranslation.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR OpenACC dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMPCommon.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace mlir;

using OpenACCIRBuilder = llvm::OpenMPIRBuilder;

//===----------------------------------------------------------------------===//
// OpenACC Runtime Function Declarations
//===----------------------------------------------------------------------===//

/// Get or create __tgt_acc_init function declaration.
static llvm::Function *getAccInitFunction(llvm::Module &module,
                                           llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  return llvm::cast<llvm::Function>(
      module.getOrInsertFunction(
          "__tgt_acc_init",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i64Ty}, false))
          .getCallee());
}

/// Get or create __tgt_acc_shutdown function declaration.
static llvm::Function *getAccShutdownFunction(llvm::Module &module,
                                               llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  return llvm::cast<llvm::Function>(
      module.getOrInsertFunction(
          "__tgt_acc_shutdown",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i64Ty}, false))
          .getCallee());
}

/// Get or create __tgt_acc_set_device_num function declaration.
static llvm::Function *getAccSetDeviceNumFunction(llvm::Module &module,
                                                    llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  return llvm::cast<llvm::Function>(
      module.getOrInsertFunction(
          "__tgt_acc_set_device_num",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i64Ty}, false))
          .getCallee());
}

/// Get or create __tgt_acc_set_device_type function declaration.
static llvm::Function *getAccSetDeviceTypeFunction(llvm::Module &module,
                                                    llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  return llvm::cast<llvm::Function>(
      module.getOrInsertFunction(
          "__tgt_acc_set_device_type",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty}, false))
          .getCallee());
}

/// Get or create __tgt_acc_set_default_async function declaration.
static llvm::Function *getAccSetDefaultAsyncFunction(llvm::Module &module,
                                                      llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  return llvm::cast<llvm::Function>(
      module.getOrInsertFunction(
          "__tgt_acc_set_default_async",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty}, false))
          .getCallee());
}

/// Get or create __tgt_acc_wait function declaration.
static llvm::Function *getAccWaitFunction(llvm::Module &module,
                                           llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module.getOrInsertFunction(
          "__tgt_acc_wait",
          llvm::FunctionType::get(llvm::Type::getInt32Ty(ctx),
              {identTy, i64Ty, i64Ty, i32Ty, i32Ty, ptrTy, i64Ty}, false))
          .getCallee());
}

/// Get or create __tgt_acc_data_enter function declaration.
static llvm::Function *getAccDataEnterFunction(llvm::Module &module,
                                               llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction("__tgt_acc_data_enter",
                               llvm::FunctionType::get(
                                   llvm::Type::getVoidTy(ctx),
                                   {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy,
                                    ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
                                   false))
          .getCallee());
}

/// Get or create __tgt_acc_declare function declaration.
static llvm::Function *getAccDeclareFunction(llvm::Module &module,
                                             llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction(
              "__tgt_acc_declare",
              llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                      {identTy, i64Ty, i64Ty, i32Ty, ptrTy,
                                       ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
                                       i64Ty, ptrTy},
                                      false))
          .getCallee());
}

/// Get or create __tgt_acc_data_exit function declaration.
static llvm::Function *getAccDataExitFunction(llvm::Module &module,
                                              llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction("__tgt_acc_data_exit",
                               llvm::FunctionType::get(
                                   llvm::Type::getVoidTy(ctx),
                                   {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy,
                                    ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
                                   false))
          .getCallee());
}

/// Get or create __tgt_acc_data_update function declaration.
static llvm::Function *getAccDataUpdateFunction(llvm::Module &module,
                                                llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction("__tgt_acc_data_update",
                               llvm::FunctionType::get(
                                   llvm::Type::getVoidTy(ctx),
                                   {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy,
                                    ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
                                   false))
          .getCallee());
}

/// Get or create __tgt_acc_data_begin function declaration.
static llvm::Function *getAccDataBeginFunction(llvm::Module &module,
                                               llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction("__tgt_acc_data_begin",
                               llvm::FunctionType::get(
                                   llvm::Type::getVoidTy(ctx),
                                   {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy,
                                    ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
                                   false))
          .getCallee());
}

/// Get or create __tgt_acc_data_end function declaration.
static llvm::Function *getAccDataEndFunction(llvm::Module &module,
                                             llvm::LLVMContext &ctx) {
  auto *identTy = llvm::PointerType::getUnqual(ctx);
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction("__tgt_acc_data_end",
                               llvm::FunctionType::get(
                                   llvm::Type::getVoidTy(ctx),
                                   {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy,
                                    ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
                                   false))
          .getCallee());
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//
/// ACC runtime specific flags from Interface.h TGT_ACC_MAPTYPE enum.
static constexpr uint64_t kAccMapTypeNone = 0x0;            // TGT_ACC_MAPTYPE_NONE
static constexpr uint64_t kAccMapTypeTo = 0x1;              // TGT_ACC_MAPTYPE_TO
static constexpr uint64_t kAccMapTypeFrom = 0x2;            // TGT_ACC_MAPTYPE_FROM
static constexpr uint64_t kAccMapTypeFinalize = 0x8;        // TGT_ACC_MAPTYPE_FINALIZE
static constexpr uint64_t kAccMapTypePtrAndObj = 0x10;      // TGT_ACC_MAPTYPE_PTR_AND_OBJ
static constexpr uint64_t kAccMapTypePrivate = 0x80;        // TGT_ACC_MAPTYPE_PRIVATE
static constexpr uint64_t kAccMapTypeLiteral = 0x100;       // TGT_ACC_MAPTYPE_LITERAL
static constexpr uint64_t kAccMapTypeDevPtr = 0x400;        // TGT_ACC_MAPTYPE_DEVPTR
static constexpr uint64_t kAccMapTypeManagedDevPtr = 0x800; // TGT_ACC_MAPTYPE_MANAGED_DEVPTR
static constexpr uint64_t kAccMapTypeNoCreate = 0x2000;     // TGT_ACC_MAPTYPE_NO_CREATE
static constexpr uint64_t kAccMapTypeGangPrivate = 0x4000;     // TGT_ACC_MAPTYPE_GANG_PRIVATE
static constexpr uint64_t kAccMapTypeWorkerPrivate = 0x8000;   // TGT_ACC_MAPTYPE_WORKER_PRIVATE
static constexpr uint64_t kAccMapTypeVectorPrivate = 0x10000;  // TGT_ACC_MAPTYPE_VECTOR_PRIVATE
static constexpr uint64_t kAccMapTypeInitZero = 0x20000;       // TGT_ACC_MAPTYPE_INIT_ZERO
static constexpr uint64_t kAccMapTypeDeviceResident = 0x40000; // TGT_ACC_MAPTYPE_DEVICE_RESIDENT
static constexpr uint64_t kAccMapTypeIfPresent = 0x80000;      // TGT_ACC_MAPTYPE_IF_PRESENT
static constexpr uint64_t kAccMapTypePresent = 0x100000;    // TGT_ACC_MAPTYPE_PRESENT

/// Default value for the device id
static constexpr int64_t kDefaultDevice = -1;

/// Create the location struct from the operation location information.
static llvm::Value *createSourceLocationInfo(OpenACCIRBuilder &builder,
                                             Operation *op) {
  auto loc = op->getLoc();
  auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  StringRef funcName = funcOp ? funcOp.getName() : "unknown";
  uint32_t strLen;
  llvm::Constant *locStr = mlir::LLVM::createSourceLocStrFromLocation(
      loc, builder, funcName, strLen);
  return builder.getOrCreateIdent(locStr, strLen);
}

/// Get or create __tgt_acc_get_deviceptr used by host_data use_device.
static llvm::Function *getAccGetDevicePtrFunction(llvm::Module &module,
                                                  llvm::LLVMContext &ctx) {
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction(
              "__tgt_acc_get_deviceptr",
              llvm::FunctionType::get(ptrTy, {ptrTy, ptrTy, i64Ty, ptrTy},
                                      false))
          .getCallee());
}

/// Get or create acc_is_present used by host_data if_present.
static llvm::Function *getAccIsPresentFunction(llvm::Module &module,
                                               llvm::LLVMContext &ctx) {
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction(
              "acc_is_present",
              llvm::FunctionType::get(i32Ty, {ptrTy}, false))
          .getCallee());
}

/// Convert acc.use_device into a lookup of the device address for its host
/// variable. The result is used by operations inside the host_data region.
static LogicalResult
convertUseDeviceOp(acc::UseDeviceOp, llvm::IRBuilderBase &,
                   LLVM::ModuleTranslation &) {
  // The result is mapped while lowering its host_data user so that the
  // host_data clauses can select between the host and device addresses.
  return success();
}

/// Convert an acc.host_data region. The region itself does not create a new
/// mapping; use_device operations retrieve pointers from the active OpenACC
/// data environment.
static LogicalResult
convertHostDataOp(acc::HostDataOp op, llvm::IRBuilderBase &builder,
                  LLVM::ModuleTranslation &moduleTranslation) {
  if (op.getRegion().empty())
    return success();

  llvm::LLVMContext &ctx = builder.getContext();
  llvm::BasicBlock *entryBlock = nullptr;
  for (Block &bb : op.getRegion()) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        ctx, "acc.host_data", builder.GetInsertBlock()->getParent());
    if (!entryBlock)
      entryBlock = llvmBB;
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
      ctx, "acc.end_host_data", builder.GetInsertBlock()->getParent());

  llvm::Value *cond = nullptr;
  if (auto ifCond = op.getIfCond()) {
    cond = moduleTranslation.lookupValue(ifCond);
    if (!cond) {
      op.emitError("could not find LLVM value for if condition");
      return failure();
    }
  }

  if (op.getIfPresent()) {
    llvm::Module *module = moduleTranslation.getLLVMModule();
    llvm::Value *allPresent = nullptr;
    for (mlir::Value dataOperand : op.getDataClauseOperands()) {
      auto useDeviceOp = dataOperand.getDefiningOp<acc::UseDeviceOp>();
      if (!useDeviceOp) {
        op.emitError("if_present requires use_device operands");
        return failure();
      }

      llvm::Value *varPtr =
          moduleTranslation.lookupValue(useDeviceOp.getVarPtr());
      if (!varPtr) {
        op.emitError("could not find LLVM value for use_device variable");
        return failure();
      }

      llvm::Value *isPresent = builder.CreateICmpNE(
          builder.CreateCall(getAccIsPresentFunction(*module, ctx), {varPtr}),
          builder.getInt32(0));
      allPresent = allPresent ? builder.CreateAnd(allPresent, isPresent)
                              : isPresent;
    }
    if (allPresent)
      cond = cond ? builder.CreateAnd(cond, allPresent) : allPresent;
  }

  // The host_data body always executes on the host. The clauses only select
  // whether the use_device operands expose device or host addresses.
  for (mlir::Value dataOperand : op.getDataClauseOperands()) {
    auto useDeviceOp = dataOperand.getDefiningOp<acc::UseDeviceOp>();
    if (!useDeviceOp) {
      op.emitError("host_data operands must be produced by use_device");
      return failure();
    }

    llvm::Value *hostPtr =
        moduleTranslation.lookupValue(useDeviceOp.getVarPtr());
    if (!hostPtr) {
      op.emitError("could not find LLVM value for use_device variable");
      return failure();
    }

    llvm::Module *module = moduleTranslation.getLLVMModule();
    llvm::Value *srcLocInfo = createSourceLocationInfo(
        *moduleTranslation.getOpenMPBuilder(), useDeviceOp);
    llvm::LLVMContext &ctx = builder.getContext();
    llvm::Value *nullPtr = llvm::ConstantPointerNull::get(
        llvm::PointerType::getUnqual(ctx));
    llvm::Value *devicePtr = builder.CreateCall(
        getAccGetDevicePtrFunction(*module, ctx),
        {srcLocInfo, nullPtr, builder.getInt64(0), hostPtr});
    llvm::Value *selectedPtr =
        cond ? builder.CreateSelect(cond, devicePtr, hostPtr) : devicePtr;
    moduleTranslation.mapValue(dataOperand, selectedPtr);
  }

  builder.CreateBr(entryBlock);

  SetVector<Block *> blocks = getBlocksSortedByDominance(op.getRegion());
  for (Block *bb : blocks) {
    if (failed(moduleTranslation.convertBlock(*bb, bb->isEntryBlock(),
                                              builder)))
      return failure();
    if (isa<acc::TerminatorOp, acc::YieldOp>(bb->getTerminator()))
      builder.CreateBr(endBlock);
  }

  LLVM::detail::connectPHINodes(op.getRegion(), moduleTranslation);
  builder.SetInsertPoint(endBlock);
  return success();
}

/// Return the runtime function used to lower the given operation.
static llvm::Function *getAssociatedFunction(Operation *op,
                                             llvm::Module &module,
                                             llvm::LLVMContext &ctx) {
  return llvm::TypeSwitch<Operation *, llvm::Function *>(op)
      .Case([&](acc::EnterDataOp) {
        return getAccDataEnterFunction(module, ctx);
      })
      .Case([&](acc::ExitDataOp) {
        return getAccDataExitFunction(module, ctx);
      });
  llvm_unreachable("Unknown OpenACC operation");
}

/// Process a data entry operation with bounds support.
/// Extracts pointer, size, and bounds information from the operation.
static LogicalResult
processDataOperandWithBounds(llvm::IRBuilderBase &builder,
                              LLVM::ModuleTranslation &moduleTranslation,
                              Operation *dataOp, mlir::Value varPtr,
                              unsigned totalNbOperand, uint64_t operandFlag,
                              SmallVector<uint64_t> &flags,
                              SmallVectorImpl<llvm::Constant *> &names,
                              unsigned &index,
                              struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::LLVMContext &ctx = builder.getContext();
  auto *i8PtrTy = llvm::PointerType::getUnqual(ctx);
  auto *arrI8PtrTy = llvm::ArrayType::get(i8PtrTy, totalNbOperand);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *arrI64Ty = llvm::ArrayType::get(i64Ty, totalNbOperand);

  llvm::Value *dataValue = moduleTranslation.lookupValue(varPtr);

  llvm::Value *dataPtrBase = dataValue;
  llvm::Value *dataPtr = dataValue;
  llvm::Value *dataSize = nullptr; // Will be set below

  // Retrieve bounds and the mapped variable type from the data entry.
  mlir::ValueRange bounds;
  mlir::Type varType;

  if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(dataOp)) {
    bounds = copyinOp.getBounds();
    varType = copyinOp.getVarType();
  } else if (auto createOp = mlir::dyn_cast_or_null<acc::CreateOp>(dataOp)) {
    bounds = createOp.getBounds();
    varType = createOp.getVarType();
  } else if (auto presentOp = mlir::dyn_cast_or_null<acc::PresentOp>(dataOp)) {
    bounds = presentOp.getBounds();
    varType = presentOp.getVarType();
  } else if (auto noCreateOp = mlir::dyn_cast_or_null<acc::NoCreateOp>(dataOp)) {
    bounds = noCreateOp.getBounds();
    varType = noCreateOp.getVarType();
  } else if (auto deviceptrOp = mlir::dyn_cast_or_null<acc::DevicePtrOp>(dataOp)) {
    bounds = deviceptrOp.getBounds();
    varType = deviceptrOp.getVarType();
  } else if (auto deviceResidentOp =
                 mlir::dyn_cast_or_null<acc::DeclareDeviceResidentOp>(dataOp)) {
    bounds = deviceResidentOp.getBounds();
    varType = deviceResidentOp.getVarType();
  } else if (auto linkOp = mlir::dyn_cast_or_null<acc::DeclareLinkOp>(dataOp)) {
    bounds = linkOp.getBounds();
    varType = linkOp.getVarType();
  } else if (auto attachOp = mlir::dyn_cast_or_null<acc::AttachOp>(dataOp)) {
    bounds = attachOp.getBounds();
    varType = attachOp.getVarType();
  }

  if (varType && isa<IntegerType, FloatType>(varType))
    operandFlag &= ~kAccMapTypePtrAndObj;

  // If bounds are present, compute the byte offset of the slice inside the
  // mapped object and the size of the slice in bytes.
  //
  // Per the OpenACC dialect documentation the bounds are zero-normalized and
  // given in rank order (rank 0 is the inner-most dimension and comes first):
  //   * a `lowerbound` of 0 means looking at data at the zero offset from the
  //     pointer,
  //   * when present, `upperbound` determines the number of mapped elements
  //     with `extent = upperbound - lowerbound + 1`; otherwise `extent`
  //     supplies that value,
  //   * the `stride` holds the distance, in units of the element or in bytes
  //     when `strideInBytes` is set, between two consecutive occurrences; for
  //     multidimensional arrays the stride of each outer dimension accounts
  //     for the complete size of all inner dimensions.
  //
  // The runtime receives either a non-null base pointer for PTR_AND_OBJ
  // mappings, or a null base pointer and the first slice element for an
  // ordinary bounded memory range. ArgSizes is the byte span from the first
  // to the last selected element:
  //     offset = sum over dims of (lowerbound * stride-in-bytes)
  //     span   = element-size-in-bytes
  //            + sum over dims of ((extent - 1) * stride-in-bytes)
  if (!bounds.empty()) {
    // A bounded array section is passed to the runtime as an ordinary memory
    // range. PTR_AND_OBJ is reserved for mappings that attach a pointer field
    // inside a parent object; using it for an array section would make the
    // runtime interpret the first bytes of the section as a pointer.
    operandFlag &= ~kAccMapTypePtrAndObj;

    // The `varType` of the data clause describes the type of the variable
    // being copied (i.e. the target pointed to by `varPtr`). Strip any array
    // wrappers to get the innermost element type and use the LLVM data layout
    // to get its byte size so element counts and strides can be turned into
    // byte offsets and spans.
    const llvm::DataLayout &dataLayout =
        moduleTranslation.getLLVMModule()->getDataLayout();
    llvm::Type *elemTy = moduleTranslation.convertType(varType);
    while (elemTy && llvm::isa<llvm::ArrayType>(elemTy))
      elemTy = llvm::cast<llvm::ArrayType>(elemTy)->getElementType();
    uint64_t elemByteSize = 1;
    if (elemTy && elemTy->isSized()) {
      llvm::TypeSize typeSize = dataLayout.getTypeAllocSize(elemTy);
      if (!typeSize.isScalable())
        elemByteSize = typeSize.getFixedValue();
    }

    llvm::Value *sliceOffset = builder.getInt64(0);
    llvm::Value *sliceSize = builder.getInt64(elemByteSize);

    for (mlir::Value boundVal : bounds) {
      auto boundsOp = boundVal.getDefiningOp<acc::DataBoundsOp>();
      if (!boundsOp) {
        dataOp->emitOpError()
            << "bounds operand is not defined by an `acc.bounds` operation";
        return failure();
      }

      // Lower bound of the dimension, defaults to 0 (bounds are zero
      // normalized).
      llvm::Value *lb = builder.getInt64(0);
      if (boundsOp.getLowerbound()) {
        llvm::Value *lbVal =
            moduleTranslation.lookupValue(boundsOp.getLowerbound());
        if (!lbVal) {
          dataOp->emitOpError("could not find LLVM value for lower bound");
          return failure();
        }
        if (!lbVal->getType()->isIntegerTy(64))
          lbVal = builder.CreateIntCast(lbVal, builder.getInt64Ty(), true);
        lb = lbVal;
      }

      // Number of elements mapped along this dimension, i.e. the extent of
      // the selected slice. This is derived from the upper bound with
      // extent = upperbound - lowerbound + 1 whenever the upper bound is
      // present: an `extent` operand supplied alongside it may carry the
      // total extent of the source array rather than the slice, so it only
      // serves as a fallback when no upper bound is available.
      llvm::Value *extent = nullptr;
      if (boundsOp.getUpperbound()) {
        llvm::Value *ubVal =
            moduleTranslation.lookupValue(boundsOp.getUpperbound());
        if (!ubVal) {
          dataOp->emitOpError("could not find LLVM value for upper bound");
          return failure();
        }
        if (!ubVal->getType()->isIntegerTy(64))
          ubVal = builder.CreateIntCast(ubVal, builder.getInt64Ty(), true);
        extent = builder.CreateAdd(builder.CreateSub(ubVal, lb),
                                   builder.getInt64(1));
      } else if (boundsOp.getExtent()) {
        extent = moduleTranslation.lookupValue(boundsOp.getExtent());
        if (!extent) {
          dataOp->emitOpError("could not find LLVM value for extent");
          return failure();
        }
      } else {
        dataOp->emitOpError()
            << "`acc.bounds` must specify an `extent` or an `upperbound`";
        return failure();
      }
      if (!extent->getType()->isIntegerTy(64))
        extent = builder.CreateIntCast(extent, builder.getInt64Ty(), true);

      // Stride between two consecutive elements, defaults to 1. Strides are
      // given in units of the element unless `strideInBytes` is set, in
      // which case they are scaled by the element size below.
      llvm::Value *stride = builder.getInt64(1);
      if (boundsOp.getStride()) {
        llvm::Value *strideVal =
            moduleTranslation.lookupValue(boundsOp.getStride());
        if (!strideVal) {
          dataOp->emitOpError("could not find LLVM value for stride");
          return failure();
        }
        if (!strideVal->getType()->isIntegerTy(64))
          strideVal = builder.CreateIntCast(strideVal, builder.getInt64Ty(),
                                            true);
        stride = strideVal;
      }
      if (!boundsOp.getStrideInBytes())
        stride = builder.CreateMul(stride, builder.getInt64(elemByteSize));

      // Accumulate the byte offset contributed by this dimension and the byte
      // span through the last selected element.
      sliceOffset =
          builder.CreateAdd(sliceOffset, builder.CreateMul(lb, stride));
      sliceSize = builder.CreateAdd(
          sliceSize, builder.CreateMul(
                         builder.CreateSub(extent, builder.getInt64(1)),
                         stride));
    }

    // Point at the first element of the slice and use its computed span.
    dataPtr = builder.CreateInBoundsGEP(llvm::Type::getInt8Ty(ctx),
                                        dataPtrBase, sliceOffset);
    dataSize = sliceSize;
  }

  // If there are no bounds, try to compute the size from the type.
  // This handles cases where Flang did not generate implicit bounds for
  // fixed-size arrays.
  if (!dataSize) {
    // First, try using MappableType interface if available.
    // Check varType first.
    if (varType && mlir::isa<mlir::acc::MappableType>(varType)) {
      std::optional<mlir::DataLayout> optDataLayout =
          mlir::acc::getDataLayout(dataOp);
      if (optDataLayout) {
        auto mappableType = mlir::cast<mlir::acc::MappableType>(varType);
        auto optSize = mappableType.getSizeInBytes(varPtr, /*accBounds=*/{},
                                                   *optDataLayout);
        if (optSize.has_value() && !optSize->isScalable()) {
          dataSize = builder.getInt64(optSize->getFixedValue());
        }
      }
    } else if (varType) {
      // Try to compute size from the MLIR type.
      // Handle LLVM array types (e.g., !llvm.array<100 x f32>).
      if (auto llvmArrayType = mlir::dyn_cast<LLVM::LLVMArrayType>(varType)) {
        uint64_t numElements = llvmArrayType.getNumElements();
        mlir::Type elemType = llvmArrayType.getElementType();

        const llvm::DataLayout &llvmDataLayout =
            moduleTranslation.getLLVMModule()->getDataLayout();
        llvm::Type *llvmElemTy = moduleTranslation.convertType(elemType);

        uint64_t elemByteSize = 1;
        if (llvmElemTy && llvmElemTy->isSized()) {
          llvm::TypeSize typeSize = llvmDataLayout.getTypeAllocSize(llvmElemTy);
          if (!typeSize.isScalable())
            elemByteSize = typeSize.getFixedValue();
        }

        dataSize = builder.getInt64(numElements * elemByteSize);
      }
      // Try MemRefType (common in MLIR).
      else if (auto memrefType = mlir::dyn_cast<MemRefType>(varType)) {
        if (memrefType.hasStaticShape()) {
          int64_t numElements = memrefType.getNumElements();

          mlir::Type elemType = memrefType.getElementType();
          const llvm::DataLayout &llvmDataLayout =
              moduleTranslation.getLLVMModule()->getDataLayout();
          llvm::Type *llvmElemTy = moduleTranslation.convertType(elemType);

          uint64_t elemByteSize = 1;
          if (llvmElemTy && llvmElemTy->isSized()) {
            llvm::TypeSize typeSize =
                llvmDataLayout.getTypeAllocSize(llvmElemTy);
            if (!typeSize.isScalable())
              elemByteSize = typeSize.getFixedValue();
          }

          dataSize = builder.getInt64(numElements * elemByteSize);
        }
      }
      // For other shaped types.
      else if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(varType)) {
        if (shapedType.hasStaticShape()) {
          int64_t numElements = shapedType.getNumElements();

          mlir::Type elemType = shapedType.getElementType();
          const llvm::DataLayout &llvmDataLayout =
              moduleTranslation.getLLVMModule()->getDataLayout();
          llvm::Type *llvmElemTy = moduleTranslation.convertType(elemType);

          uint64_t elemByteSize = 1;
          if (llvmElemTy && llvmElemTy->isSized()) {
            llvm::TypeSize typeSize =
                llvmDataLayout.getTypeAllocSize(llvmElemTy);
            if (!typeSize.isScalable())
              elemByteSize = typeSize.getFixedValue();
          }

          dataSize = builder.getInt64(numElements * elemByteSize);
        }
      }
    }

    // If varType didn't work, try varPtr's type.
    if (!dataSize) {
      mlir::Type varPtrType = varPtr.getType();
      if (mlir::isa<mlir::acc::MappableType>(varPtrType)) {
        std::optional<mlir::DataLayout> optDataLayout =
            mlir::acc::getDataLayout(dataOp);
        if (optDataLayout) {
          auto mappableType = mlir::cast<mlir::acc::MappableType>(varPtrType);
          auto optSize = mappableType.getSizeInBytes(varPtr, /*accBounds=*/{},
                                                     *optDataLayout);
          if (optSize.has_value() && !optSize->isScalable()) {
            dataSize = builder.getInt64(optSize->getFixedValue());
          }
        }
      }
    }

    // If still no size, try MemRefType or ShapedType as fallback.
    if (!dataSize && varType) {
      // Try to compute size from the MLIR type.
      // First, try MemRefType (common in MLIR).
      if (auto memrefType = mlir::dyn_cast<MemRefType>(varType)) {
        // For memref with static shape, compute total size.
        if (memrefType.hasStaticShape()) {
          // Use ShapedType's getNumElements() to compute element count.
          int64_t numElements = memrefType.getNumElements();

          // Get element type size.
          mlir::Type elemType = memrefType.getElementType();
          const llvm::DataLayout &llvmDataLayout =
              moduleTranslation.getLLVMModule()->getDataLayout();
          llvm::Type *llvmElemTy = moduleTranslation.convertType(elemType);

          uint64_t elemByteSize = 1;
          if (llvmElemTy && llvmElemTy->isSized()) {
            llvm::TypeSize typeSize =
                llvmDataLayout.getTypeAllocSize(llvmElemTy);
            if (!typeSize.isScalable())
              elemByteSize = typeSize.getFixedValue();
          }

          dataSize = builder.getInt64(numElements * elemByteSize);
        }
      } else {
        // For other types (e.g., FIR types), check if it's an array type.
        // Try to extract shape information from the type.
        if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(varType)) {
          if (shapedType.hasStaticShape()) {
            int64_t numElements = shapedType.getNumElements();

            // Get element type size.
            mlir::Type elemType = shapedType.getElementType();
            const llvm::DataLayout &llvmDataLayout =
                moduleTranslation.getLLVMModule()->getDataLayout();
            llvm::Type *llvmElemTy = moduleTranslation.convertType(elemType);

            uint64_t elemByteSize = 1;
            if (llvmElemTy && llvmElemTy->isSized()) {
              llvm::TypeSize typeSize =
                  llvmDataLayout.getTypeAllocSize(llvmElemTy);
              if (!typeSize.isScalable())
                elemByteSize = typeSize.getFixedValue();
            }

            dataSize = builder.getInt64(numElements * elemByteSize);
          }
        }
      }
    }

    // Fallback to pointer size if we couldn't compute from type.
    if (!dataSize) {
      dataSize = accBuilder->getSizeInBytes(dataValue);
    }
  }

  // Store base pointer. The runtime (accTargetDataBegin/accTargetDataEnd)
  // requires ArgBasePtr to be non-null if and only if the entry carries
  // PTR_AND_OBJ. Bounded array sections are mapped as ordinary contiguous
  // memory ranges with PTR_AND_OBJ stripped, so their base pointer is null:
  // the runtime then treats ArgPtr as both the base and the first byte of
  // the mapped range.
  llvm::Value *basePtr = (operandFlag & kAccMapTypePtrAndObj)
                             ? dataPtrBase
                             : llvm::ConstantPointerNull::get(i8PtrTy);
  llvm::Value *ptrBaseGEP = builder.CreateInBoundsGEP(
      arrI8PtrTy, mapperAllocas.ArgsBase,
      {builder.getInt32(0), builder.getInt32(index)});
  builder.CreateStore(basePtr, ptrBaseGEP);

  // Store pointer
  llvm::Value *ptrGEP = builder.CreateInBoundsGEP(
      arrI8PtrTy, mapperAllocas.Args,
      {builder.getInt32(0), builder.getInt32(index)});
  builder.CreateStore(dataPtr, ptrGEP);

  // Store size
  llvm::Value *sizeGEP = builder.CreateInBoundsGEP(
      arrI64Ty, mapperAllocas.ArgSizes,
      {builder.getInt32(0), builder.getInt32(index)});
  builder.CreateStore(dataSize, sizeGEP);

  flags.push_back(operandFlag);
  llvm::Constant *mapName =
      mlir::LLVM::createMappingInformation(varPtr.getLoc(), *accBuilder);
  names.push_back(mapName);
  ++index;

  return success();
}

/// Extract pointer, size and mapping information from operands
/// to populate the future functions arguments.
static LogicalResult
processOperands(llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation, Operation *op,
                ValueRange operands, unsigned totalNbOperand,
                uint64_t operandFlag, SmallVector<uint64_t> &flags,
                SmallVectorImpl<llvm::Constant *> &names, unsigned &index,
                struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::LLVMContext &ctx = builder.getContext();
  auto *i8PtrTy = llvm::PointerType::getUnqual(ctx);
  auto *arrI8PtrTy = llvm::ArrayType::get(i8PtrTy, totalNbOperand);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  auto *arrI64Ty = llvm::ArrayType::get(i64Ty, totalNbOperand);

  for (Value data : operands) {
    llvm::Value *dataValue = moduleTranslation.lookupValue(data);

    llvm::Value *dataPtrBase;
    llvm::Value *dataPtr;
    llvm::Value *dataSize;

    if (isa<LLVM::LLVMPointerType>(data.getType())) {
      dataPtrBase = dataValue;
      dataPtr = dataValue;
      dataSize = accBuilder->getSizeInBytes(dataValue);
    } else {
      return op->emitOpError()
             << "Data operand must be legalized before translation."
             << "Unsupported type: " << data.getType();
    }

    // Store base pointer extracted from operand into the i-th position of
    // argBase.
    llvm::Value *ptrBaseGEP = builder.CreateInBoundsGEP(
        arrI8PtrTy, mapperAllocas.ArgsBase,
        {builder.getInt32(0), builder.getInt32(index)});
    builder.CreateStore(dataPtrBase, ptrBaseGEP);

    // Store pointer extracted from operand into the i-th position of args.
    llvm::Value *ptrGEP = builder.CreateInBoundsGEP(
        arrI8PtrTy, mapperAllocas.Args,
        {builder.getInt32(0), builder.getInt32(index)});
    builder.CreateStore(dataPtr, ptrGEP);

    // Store size extracted from operand into the i-th position of argSizes.
    llvm::Value *sizeGEP = builder.CreateInBoundsGEP(
        arrI64Ty, mapperAllocas.ArgSizes,
        {builder.getInt32(0), builder.getInt32(index)});
    builder.CreateStore(dataSize, sizeGEP);

    flags.push_back(operandFlag);
    llvm::Constant *mapName =
        mlir::LLVM::createMappingInformation(data.getLoc(), *accBuilder);
    names.push_back(mapName);
    ++index;
  }
  return success();
}

/// Return the first element of a bounded data entry so later operations use
/// the same host mapping key as the data-begin call.
static llvm::Value *
getBoundedMappingPointer(Operation *entry, llvm::Value *dataValue,
                         LLVM::ModuleTranslation &moduleTranslation,
                         llvm::IRBuilderBase &builder) {
  ValueRange bounds = acc::getBounds(entry);
  if (bounds.empty())
    return dataValue;

  mlir::Type varType;
  TypeSwitch<Operation *>(entry)
      .Case<acc::CopyinOp, acc::CreateOp, acc::CopyoutOp, acc::PresentOp,
            acc::NoCreateOp, acc::DevicePtrOp, acc::AttachOp,
            acc::GetDevicePtrOp>(
          [&](auto op) { varType = op.getVarType(); });
  llvm::Type *elementType =
      varType ? moduleTranslation.convertType(varType) : nullptr;
  if (!elementType)
    return dataValue;
  while (auto *arrayType = llvm::dyn_cast<llvm::ArrayType>(elementType))
    elementType = arrayType->getElementType();
  if (!elementType->isSized())
    return dataValue;

  llvm::TypeSize elementSize =
      moduleTranslation.getLLVMModule()->getDataLayout().getTypeAllocSize(
          elementType);
  if (elementSize.isScalable())
    return dataValue;

  llvm::Value *offset = builder.getInt64(0);
  auto getBoundValue = [&](Value value, int64_t defaultValue) -> llvm::Value * {
    if (!value)
      return builder.getInt64(defaultValue);
    llvm::Value *result = moduleTranslation.lookupValue(value);
    if (!result)
      return builder.getInt64(defaultValue);
    if (!result->getType()->isIntegerTy(64))
      result = builder.CreateIntCast(result, builder.getInt64Ty(), true);
    return result;
  };
  for (Value bound : bounds) {
    auto boundsOp = bound.getDefiningOp<acc::DataBoundsOp>();
    if (!boundsOp)
      return dataValue;
    llvm::Value *lower = getBoundValue(boundsOp.getLowerbound(), 0);
    llvm::Value *stride = getBoundValue(boundsOp.getStride(), 1);
    if (!boundsOp.getStrideInBytes())
      stride = builder.CreateMul(stride,
                                 builder.getInt64(elementSize.getFixedValue()));
    offset = builder.CreateAdd(offset, builder.CreateMul(lower, stride));
  }
  return builder.CreateGEP(builder.getInt8Ty(), dataValue, {offset});
}

/// Process data operands from acc::EnterDataOp
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::EnterDataOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  unsigned index = 0;

  // Create operands are handled as `alloc` call.
  // Copyin operands are handled as `to` call.
  // Attach operands are handled as `attach` call.
  // Create_zero operands are handled as `alloc` call with zero initialization.
  llvm::SmallVector<mlir::Value> create, copyin, attachOperands,
      createZeroOperands;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto createOp = dataOp.getDefiningOp<acc::CreateOp>()) {
      if (createOp.isCreateZero()) {
        createZeroOperands.push_back(createOp.getVarPtr());
      } else {
        create.push_back(createOp.getVarPtr());
      }
    } else if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(
                   dataOp.getDefiningOp())) {
      copyin.push_back(copyinOp.getVarPtr());
    } else if (auto attachOp = mlir::dyn_cast_or_null<acc::AttachOp>(
                   dataOp.getDefiningOp())) {
      attachOperands.push_back(attachOp.getVarPtr());
    }
  }

  auto nbTotalOperands = create.size() + copyin.size() +
                         attachOperands.size() + createZeroOperands.size();

  // Create operands are handled as `alloc` call with PTR_AND_OBJ.
  if (failed(processOperands(builder, moduleTranslation, op, create,
                             nbTotalOperands,
                             kAccMapTypeNone | kAccMapTypePtrAndObj, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Copyin operands are handled as `to` call with PTR_AND_OBJ.
  if (failed(processOperands(builder, moduleTranslation, op, copyin,
                             nbTotalOperands,
                             kAccMapTypeTo | kAccMapTypePtrAndObj, flags, names,
                             index, mapperAllocas)))
    return failure();

  // Attach operands are handled as `attach` call (PTR_AND_OBJ only).
  if (failed(processOperands(builder, moduleTranslation, op, attachOperands,
                             nbTotalOperands, kAccMapTypePtrAndObj, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Create_zero operands are handled as `alloc` call with PTR_AND_OBJ.
  if (failed(processOperands(
          builder, moduleTranslation, op, createZeroOperands, nbTotalOperands,
          kAccMapTypeNone | kAccMapTypePtrAndObj | kAccMapTypeInitZero, flags,
          names, index, mapperAllocas)))
    return failure();

  return success();
}

/// Process data operands from acc::ExitDataOp
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::ExitDataOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  unsigned index = 0;

  llvm::SmallVector<mlir::Value> deleteOperands, detachOperands;
  // Keep track of copyout operations to check for zero modifier.
  llvm::SmallVector<std::pair<mlir::Value, acc::CopyoutOp>> copyoutOperands;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(
            dataOp.getDefiningOp())) {
      for (auto &u : devicePtrOp.getAccPtr().getUses()) {
        if (auto deleteOp = mlir::dyn_cast_or_null<acc::DeleteOp>(u.getOwner()))
          deleteOperands.push_back(devicePtrOp.getVarPtr());
        else if (auto copyoutOp = mlir::dyn_cast_or_null<acc::CopyoutOp>(u.getOwner()))
          copyoutOperands.emplace_back(devicePtrOp.getVarPtr(), copyoutOp);
        else if (mlir::dyn_cast_or_null<acc::DetachOp>(u.getOwner()))
          detachOperands.push_back(devicePtrOp.getVarPtr());
      }
    }
  }

  auto nbTotalOperands = deleteOperands.size() + copyoutOperands.size() +
                         detachOperands.size();
  uint64_t finalizeFlag =
      op.getFinalize() ? kAccMapTypeFinalize : kAccMapTypeNone;

  // Delete operands are handled as `delete` call, with the finalize flag
  // if the exit_data operation has the finalize clause.
  if (failed(processOperands(builder, moduleTranslation, op, deleteOperands,
                             nbTotalOperands,
                             finalizeFlag | kAccMapTypePtrAndObj, flags, names,
                             index, mapperAllocas)))
    return failure();

  // Copyout operands are handled as `from` call.
  // Add kAccMapTypeInitZero for copyout(zero) modifier.
  for (auto &[varPtr, copyoutOp] : copyoutOperands) {
    uint64_t copyoutFlag = kAccMapTypeFrom | finalizeFlag | kAccMapTypePtrAndObj;
    if (copyoutOp.isCopyoutZero())
      copyoutFlag |= kAccMapTypeInitZero;
    if (failed(processOperands(builder, moduleTranslation, op,
                               mlir::ValueRange(varPtr),
                               nbTotalOperands, copyoutFlag, flags, names,
                               index, mapperAllocas)))
      return failure();
  }

  // Detach operands are handled as `detach` call (FINALIZE | PTR_AND_OBJ).
  if (failed(processOperands(builder, moduleTranslation, op, detachOperands,
                             nbTotalOperands,
                             finalizeFlag | kAccMapTypePtrAndObj,
                             flags, names, index, mapperAllocas)))
    return failure();

  return success();
}

/// Process data operands from acc::UpdateOp.
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::UpdateOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  unsigned index = 0;
  llvm::SmallVector<mlir::Value> from, to;

  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto getDevicePtrOp = dataOp.getDefiningOp<acc::GetDevicePtrOp>())
      from.push_back(getDevicePtrOp.getVarPtr());
    else if (auto updateHostOp = dataOp.getDefiningOp<acc::UpdateHostOp>())
      from.push_back(updateHostOp.getVarPtr());
    else if (auto updateDeviceOp =
                 dataOp.getDefiningOp<acc::UpdateDeviceOp>())
      to.push_back(updateDeviceOp.getVarPtr());
    else
      return op.emitError()
             << "expected update data operand to be produced by "
                "acc.getdeviceptr, acc.update_host, or acc.update_device";
  }

  unsigned totalNbOperand = from.size() + to.size();
  // The runtime only skips the update of arguments that are not present on
  // the device when the if_present maptype bit is set; without it, it
  // terminates.
  uint64_t flag = kAccMapTypePtrAndObj;
  if (op.getIfPresent())
    flag |= kAccMapTypeIfPresent;
  if (failed(processOperands(
          builder, moduleTranslation, op, from, totalNbOperand,
          flag | kAccMapTypeFrom, flags, names, index, mapperAllocas)))
    return failure();
  if (failed(processOperands(
          builder, moduleTranslation, op, to, totalNbOperand,
          flag | kAccMapTypeTo, flags, names, index, mapperAllocas)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Conversion functions
//===----------------------------------------------------------------------===//

/// Converts an OpenACC data operation into LLVM IR.
static LogicalResult convertDataOp(acc::DataOp &op,
                                   llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::LLVMContext &ctx = builder.getContext();
  auto enclosingFuncOp = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());

  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::Value *srcLocInfo = createSourceLocationInfo(*accBuilder, op);

  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::Function *beginMapperFunc = getAccDataBeginFunction(*module, ctx);
  llvm::Function *endMapperFunc = getAccDataEndFunction(*module, ctx);

  // Number of arguments in the data operation.
  unsigned totalNbOperand = op.getNumDataOperands();

  struct OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, totalNbOperand,
                                  mapperAllocas);

  SmallVector<uint64_t> flags;
  SmallVector<llvm::Constant *> names;
  unsigned index = 0;

  // Keep each data entry paired with its variable pointer for bounds lowering.
  struct DataOpInfo {
    Operation *op;
    mlir::Value varPtr;
  };

  llvm::SmallVector<DataOpInfo> copyinOps, copyoutOps, createOps, presentOps,
      deleteOps, copyOps, noCreateOps, devicePtrOps, attachOps;

  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (mlir::isa<acc::GetDevicePtrOp>(dataOp.getDefiningOp())) {
      return op.emitError()
             << "acc.getdeviceptr is not a data entry for acc.data; "
                "use acc.deviceptr for a deviceptr data clause";
    } else if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(
                   dataOp.getDefiningOp())) {
      if (copyinOp.getDataClause() == acc::DataClause::acc_copy)
        copyOps.push_back({copyinOp, copyinOp.getVarPtr()});
      else
        copyinOps.push_back({copyinOp, copyinOp.getVarPtr()});
    } else if (auto createOp = mlir::dyn_cast_or_null<acc::CreateOp>(
                   dataOp.getDefiningOp())) {
      if (createOp.getDataClause() == acc::DataClause::acc_copyout ||
          createOp.getDataClause() == acc::DataClause::acc_copyout_zero) {
        copyoutOps.push_back({createOp, createOp.getVarPtr()});
      } else {
        createOps.push_back({createOp, createOp.getVarPtr()});
      }
    } else if (auto presentOp = mlir::dyn_cast_or_null<acc::PresentOp>(
                   dataOp.getDefiningOp())) {
      presentOps.push_back({presentOp, presentOp.getVarPtr()});
    } else if (auto noCreateOp = mlir::dyn_cast_or_null<acc::NoCreateOp>(
                   dataOp.getDefiningOp())) {
      noCreateOps.push_back({noCreateOp, noCreateOp.getVarPtr()});
    } else if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::DevicePtrOp>(
                   dataOp.getDefiningOp())) {
      devicePtrOps.push_back({devicePtrOp, devicePtrOp.getVarPtr()});
    } else if (auto attachOp = mlir::dyn_cast_or_null<acc::AttachOp>(
                   dataOp.getDefiningOp())) {
      attachOps.push_back({attachOp, attachOp.getVarPtr()});
    } else {
      return op.emitError() << "unsupported acc.data operand: "
                            << dataOp.getDefiningOp()->getName();
    }
  }

  // Process each data operation with bounds support
  for (auto &info : copyinOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypeTo | kAccMapTypePtrAndObj, flags, names, index,
            mapperAllocas)))
      return failure();
  }

  for (auto &info : deleteOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypeFinalize, flags, names, index, mapperAllocas)))
      return failure();
  }

  for (auto &info : copyoutOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypePtrAndObj, flags, names, index, mapperAllocas)))
      return failure();
  }

  for (auto &info : createOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypePtrAndObj, flags, names, index, mapperAllocas)))
      return failure();
  }

  for (auto &info : presentOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypePresent | kAccMapTypeNoCreate, flags, names, index,
            mapperAllocas)))
      return failure();
  }

  for (auto &info : copyOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypeTo | kAccMapTypePtrAndObj, flags, names, index,
            mapperAllocas)))
      return failure();
  }

  for (auto &info : noCreateOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypeNoCreate | kAccMapTypePtrAndObj, flags, names, index,
            mapperAllocas)))
      return failure();
  }

  for (auto &info : devicePtrOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypeDevPtr | kAccMapTypePtrAndObj, flags, names, index,
            mapperAllocas)))
      return failure();
  }

  for (auto &info : attachOps) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, info.op, info.varPtr, totalNbOperand,
            kAccMapTypePtrAndObj, flags, names, index, mapperAllocas)))
      return failure();
  }

  assert(index == totalNbOperand &&
         "all acc.data operands must be classified and processed");

  // Generate end flags for data_end call
  // According to OpenACC MLIR dialect decomposition:
  // - copyin: acc.copyin (entry) -> acc.delete (exit)
  // - copy: acc.copyin (entry) -> acc.copyout (exit)
  // - copyout: acc.create (entry) -> acc.copyout (exit)
  // - create: acc.create (entry) -> acc.delete (exit)
  // - present: acc.present (entry) -> acc.delete (exit)
  // - no_create: acc.nocreate (entry) -> acc.delete (exit)
  // - attach: acc.attach (entry) -> acc.detach (exit)
  SmallVector<uint64_t> endFlags;

  // copyin: entry TO|PTR_AND_OBJ, exit FINALIZE (acc.delete)
  for (size_t i = 0; i < copyinOps.size(); ++i)
    endFlags.push_back(kAccMapTypeFinalize);

  // delete: entry FINALIZE, exit FINALIZE
  for (size_t i = 0; i < deleteOps.size(); ++i)
    endFlags.push_back(kAccMapTypeFinalize);

  // copyout: entry PTR_AND_OBJ, exit FROM|PTR_AND_OBJ (acc.copyout).
  // Bounded sections strip PTR_AND_OBJ so ArgBasePtr stays null at the end.
  // copyout(zero) adds kAccMapTypeInitZero to zero the device memory after copy.
  // Scalar types also strip PTR_AND_OBJ (handled in processDataOperandWithBounds).
  for (auto &info : copyoutOps) {
    uint64_t endFlag = kAccMapTypeFrom | kAccMapTypePtrAndObj;
    // copyoutOps stores CreateOp (entry), check for zero modifier
    if (auto createOp = mlir::dyn_cast_or_null<acc::CreateOp>(info.op)) {
      if (createOp.isCreateZero())
        endFlag |= kAccMapTypeInitZero;
      // Strip PTR_AND_OBJ for scalar types (matching entry behavior)
      if (createOp.getVarType() && isa<IntegerType, FloatType>(createOp.getVarType()))
        endFlag &= ~kAccMapTypePtrAndObj;
    }
    if (!acc::getBounds(info.op).empty())
      endFlag &= ~kAccMapTypePtrAndObj;
    endFlags.push_back(endFlag);
  }

  // create: entry PTR_AND_OBJ, exit FINALIZE (acc.delete)
  for (size_t i = 0; i < createOps.size(); ++i)
    endFlags.push_back(kAccMapTypeFinalize);

  // present: entry PRESENT|NO_CREATE, exit PRESENT|NO_CREATE.
  for (size_t i = 0; i < presentOps.size(); ++i)
    endFlags.push_back(kAccMapTypePresent | kAccMapTypeNoCreate);

  // copy: entry TO|PTR_AND_OBJ, exit FROM|PTR_AND_OBJ (acc.copyout).
  // Bounded sections strip PTR_AND_OBJ so ArgBasePtr stays null at the end.
  // copy(zero) adds kAccMapTypeInitZero to zero the device memory after copy.
  for (auto &info : copyOps) {
    uint64_t endFlag = kAccMapTypeFrom | kAccMapTypePtrAndObj;
    // Check if original clause was copy with zero modifier (stored in copyinOp)
    if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(info.op))
      if (copyinOp.getModifiersAttr() &&
          acc::bitEnumContainsAny(copyinOp.getModifiers(),
                                  acc::DataClauseModifier::zero))
        endFlag |= kAccMapTypeInitZero;
    if (!acc::getBounds(info.op).empty())
      endFlag &= ~kAccMapTypePtrAndObj;
    endFlags.push_back(endFlag);
  }

  // no_create: entry NO_CREATE|PTR_AND_OBJ, exit FINALIZE (acc.delete)
  for (size_t i = 0; i < noCreateOps.size(); ++i)
    endFlags.push_back(kAccMapTypeFinalize);

  // deviceptr: entry DEVPTR|PTR_AND_OBJ, no exit operation
  for (size_t i = 0; i < devicePtrOps.size(); ++i)
    endFlags.push_back(kAccMapTypeDevPtr | kAccMapTypePtrAndObj);

  // attach: entry PTR_AND_OBJ, exit FINALIZE|PTR_AND_OBJ (acc.detach).
  // Bounded sections strip PTR_AND_OBJ so ArgBasePtr stays null at the end.
  for (auto &info : attachOps) {
    uint64_t endFlag = kAccMapTypeFinalize | kAccMapTypePtrAndObj;
    if (!acc::getBounds(info.op).empty())
      endFlag &= ~kAccMapTypePtrAndObj;
    endFlags.push_back(endFlag);
  }

  llvm::GlobalVariable *maptypes =
      accBuilder->createOffloadMaptypes(flags, ".offload_maptypes");
  llvm::Value *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      maptypes, /*Idx0=*/0, /*Idx1=*/0);

  llvm::GlobalVariable *endMaptypes =
      accBuilder->createOffloadMaptypes(endFlags, ".offload_maptypes_end");
  llvm::Value *endMaptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      endMaptypes, /*Idx0=*/0, /*Idx1=*/0);

  llvm::GlobalVariable *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames");
  llvm::Value *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::PointerType::getUnqual(ctx), totalNbOperand),
      mapnames, /*Idx0=*/0, /*Idx1=*/0);

  // Prepare arguments for ACC runtime calls.
  auto *nullPtr = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));

  // The data construct defaults to acc_device_default.
  llvm::Value *deviceType = builder.getInt64(1);

  // TODO: Lower the default clause into the runtime flags parameter.
  llvm::Value *flagsVal = builder.getInt64(0);

  // Handle async clause.
  // -1 = sync, -2 = async-only, >= 0 = async queue id
  llvm::Value *asyncVal = builder.getInt64(-1);
  if (op.getAsyncValue()) {
    llvm::Value *asyncValue = moduleTranslation.lookupValue(op.getAsyncValue());
    if (asyncValue) {
      if (!asyncValue->getType()->isIntegerTy(64))
        asyncValue = builder.CreateIntCast(asyncValue, builder.getInt64Ty(), true);
      asyncVal = asyncValue;
    }
  } else if (op.hasAsyncOnly()) {
    asyncVal = builder.getInt64(-2);
  }

  // Emit a wait call before data_begin when the clause is present.
  if (!op.getWaitValues().empty() || op.hasWaitOnly()) {
    uint32_t waitNum = op.getWaitValues().size();
    llvm::Value *waitList = nullPtr;
    if (waitNum != 0) {
      auto *arrayType = llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), waitNum);
      waitList = builder.CreateAlloca(arrayType);
      for (uint32_t i = 0; i < waitNum; ++i) {
        llvm::Value *waitValue =
            moduleTranslation.lookupValue(op.getWaitValues()[i]);
        if (!waitValue) {
          op.emitError("could not find LLVM value for wait operand") << i;
          return failure();
        }
        if (waitValue->getType() != llvm::Type::getInt64Ty(ctx))
          waitValue = builder.CreateIntCast(waitValue, llvm::Type::getInt64Ty(ctx), true);
        llvm::Value *element = builder.CreateInBoundsGEP(
            arrayType, waitList, {builder.getInt32(0), builder.getInt32(i)});
        builder.CreateStore(waitValue, element);
      }
    }
    llvm::Value *waitDevnum = builder.getInt32(-1);
    if (op.getWaitDevnum()) {
      llvm::Value *devnum = moduleTranslation.lookupValue(op.getWaitDevnum());
      if (devnum) {
        if (devnum->getType() != llvm::Type::getInt32Ty(ctx))
          devnum = builder.CreateIntCast(devnum, llvm::Type::getInt32Ty(ctx), true);
        waitDevnum = devnum;
      }
    }
    builder.CreateCall(getAccWaitFunction(*module, ctx),
        {srcLocInfo, builder.getInt64(0), deviceType, waitDevnum,
         builder.getInt32(waitNum), waitList, asyncVal});
  }

  // An if-clause data_begin belongs in the taken entry block.
  if (!op.getIfCond()) {
    // No if clause: call directly.
    builder.CreateCall(beginMapperFunc,
        {srcLocInfo, flagsVal, deviceType, builder.getInt32(totalNbOperand),
         mapperAllocas.ArgsBase, mapperAllocas.Args, mapperAllocas.ArgSizes,
         maptypesArg, mapnamesArg, nullPtr, nullPtr, asyncVal});
  }

  // Materialize the optional if condition.
  llvm::Value *cond = nullptr;
  if (op.getIfCond()) {
    cond = moduleTranslation.lookupValue(op.getIfCond());
    if (!cond) {
      op.emitError("could not find LLVM value for if condition");
      return failure();
    }
  }

  // Convert the region.
  llvm::BasicBlock *entryBlock = nullptr;

  for (Block &bb : op.getRegion()) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        ctx, "acc.data", builder.GetInsertBlock()->getParent());
    if (entryBlock == nullptr)
      entryBlock = llvmBB;
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  auto afterDataRegion = builder.saveIP();

  llvm::BasicBlock *endBlock = nullptr;
  llvm::Instruction *sourceTerminator;

  if (cond) {
    // Create a block for skipping the data region.
    endBlock = llvm::BasicBlock::Create(ctx, "acc.data.skip",
                                        builder.GetInsertBlock()->getParent());
    // Create a conditional branch.
    sourceTerminator = builder.CreateCondBr(cond, entryBlock, endBlock);
  } else {
    // Create an unconditional branch.
    sourceTerminator = builder.CreateBr(entryBlock);
  }

  // Emit data_begin inside the entry block for an if clause.
  if (op.getIfCond()) {
    // Set the insertion point to the entry block.
    builder.SetInsertPoint(entryBlock, entryBlock->getFirstInsertionPt());
    // Call data_begin.
    builder.CreateCall(
        beginMapperFunc,
        {srcLocInfo, flagsVal, deviceType, builder.getInt32(totalNbOperand),
         mapperAllocas.ArgsBase, mapperAllocas.Args, mapperAllocas.ArgSizes,
         maptypesArg, mapnamesArg, nullPtr, nullPtr, asyncVal});
  }

  builder.restoreIP(afterDataRegion);
  llvm::BasicBlock *endDataBlock = llvm::BasicBlock::Create(
      ctx, "acc.end_data", builder.GetInsertBlock()->getParent());

  SetVector<Block *> blocks = getBlocksSortedByDominance(op.getRegion());
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    if (bb->isEntryBlock())
      sourceTerminator->setSuccessor(0, llvmBB);

    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder))) {
      return failure();
    }

    if (isa<acc::TerminatorOp, acc::YieldOp>(bb->getTerminator()))
      builder.CreateBr(endDataBlock);
  }

  // Create call to end the data region.
  builder.SetInsertPoint(endDataBlock);
  builder.CreateCall(endMapperFunc,
      {srcLocInfo, flagsVal, deviceType, builder.getInt32(totalNbOperand),
       mapperAllocas.ArgsBase, mapperAllocas.Args, mapperAllocas.ArgSizes,
       endMaptypesArg, mapnamesArg, nullPtr, nullPtr, asyncVal});

  // Merge the data and skip paths after an if clause.
  if (endBlock) {
    // Create a continuation block after both paths converge.
    llvm::BasicBlock *continueBlock = llvm::BasicBlock::Create(
        ctx, "acc.data.continue", endDataBlock->getParent());

    // End-of-data path jumps to the continuation.
    builder.SetInsertPoint(endDataBlock);
    builder.CreateBr(continueBlock);

    // Skip path jumps to the continuation.
    builder.SetInsertPoint(endBlock);
    builder.CreateBr(continueBlock);

    // Continue subsequent code generation from the merged block.
    builder.SetInsertPoint(continueBlock);
  }

  return success();
}

/// Converts an OpenACC standalone data operation into LLVM IR.
template <typename OpTy>
static LogicalResult
convertStandaloneDataOp(OpTy &op, llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  auto enclosingFuncOp =
      op.getOperation()->template getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());

  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();

  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();
  auto *mapperFunc = getAssociatedFunction(op, *module, ctx);

  // Number of arguments in the enter_data operation.
  unsigned totalNbOperand = op.getNumDataOperands();

  struct OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, totalNbOperand,
                                  mapperAllocas);

  SmallVector<uint64_t> flags;
  SmallVector<llvm::Constant *> names;

  if (failed(processDataOperands(builder, moduleTranslation, op, flags, names,
                                 mapperAllocas)))
    return failure();

  llvm::GlobalVariable *maptypes =
      accBuilder->createOffloadMaptypes(flags, ".offload_maptypes");
  llvm::Value *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      maptypes, /*Idx0=*/0, /*Idx1=*/0);

  llvm::GlobalVariable *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames");
  llvm::Value *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::PointerType::getUnqual(ctx), totalNbOperand),
      mapnames, /*Idx0=*/0, /*Idx1=*/0);

  auto *nullPtr =
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));
  auto *flagsVal = builder.getInt64(0);
  auto *deviceTypeVal = builder.getInt64(1);
  auto *argNumVal = builder.getInt32(totalNbOperand);

  // Get async value based on operation type
  llvm::Value *asyncVal = builder.getInt64(-1);
  if constexpr (std::is_same_v<OpTy, acc::EnterDataOp> ||
                std::is_same_v<OpTy, acc::ExitDataOp>) {
    if (op.getAsyncOperand()) {
      llvm::Value *asyncValue =
          moduleTranslation.lookupValue(op.getAsyncOperand());
      if (asyncValue) {
        if (!asyncValue->getType()->isIntegerTy(64))
          asyncValue =
              builder.CreateIntCast(asyncValue, builder.getInt64Ty(), true);
        asyncVal = asyncValue;
      }
    } else if (op.getAsync()) {
      asyncVal = builder.getInt64(-2);
    }
  }

  // Emit wait call if needed (for EnterDataOp and ExitDataOp)
  if constexpr (std::is_same_v<OpTy, acc::EnterDataOp> ||
                std::is_same_v<OpTy, acc::ExitDataOp>) {
    if (!op.getWaitOperands().empty() || op.getWait()) {
      uint32_t waitNum = op.getWaitOperands().size();
      llvm::Value *waitList = nullPtr;
      if (waitNum != 0) {
        auto *arrayType =
            llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), waitNum);
        waitList = builder.CreateAlloca(arrayType);
        for (auto [index, waitOperand] :
             llvm::enumerate(op.getWaitOperands())) {
          llvm::Value *waitValue = moduleTranslation.lookupValue(waitOperand);
          if (!waitValue) {
            op.emitError("could not find LLVM value for wait operand") << index;
            return failure();
          }
          if (!waitValue->getType()->isIntegerTy(64))
            waitValue =
                builder.CreateIntCast(waitValue, builder.getInt64Ty(), true);
          llvm::Value *element = builder.CreateInBoundsGEP(
              arrayType, waitList,
              {builder.getInt32(0), builder.getInt32(index)});
          builder.CreateStore(waitValue, element);
        }
      }
      llvm::Value *deviceNum = builder.getInt32(-1);
      if (op.getWaitDevnum()) {
        llvm::Value *devnum = moduleTranslation.lookupValue(op.getWaitDevnum());
        if (devnum) {
          if (!devnum->getType()->isIntegerTy(32))
            devnum = builder.CreateIntCast(devnum, builder.getInt32Ty(), true);
          deviceNum = devnum;
        }
      }
      builder.CreateCall(getAccWaitFunction(*module, ctx),
                         {srcLocInfo, builder.getInt64(0), builder.getInt64(0),
                          deviceNum, builder.getInt32(waitNum), waitList,
                          asyncVal});
    }
  }

  // Helper lambda to emit the data call
  auto emitCall = [&]() {
    builder.CreateCall(mapperFunc,
                       {srcLocInfo, flagsVal, deviceTypeVal, argNumVal,
                        mapperAllocas.ArgsBase, mapperAllocas.Args,
                        mapperAllocas.ArgSizes, maptypesArg, mapnamesArg,
                        nullPtr, nullPtr, asyncVal});
  };

  // Handle if clause
  if constexpr (std::is_same_v<OpTy, acc::EnterDataOp> ||
                std::is_same_v<OpTy, acc::ExitDataOp>) {
    if (op.getIfCond()) {
      llvm::Value *cond = moduleTranslation.lookupValue(op.getIfCond());
      if (!cond) {
        op.emitError("could not find LLVM value for if condition");
        return failure();
      }
      llvm::Function *function = builder.GetInsertBlock()->getParent();
      llvm::BasicBlock *thenBlock =
          llvm::BasicBlock::Create(ctx, "acc.standalone.then", function);
      llvm::BasicBlock *endBlock =
          llvm::BasicBlock::Create(ctx, "acc.standalone.end", function);
      builder.CreateCondBr(cond, thenBlock, endBlock);
      builder.SetInsertPoint(thenBlock);
      emitCall();
      builder.CreateBr(endBlock);
      builder.SetInsertPoint(endBlock);
    } else {
      emitCall();
    }
  } else {
    emitCall();
  }

  return success();
}

/// Converts an OpenACC update operation into LLVM IR.
static LogicalResult
convertUpdateOp(acc::UpdateOp op, llvm::IRBuilderBase &builder,
                LLVM::ModuleTranslation &moduleTranslation) {
  auto enclosingFuncOp = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  unsigned totalNbOperand = op.getNumDataOperands();
  OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, totalNbOperand,
                                  mapperAllocas);

  SmallVector<uint64_t> flags;
  SmallVector<llvm::Constant *> names;
  if (failed(processDataOperands(builder, moduleTranslation, op, flags, names,
                                 mapperAllocas)))
    return failure();

  auto *maptypes =
      accBuilder->createOffloadMaptypes(flags, ".offload_maptypes_update");
  auto *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      maptypes, 0, 0);
  auto *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames_update");
  auto *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::PointerType::getUnqual(ctx), totalNbOperand),
      mapnames, 0, 0);
  auto *nullPtr =
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));

  llvm::Value *asyncVal = builder.getInt64(-1);
  // Only the async and wait values of DeviceType::None are translated;
  // device_type specific values are silently dropped for now.
  // TODO: support device_type specific async and wait values.
  if (mlir::Value asyncValue = op.getAsyncValue()) {
    asyncVal = moduleTranslation.lookupValue(asyncValue);
    if (!asyncVal) {
      op.emitError("could not find LLVM value for async operand");
      return failure();
    }
    if (!asyncVal->getType()->isIntegerTy(64))
      asyncVal = builder.CreateIntCast(asyncVal, builder.getInt64Ty(), true);
  } else if (op.hasAsyncOnly()) {
    asyncVal = builder.getInt64(-2);
  }

  auto emitUpdate = [&]() -> LogicalResult {
    if (!op.getWaitValues().empty() || op.hasWaitOnly()) {
      uint32_t waitNum = op.getWaitValues().size();
      llvm::Value *waitList = nullPtr;
      if (waitNum) {
        auto *arrayType =
            llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), waitNum);
        waitList = builder.CreateAlloca(arrayType);
        for (auto [waitIndex, waitOperand] :
             llvm::enumerate(op.getWaitValues())) {
          llvm::Value *waitValue = moduleTranslation.lookupValue(waitOperand);
          if (!waitValue) {
            op.emitError("could not find LLVM value for wait operand")
                << waitIndex;
            return failure();
          }
          if (!waitValue->getType()->isIntegerTy(64))
            waitValue =
                builder.CreateIntCast(waitValue, builder.getInt64Ty(), true);
          auto *element = builder.CreateInBoundsGEP(
              arrayType, waitList,
              {builder.getInt32(0), builder.getInt32(waitIndex)});
          builder.CreateStore(waitValue, element);
        }
      }

      llvm::Value *waitDevnum = builder.getInt32(-1);
      if (mlir::Value devnumValue = op.getWaitDevnum()) {
        waitDevnum = moduleTranslation.lookupValue(devnumValue);
        if (!waitDevnum) {
          op.emitError("could not find LLVM value for wait device number");
          return failure();
        }
        if (!waitDevnum->getType()->isIntegerTy(32))
          waitDevnum =
              builder.CreateIntCast(waitDevnum, builder.getInt32Ty(), true);
      }

      builder.CreateCall(
          getAccWaitFunction(*module, ctx),
          {createSourceLocationInfo(*accBuilder, op), builder.getInt64(0),
           builder.getInt64(1), waitDevnum, builder.getInt32(waitNum), waitList,
           asyncVal});
    }

    builder.CreateCall(
        getAccDataUpdateFunction(*module, ctx),
        {createSourceLocationInfo(*accBuilder, op), builder.getInt64(0),
         builder.getInt64(1), builder.getInt32(totalNbOperand),
         mapperAllocas.ArgsBase, mapperAllocas.Args, mapperAllocas.ArgSizes,
         maptypesArg, mapnamesArg, nullPtr, nullPtr, asyncVal});
    return success();
  };

  if (op.getIfCond()) {
    llvm::Value *cond = moduleTranslation.lookupValue(op.getIfCond());
    if (!cond) {
      op.emitError("could not find LLVM value for if condition");
      return failure();
    }
    llvm::Function *function = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *thenBlock =
        llvm::BasicBlock::Create(ctx, "acc.update.then", function);
    llvm::BasicBlock *endBlock =
        llvm::BasicBlock::Create(ctx, "acc.update.end", function);
    builder.CreateCondBr(cond, thenBlock, endBlock);
    builder.SetInsertPoint(thenBlock);
    if (failed(emitUpdate()))
      return failure();
    builder.CreateBr(endBlock);
    builder.SetInsertPoint(endBlock);
  } else if (failed(emitUpdate())) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Conversion functions for init/shutdown/set/wait
//===----------------------------------------------------------------------===//

/// Map MLIR DeviceType enum to OpenACC runtime acc_device_t values.
/// MLIR enum values match OpenACC spec; runtime values differ.
/// TODO: Unify MLIR and runtime enums to eliminate this mapping.
/// TODO: Multicore  mapped to multicore, currently mapped to host.
static int64_t mapDeviceTypeToRuntime(acc::DeviceType mlirType) {
  switch (mlirType) {
  case acc::DeviceType::None:      return 0; // none
  case acc::DeviceType::Star:      return 0; // (don't know)
  case acc::DeviceType::Default:   return 1; // default
  case acc::DeviceType::Host:      return 2; // host
  case acc::DeviceType::Multicore: return 2; // host
  case acc::DeviceType::Nvidia:    return 4; // nvidia
  case acc::DeviceType::Radeon:    return 5; // amd
  default:                         return 1; // Unknown->default
  }
}

/// Converts acc.init operation into LLVM IR.
static LogicalResult convertInitOp(acc::InitOp op,
                                    llvm::IRBuilderBase &builder,
                                    LLVM::ModuleTranslation &moduleTranslation) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);
  auto *fn = getAccInitFunction(*module, ctx);

  int64_t deviceType = 1; // default
  if (op.getDeviceTypes()) {
    auto dtypes = op.getDeviceTypes()->getValue();
    if (!dtypes.empty()) {
      auto initDeviceType = mlir::cast<mlir::acc::DeviceTypeAttr>(dtypes[0]).getValue();
      deviceType = mapDeviceTypeToRuntime(initDeviceType);
    }
  }

  llvm::Value *deviceNumVal = builder.getInt64(-1);
  if (op.getDeviceNum()) {
    deviceNumVal = moduleTranslation.lookupValue(op.getDeviceNum());
    if (deviceNumVal->getType() != llvm::Type::getInt64Ty(ctx))
      deviceNumVal = builder.CreateZExt(deviceNumVal, llvm::Type::getInt64Ty(ctx));
  }

  // Handle if(condition)
  if (op.getIfCond()) {
    llvm::Value *cond = moduleTranslation.lookupValue(op.getIfCond());
    if (!cond) {
      op.emitError("could not find LLVM value for if condition");
      return failure();
    }
    llvm::Function *func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *thenBlock = llvm::BasicBlock::Create(ctx, "acc.init.then", func);
    llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(ctx, "acc.init.end", func);
    builder.CreateCondBr(cond, thenBlock, endBlock);
    builder.SetInsertPoint(thenBlock);
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                            builder.getInt64(deviceType), deviceNumVal});
    builder.CreateBr(endBlock);
    builder.SetInsertPoint(endBlock);
  } else {
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                            builder.getInt64(deviceType), deviceNumVal});
  }
  return success();
}

/// Converts acc.shutdown operation into LLVM IR.
static LogicalResult convertShutdownOp(acc::ShutdownOp op,
                                        llvm::IRBuilderBase &builder,
                                        LLVM::ModuleTranslation &moduleTranslation) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);
  auto *fn = getAccShutdownFunction(*module, ctx);

  int64_t deviceType = 1; // default
  if (op.getDeviceTypes()) {
    auto dtypes = op.getDeviceTypes()->getValue();
    if (!dtypes.empty()) {
      auto shutdownDeviceType = mlir::cast<mlir::acc::DeviceTypeAttr>(dtypes[0]).getValue();
      deviceType = mapDeviceTypeToRuntime(shutdownDeviceType);
    }
  }

  llvm::Value *deviceNumVal = builder.getInt64(-1);
  if (op.getDeviceNum()) {
    deviceNumVal = moduleTranslation.lookupValue(op.getDeviceNum());
    if (deviceNumVal->getType() != llvm::Type::getInt64Ty(ctx))
      deviceNumVal = builder.CreateZExt(deviceNumVal, llvm::Type::getInt64Ty(ctx));
  }

  // Handle if(condition)
  if (op.getIfCond()) {
    llvm::Value *cond = moduleTranslation.lookupValue(op.getIfCond());
    if (!cond) {
      op.emitError("could not find LLVM value for if condition");
      return failure();
    }
    llvm::Function *func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *thenBlock = llvm::BasicBlock::Create(ctx, "acc.shutdown.then", func);
    llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(ctx, "acc.shutdown.end", func);
    builder.CreateCondBr(cond, thenBlock, endBlock);
    builder.SetInsertPoint(thenBlock);
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                            builder.getInt64(deviceType), deviceNumVal});
    builder.CreateBr(endBlock);
    builder.SetInsertPoint(endBlock);
  } else {
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                            builder.getInt64(deviceType), deviceNumVal});
  }
  return success();
}

/// Converts acc.set operation into LLVM IR.
static LogicalResult convertSetOp(acc::SetOp op,
                                   llvm::IRBuilderBase &builder,
                                   LLVM::ModuleTranslation &moduleTranslation) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);

  // Handle if(condition)
  llvm::Value *cond = nullptr;
  if (op.getIfCond()) {
    cond = moduleTranslation.lookupValue(op.getIfCond());
    if (!cond) {
      op.emitError("could not find LLVM value for if condition");
      return failure();
    }
  }

  // Create basic blocks for conditional execution
  llvm::Function *func = builder.GetInsertBlock()->getParent();
  llvm::BasicBlock *thenBlock = cond ? llvm::BasicBlock::Create(ctx, "acc.set.then", func) : nullptr;
  llvm::BasicBlock *endBlock = cond ? llvm::BasicBlock::Create(ctx, "acc.set.end", func) : nullptr;

  if (cond) {
    builder.CreateCondBr(cond, thenBlock, endBlock);
    builder.SetInsertPoint(thenBlock);
  }

  if (op.getDefaultAsync()) {
    auto *fn = getAccSetDefaultAsyncFunction(*module, ctx);
    llvm::Value *asyncVal = moduleTranslation.lookupValue(op.getDefaultAsync());
    if (asyncVal->getType() != llvm::Type::getInt64Ty(ctx))
      asyncVal = builder.CreateZExt(asyncVal, llvm::Type::getInt64Ty(ctx));
    builder.CreateCall(fn, {srcLocInfo, asyncVal});
  }

  if (op.getDeviceNum()) {
    auto *fn = getAccSetDeviceNumFunction(*module, ctx);
    llvm::Value *deviceNumVal = moduleTranslation.lookupValue(op.getDeviceNum());
    if (deviceNumVal->getType() != llvm::Type::getInt64Ty(ctx))
      deviceNumVal = builder.CreateZExt(deviceNumVal, llvm::Type::getInt64Ty(ctx));
    int64_t deviceType = 1; // default
    if (op.getDeviceType())
      deviceType = mapDeviceTypeToRuntime(*op.getDeviceType());
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                           builder.getInt64(deviceType), deviceNumVal});
  } else if (op.getDeviceType()) {
    auto *fn = getAccSetDeviceTypeFunction(*module, ctx);
    int64_t rtDeviceType = mapDeviceTypeToRuntime(*op.getDeviceType());
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                           builder.getInt64(rtDeviceType)});
  }

  if (cond) {
    builder.CreateBr(endBlock);
    builder.SetInsertPoint(endBlock);
  }

  return success();
}

/// Converts acc.wait operation into LLVM IR.
static LogicalResult convertWaitOp(acc::WaitOp op,
                                    llvm::IRBuilderBase &builder,
                                    LLVM::ModuleTranslation &moduleTranslation) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();
  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);
  auto *fn = getAccWaitFunction(*module, ctx);

  // Handle if(condition)
  llvm::Value *cond = nullptr;
  if (op.getIfCond()) {
    cond = moduleTranslation.lookupValue(op.getIfCond());
    if (!cond) {
      op.emitError("could not find LLVM value for if condition");
      return failure();
    }
  }

  llvm::Value *deviceNum = builder.getInt32(-1);
  if (op.getWaitDevnum()) {
    deviceNum = moduleTranslation.lookupValue(op.getWaitDevnum());
    if (!deviceNum) {
      op.emitError("could not find LLVM value for wait device number");
      return failure();
    }
    if (deviceNum->getType() != llvm::Type::getInt32Ty(ctx))
      deviceNum = builder.CreateIntCast(deviceNum,
                                        llvm::Type::getInt32Ty(ctx), true);
  }

  unsigned waitNum = op.getWaitOperands().size();
  llvm::Value *waitListPtr =
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));
  if (waitNum) {
    auto *arrTy = llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), waitNum);
    auto *waitList = builder.CreateAlloca(arrTy);
    for (auto [index, waitValue] : llvm::enumerate(op.getWaitOperands())) {
      llvm::Value *wait = moduleTranslation.lookupValue(waitValue);
      if (!wait) {
        op.emitError("could not find LLVM value for wait operand");
        return failure();
      }
      if (wait->getType() != llvm::Type::getInt64Ty(ctx))
        wait = builder.CreateIntCast(wait, llvm::Type::getInt64Ty(ctx), true);
      auto *gep = builder.CreateInBoundsGEP(
          arrTy, waitList, {builder.getInt32(0), builder.getInt32(index)});
      builder.CreateStore(wait, gep);
    }
    waitListPtr = waitList;
  }

  llvm::Value *async = builder.getInt64(-1);
  bool asyncOnly = false;
  if (op.getAsync()) {
    asyncOnly = true;
  }
  if (op.getAsyncOperand()) {
    async = moduleTranslation.lookupValue(op.getAsyncOperand());
    if (!async) {
      op.emitError("could not find LLVM value for async operand");
      return failure();
    }
    if (async->getType() != llvm::Type::getInt64Ty(ctx))
      async = builder.CreateIntCast(async, llvm::Type::getInt64Ty(ctx), true);
  } else if (asyncOnly) {
    async = builder.getInt64(-2);
  }

  // Create basic blocks for conditional execution
  llvm::Function *func = builder.GetInsertBlock()->getParent();
  llvm::BasicBlock *thenBlock = cond ? llvm::BasicBlock::Create(ctx, "acc.wait.then", func) : nullptr;
  llvm::BasicBlock *endBlock = cond ? llvm::BasicBlock::Create(ctx, "acc.wait.end", func) : nullptr;

  if (cond) {
    builder.CreateCondBr(cond, thenBlock, endBlock);
    builder.SetInsertPoint(thenBlock);
  }

  builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                          builder.getInt64(0), deviceNum,
                          builder.getInt32(waitNum), waitListPtr, async});

  if (cond) {
    builder.CreateBr(endBlock);
    builder.SetInsertPoint(endBlock);
  }

  return success();
}

/// Convert acc.declare_enter using the structured reference counter runtime
/// ABI.
static LogicalResult
convertDeclareEnterOp(acc::DeclareEnterOp op, llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  auto enclosingFuncOp = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  unsigned totalNbOperand = 0;
  for (Value data : op.getDataClauseOperands()) {
    Operation *entry = data.getDefiningOp();
    if (!entry)
      return op.emitError("declare operand has no defining operation");
    auto clause = acc::getDataClause(entry);
    if (!clause)
      return op.emitError("declare operand has no data clause");
    if (*clause != acc::DataClause::acc_deviceptr &&
        *clause != acc::DataClause::acc_declare_link &&
        *clause != acc::DataClause::acc_getdeviceptr)
      ++totalNbOperand;
  }

  if (totalNbOperand == 0)
    return success();

  OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, totalNbOperand,
                                  mapperAllocas);

  SmallVector<uint64_t> flags;
  SmallVector<llvm::Constant *> names;
  unsigned index = 0;
  for (Value data : op.getDataClauseOperands()) {
    Operation *entry = data.getDefiningOp();
    acc::DataClause clause = acc::getDataClause(entry).value();
    if (clause == acc::DataClause::acc_deviceptr ||
        clause == acc::DataClause::acc_declare_link ||
        clause == acc::DataClause::acc_getdeviceptr)
      continue;

    uint64_t flag = kAccMapTypeNone;
    switch (clause) {
    case acc::DataClause::acc_copy:
    case acc::DataClause::acc_copyin:
    case acc::DataClause::acc_copyin_readonly:
      flag |= kAccMapTypeTo;
      break;
    case acc::DataClause::acc_present:
      flag |= kAccMapTypePresent | kAccMapTypeNoCreate;
      break;
    case acc::DataClause::acc_declare_device_resident:
      flag |= kAccMapTypeDeviceResident;
      break;
    case acc::DataClause::acc_copyout:
    case acc::DataClause::acc_copyout_zero:
    case acc::DataClause::acc_create:
    case acc::DataClause::acc_create_zero:
      break;
    default:
      return op.emitError("unsupported data clause on declare_enter");
    }

    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, entry, acc::getVarPtr(entry),
            totalNbOperand, flag, flags, names, index, mapperAllocas)))
      return failure();
  }

  auto *maptypes =
      accBuilder->createOffloadMaptypes(flags, ".offload_maptypes_declare");
  auto *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      maptypes, 0, 0);
  auto *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames_declare");
  auto *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::PointerType::getUnqual(ctx), totalNbOperand),
      mapnames, 0, 0);
  auto *nullPtr =
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));

  builder.CreateCall(getAccDeclareFunction(*module, ctx),
                     {createSourceLocationInfo(*accBuilder, op),
                      builder.getInt64(0), builder.getInt64(0),
                      builder.getInt32(totalNbOperand), mapperAllocas.ArgsBase,
                      mapperAllocas.Args, mapperAllocas.ArgSizes, maptypesArg,
                      mapnamesArg, nullPtr, nullPtr, builder.getInt64(-1),
                      nullPtr});
  return success();
}

/// Convert acc.declare_exit. Direct data-entry operands are the normal form
/// emitted by Flang; getdeviceptr operands are also accepted for compatibility.
static LogicalResult
convertDeclareExitOp(acc::DeclareExitOp op, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  auto enclosingFuncOp = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  struct Mapping {
    Operation *entry;
    Value varPtr;
    uint64_t flag;
  };
  SmallVector<Mapping> mappings;
  OperandRange exitDataOperands = op.getDataClauseOperands();
  // A token-only declare_exit inherits the data operands of its matching
  // declare_enter so the structured reference count is properly released.
  if (exitDataOperands.empty())
    if (Value token = op.getToken())
      if (auto enterOp = dyn_cast_or_null<acc::DeclareEnterOp>(
              token.getDefiningOp()))
        exitDataOperands = enterOp.getDataClauseOperands();
  for (Value data : exitDataOperands) {
    Operation *entry = data.getDefiningOp();
    if (!entry)
      return op.emitError("declare operand has no defining operation");
    auto clause = acc::getDataClause(entry);
    if (!clause)
      return op.emitError("declare operand has no data clause");
    if (*clause == acc::DataClause::acc_deviceptr ||
        *clause == acc::DataClause::acc_declare_link ||
        *clause == acc::DataClause::acc_getdeviceptr)
      continue;

    Value varPtr = acc::getVarPtr(entry);
    Value accPtr = acc::getAccPtr(entry);
    bool hasAction = false;
    if (accPtr) {
      for (OpOperand &use : accPtr.getUses()) {
        if (isa<acc::CopyoutOp>(use.getOwner())) {
          mappings.push_back(
              {entry, varPtr, kAccMapTypeFrom});
          hasAction = true;
        } else if (isa<acc::DeleteOp, acc::DetachOp>(use.getOwner())) {
          mappings.push_back(
              {entry, varPtr, kAccMapTypeFinalize});
          hasAction = true;
        }
      }
    }

    if (hasAction)
      continue;
    switch (*clause) {
    case acc::DataClause::acc_copy:
    case acc::DataClause::acc_copyout:
    case acc::DataClause::acc_copyout_zero:
      mappings.push_back(
          {entry, varPtr, kAccMapTypeFrom});
      break;
    case acc::DataClause::acc_present:
      mappings.push_back(
          {entry, varPtr,
           kAccMapTypePresent | kAccMapTypeNoCreate});
      break;
    case acc::DataClause::acc_copyin:
    case acc::DataClause::acc_copyin_readonly:
    case acc::DataClause::acc_create:
    case acc::DataClause::acc_create_zero:
    case acc::DataClause::acc_declare_device_resident:
      mappings.push_back(
          {entry, varPtr, kAccMapTypeFinalize});
      break;
    default:
      return op.emitError("unsupported data clause on declare_exit");
    }
  }

  if (mappings.empty())
    return success();

  OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, mappings.size(),
                                  mapperAllocas);

  SmallVector<uint64_t> flags;
  SmallVector<llvm::Constant *> names;
  unsigned index = 0;
  for (const Mapping &mapping : mappings) {
    if (isa<acc::GetDevicePtrOp>(mapping.entry)) {
      if (failed(processOperands(builder, moduleTranslation, op,
                                 ValueRange(mapping.varPtr), mappings.size(),
                                 mapping.flag, flags, names, index,
                                 mapperAllocas)))
        return failure();
    } else if (failed(processDataOperandWithBounds(
                   builder, moduleTranslation, mapping.entry, mapping.varPtr,
                   mappings.size(), mapping.flag, flags, names, index,
                   mapperAllocas))) {
      return failure();
    }
  }

  auto *maptypes =
      accBuilder->createOffloadMaptypes(flags, ".offload_maptypes_declare_end");
  auto *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), mappings.size()),
      maptypes, 0, 0);
  auto *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames_declare_end");
  auto *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::PointerType::getUnqual(ctx), mappings.size()),
      mapnames, 0, 0);
  auto *nullPtr =
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));
  // declare_enter uses structured reference counting through
  // __tgt_acc_declare, so its matching exit must use the structured
  // __tgt_acc_data_end entry point rather than dynamic __tgt_acc_data_exit.
  builder.CreateCall(getAccDataEndFunction(*module, ctx),
                     {createSourceLocationInfo(*accBuilder, op),
                      builder.getInt64(0), builder.getInt64(0),
                      builder.getInt32(mappings.size()), mapperAllocas.ArgsBase,
                      mapperAllocas.Args, mapperAllocas.ArgSizes, maptypesArg,
                      mapnamesArg, nullPtr, nullPtr, builder.getInt64(-1)});
  return success();
}

/// Convert an acc.declare implicit region. Unlike acc.data, the entry uses
/// the structured declare ABI and the region exit uses the structured data_end
/// ABI.
static LogicalResult
convertDeclareOp(acc::DeclareOp op, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  auto enclosingFuncOp = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  struct Mapping {
    Operation *entry;
    Value varPtr;
    uint64_t entryFlag;
    uint64_t exitFlag;
  };
  SmallVector<Mapping> mappings;
  for (Value data : op.getDataClauseOperands()) {
    Operation *entry = data.getDefiningOp();
    if (!entry)
      return op.emitError("declare operand has no defining operation");
    auto clause = acc::getDataClause(entry);
    if (!clause)
      return op.emitError("declare operand has no data clause");
    if (*clause == acc::DataClause::acc_deviceptr ||
        *clause == acc::DataClause::acc_declare_link)
      continue;

    uint64_t entryFlag = kAccMapTypeNone;
    uint64_t exitFlag = kAccMapTypeFinalize;
    switch (*clause) {
    case acc::DataClause::acc_copy:
    case acc::DataClause::acc_copyin:
    case acc::DataClause::acc_copyin_readonly:
      entryFlag |= kAccMapTypeTo;
      break;
    case acc::DataClause::acc_copyout:
    case acc::DataClause::acc_copyout_zero:
      exitFlag = kAccMapTypeFrom;
      break;
    case acc::DataClause::acc_present:
      entryFlag |= kAccMapTypePresent | kAccMapTypeNoCreate;
      exitFlag =
          kAccMapTypePresent | kAccMapTypeNoCreate;
      break;
    case acc::DataClause::acc_declare_device_resident:
      entryFlag |= kAccMapTypeDeviceResident;
      break;
    case acc::DataClause::acc_create:
    case acc::DataClause::acc_create_zero:
      break;
    default:
      return op.emitError("unsupported data clause on declare");
    }
    mappings.push_back({entry, acc::getVarPtr(entry), entryFlag, exitFlag});
  }

  if (mappings.empty()) {
    for (Block &bb : op.getRegion()) {
      if (failed(
              moduleTranslation.convertBlock(bb, bb.isEntryBlock(), builder)))
        return failure();
    }
    return success();
  }

  OpenACCIRBuilder::MapperAllocas mapperAllocas;
  OpenACCIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  accBuilder->createMapperAllocas(builder.saveIP(), allocaIP, mappings.size(),
                                  mapperAllocas);

  SmallVector<uint64_t> entryFlags;
  SmallVector<llvm::Constant *> names;
  unsigned index = 0;
  for (const Mapping &mapping : mappings) {
    if (failed(processDataOperandWithBounds(
            builder, moduleTranslation, mapping.entry, mapping.varPtr,
            mappings.size(), mapping.entryFlag, entryFlags, names, index,
            mapperAllocas)))
      return failure();
  }

  auto *maptypes = accBuilder->createOffloadMaptypes(
      entryFlags, ".offload_maptypes_declare");
  auto *maptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), mappings.size()),
      maptypes, 0, 0);
  auto *mapnames =
      accBuilder->createOffloadMapnames(names, ".offload_mapnames_declare");
  auto *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::PointerType::getUnqual(ctx), mappings.size()),
      mapnames, 0, 0);
  auto *nullPtr =
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));
  builder.CreateCall(getAccDeclareFunction(*module, ctx),
                     {createSourceLocationInfo(*accBuilder, op),
                      builder.getInt64(0), builder.getInt64(0),
                      builder.getInt32(mappings.size()), mapperAllocas.ArgsBase,
                      mapperAllocas.Args, mapperAllocas.ArgSizes, maptypesArg,
                      mapnamesArg, nullPtr, nullPtr, builder.getInt64(-1),
                      nullPtr});

  SmallVector<uint64_t> exitFlags;
  for (const Mapping &mapping : mappings)
    exitFlags.push_back(mapping.exitFlag);
  auto *endMaptypes = accBuilder->createOffloadMaptypes(
      exitFlags, ".offload_maptypes_declare_end");
  auto *endMaptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), mappings.size()),
      endMaptypes, 0, 0);

  // An implicit declare region may be empty. In that case there is no block
  // to branch to, but the structured mapping still needs to be ended.
  if (op.getRegion().empty()) {
    builder.CreateCall(
        getAccDataEndFunction(*module, ctx),
        {createSourceLocationInfo(*accBuilder, op), builder.getInt64(0),
         builder.getInt64(0), builder.getInt32(mappings.size()),
         mapperAllocas.ArgsBase, mapperAllocas.Args, mapperAllocas.ArgSizes,
         endMaptypesArg, mapnamesArg, nullPtr, nullPtr, builder.getInt64(-1)});
    return success();
  }

  llvm::BasicBlock *entryBlock = nullptr;
  for (Block &bb : op.getRegion()) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        ctx, "acc.declare", builder.GetInsertBlock()->getParent());
    if (!entryBlock)
      entryBlock = llvmBB;
    moduleTranslation.mapBlock(&bb, llvmBB);
  }
  llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
      ctx, "acc.end_declare", builder.GetInsertBlock()->getParent());
  auto afterDeclare = builder.saveIP();
  builder.CreateBr(entryBlock);
  builder.restoreIP(afterDeclare);

  for (Block *bb : getBlocksSortedByDominance(op.getRegion())) {
    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder)))
      return failure();
    if (!bb->getTerminator() ||
        isa<acc::TerminatorOp, acc::YieldOp>(bb->getTerminator()))
      builder.CreateBr(endBlock);
  }
  LLVM::detail::connectPHINodes(op.getRegion(), moduleTranslation);
  builder.SetInsertPoint(endBlock);
  builder.CreateCall(
      getAccDataEndFunction(*module, ctx),
      {createSourceLocationInfo(*accBuilder, op), builder.getInt64(0),
       builder.getInt64(0), builder.getInt32(mappings.size()),
       mapperAllocas.ArgsBase, mapperAllocas.Args, mapperAllocas.ArgSizes,
       endMaptypesArg, mapnamesArg, nullPtr, nullPtr, builder.getInt64(-1)});
  return success();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the OpenACC dialect to LLVM IR.
class OpenACCDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // namespace

/// Given an OpenACC MLIR operation, create the corresponding LLVM IR
/// (including OpenACC runtime calls).
LogicalResult OpenACCDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](acc::DataOp dataOp) {
        return convertDataOp(dataOp, builder, moduleTranslation);
      })
      .Case([&](acc::DeclareOp declareOp) {
        return convertDeclareOp(declareOp, builder, moduleTranslation);
      })
      .Case([&](acc::HostDataOp hostDataOp) {
        return convertHostDataOp(hostDataOp, builder, moduleTranslation);
      })
      .Case([&](acc::EnterDataOp enterDataOp) {
        return convertStandaloneDataOp<acc::EnterDataOp>(enterDataOp, builder,
                                                         moduleTranslation);
      })
      .Case([&](acc::ExitDataOp exitDataOp) {
        return convertStandaloneDataOp<acc::ExitDataOp>(exitDataOp, builder,
                                                        moduleTranslation);
      })
      .Case([&](acc::UpdateOp updateOp) {
        return convertUpdateOp(updateOp, builder, moduleTranslation);
      })
      .Case([&](acc::InitOp initOp) {
        return convertInitOp(initOp, builder, moduleTranslation);
      })
      .Case([&](acc::ShutdownOp shutdownOp) {
        return convertShutdownOp(shutdownOp, builder, moduleTranslation);
      })
      .Case([&](acc::SetOp setOp) {
        return convertSetOp(setOp, builder, moduleTranslation);
      })
      .Case([&](acc::WaitOp waitOp) {
        return convertWaitOp(waitOp, builder, moduleTranslation);
      })
      .Case([&](acc::DeclareEnterOp declareEnterOp) {
        return convertDeclareEnterOp(declareEnterOp, builder,
                                     moduleTranslation);
      })
      .Case([&](acc::DeclareExitOp declareExitOp) {
        return convertDeclareExitOp(declareExitOp, builder,
                                    moduleTranslation);
      })
      .Case<acc::TerminatorOp, acc::YieldOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        assert(op->getNumOperands() == 0 &&
               "unexpected OpenACC terminator with operands");
        return success();
      })
      .Case<acc::DataBoundsOp>([](auto op) {
        // Bounds are metadata consumed by data operations above.
        return success();
      })
      .Case<acc::ParallelOp, acc::SerialOp, acc::KernelsOp>([&](auto op) {
        // TODO: Implement compute construct lowering to generate runtime call
        //
        // Required implementation:
        // 1. Process data clause operands (copyin, copyout, create, present)
        // 2. Emit __tgt_acc_data_begin() before the region
        // 3. Convert the region body (inline into host function)
        // 4. Emit __tgt_acc_data_end() after the region
        //
        // For kernels/parallel:
        // - Should also generate target kernel launch for GPU offloading
        // - CPU fallback: execute region inline on host
        //
        // For serial:
        // - Execute region inline on host (no parallelism)
        //
        // Current behavior: NOP - region code is inlined but no data transfer
        // or kernel launch occurs, leading to incorrect execution.
        return success();
      })
      .Case<acc::LoopOp>([](auto op) {
        // Loop construct - NOP for now.
        return success();
      })
      .Case<acc::AtomicUpdateOp, acc::AtomicCaptureOp, acc::AtomicReadOp,
            acc::AtomicWriteOp>([](auto op) {
        // Atomic operations - NOP for now, should be implemented with
        // LLVM atomic instructions. TODO: Add proper implementation.
        return success();
      })
      .Case<acc::UpdateHostOp, acc::UpdateDeviceOp>([](auto op) {
        // Data update operations - NOP, consumed by data operations.
        return success();
      })
      .Case<acc::ReductionInitOp, acc::ReductionCombineOp,
            acc::ReductionCombineRegionOp, acc::ReductionAccumulateOp,
            acc::ReductionAccumulateArrayOp>([](auto op) {
        // Reduction operations - NOP, consumed by compute constructs.
        return success();
      })
      .Case<acc::FirstprivateMapInitialOp, acc::PrivatizeOp,
            acc::UnwrapPrivateOp, acc::PrivateLocalOp>(
          [](auto op) { return success(); })
      .Case<acc::OnDeviceOp>([&](acc::OnDeviceOp op) {
        // acc_on_device: returns true when running on device.
        // For host compilation, return false (i1 zero).
        // TODO: Add proper device detection.
        llvm::Value *result = builder.getInt1(false);
        moduleTranslation.mapValue(op.getResult(), result);
        return success();
      })
      .Case<acc::CreateOp, acc::CopyinOp, acc::CopyoutOp, acc::PresentOp,
            acc::NoCreateOp, acc::DevicePtrOp, acc::AttachOp,
            acc::DeclareDeviceResidentOp, acc::DeclareLinkOp,
            acc::GetDevicePtrOp>([&](auto op) {
        llvm::Value *varPtr = moduleTranslation.lookupValue(op.getVarPtr());
        if (!varPtr) {
          op.emitError("could not find LLVM value for varPtr");
          return failure();
        }
        varPtr = getBoundedMappingPointer(op.getOperation(), varPtr,
                                          moduleTranslation, builder);
        if (!moduleTranslation.lookupValue(op.getAccPtr()))
          moduleTranslation.mapValue(op.getAccPtr(), varPtr);
        return success();
      })
      .Case<acc::UseDeviceOp>([&](acc::UseDeviceOp op) {
        return convertUseDeviceOp(op, builder, moduleTranslation);
      })
      .Case<acc::DeleteOp, acc::DetachOp>(
          [](auto op) { return success(); })
      .Default([&](Operation *op) {
        return op->emitError("unsupported OpenACC operation: ")
               << op->getName();
      });
}

void mlir::registerOpenACCDialectTranslation(DialectRegistry &registry) {
  registry.insert<acc::OpenACCDialect>();
  registry.addExtension(+[](MLIRContext *ctx, acc::OpenACCDialect *dialect) {
    dialect->addInterfaces<OpenACCDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerOpenACCDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOpenACCDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
