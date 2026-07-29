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

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Flag values are extracted from openmp/libomptarget/include/omptarget.h and
/// mapped to corresponding OpenACC flags.
static constexpr uint64_t kCreateFlag = 0x000;
static constexpr uint64_t kDeviceCopyinFlag = 0x001;
static constexpr uint64_t kHostCopyoutFlag = 0x002;
static constexpr uint64_t kPresentFlag = 0x1000;
static constexpr uint64_t kDeleteFlag = 0x008;
// Runtime extension to implement the OpenACC second reference counter.
static constexpr uint64_t kHoldFlag = 0x2000;

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

/// Return the runtime function used to lower the given operation.
static llvm::Function *getAssociatedFunction(OpenACCIRBuilder &builder,
                                             Operation *op) {
  return llvm::TypeSwitch<Operation *, llvm::Function *>(op)
      .Case([&](acc::EnterDataOp) {
        return builder.getOrCreateRuntimeFunctionPtr(
            llvm::omp::OMPRTL___tgt_target_data_begin_mapper);
      })
      .Case([&](acc::ExitDataOp) {
        return builder.getOrCreateRuntimeFunctionPtr(
            llvm::omp::OMPRTL___tgt_target_data_end_mapper);
      })
      .Case([&](acc::UpdateOp) {
        return builder.getOrCreateRuntimeFunctionPtr(
            llvm::omp::OMPRTL___tgt_target_data_update_mapper);
      });
  llvm_unreachable("Unknown OpenACC operation");
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

/// Process data operands from acc::EnterDataOp
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::EnterDataOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  // TODO add `create_zero` and `attach` operands

  unsigned index = 0;

  // Create operands are handled as `alloc` call.
  // Copyin operands are handled as `to` call.
  llvm::SmallVector<mlir::Value> create, copyin;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto createOp = dataOp.getDefiningOp<acc::CreateOp>()) {
      create.push_back(createOp.getVarPtr());
    } else if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(
                   dataOp.getDefiningOp())) {
      copyin.push_back(copyinOp.getVarPtr());
    }
  }

  auto nbTotalOperands = create.size() + copyin.size();

  // Create operands are handled as `alloc` call.
  if (failed(processOperands(builder, moduleTranslation, op, create,
                             nbTotalOperands, kCreateFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  // Copyin operands are handled as `to` call.
  if (failed(processOperands(builder, moduleTranslation, op, copyin,
                             nbTotalOperands, kDeviceCopyinFlag, flags, names,
                             index, mapperAllocas)))
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
  // TODO add `detach` operands

  unsigned index = 0;

  llvm::SmallVector<mlir::Value> deleteOperands, copyoutOperands;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(
            dataOp.getDefiningOp())) {
      for (auto &u : devicePtrOp.getAccPtr().getUses()) {
        if (mlir::dyn_cast_or_null<acc::DeleteOp>(u.getOwner()))
          deleteOperands.push_back(devicePtrOp.getVarPtr());
        else if (mlir::dyn_cast_or_null<acc::CopyoutOp>(u.getOwner()))
          copyoutOperands.push_back(devicePtrOp.getVarPtr());
      }
    }
  }

  auto nbTotalOperands = deleteOperands.size() + copyoutOperands.size();

  // Delete operands are handled as `delete` call.
  if (failed(processOperands(builder, moduleTranslation, op, deleteOperands,
                             nbTotalOperands, kDeleteFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  // Copyout operands are handled as `from` call.
  if (failed(processOperands(builder, moduleTranslation, op, copyoutOperands,
                             nbTotalOperands, kHostCopyoutFlag, flags, names,
                             index, mapperAllocas)))
    return failure();

  return success();
}

/// Process data operands from acc::UpdateOp
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::UpdateOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  unsigned index = 0;

  // Host operands are handled as `from` call.
  // Device operands are handled as `to` call.
  llvm::SmallVector<mlir::Value> from, to;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto getDevicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(
            dataOp.getDefiningOp())) {
      from.push_back(getDevicePtrOp.getVarPtr());
    } else if (auto updateDeviceOp =
                   mlir::dyn_cast_or_null<acc::UpdateDeviceOp>(
                       dataOp.getDefiningOp())) {
      to.push_back(updateDeviceOp.getVarPtr());
    }
  }

  if (failed(processOperands(builder, moduleTranslation, op, from, from.size(),
                             kHostCopyoutFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  if (failed(processOperands(builder, moduleTranslation, op, to, to.size(),
                             kDeviceCopyinFlag, flags, names, index,
                             mapperAllocas)))
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

  llvm::Function *beginMapperFunc = accBuilder->getOrCreateRuntimeFunctionPtr(
      llvm::omp::OMPRTL___tgt_target_data_begin_mapper);

  llvm::Function *endMapperFunc = accBuilder->getOrCreateRuntimeFunctionPtr(
      llvm::omp::OMPRTL___tgt_target_data_end_mapper);

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

  // TODO handle no_create, deviceptr and attach operands.

  llvm::SmallVector<mlir::Value> copyin, copyout, create, present,
      deleteOperands;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(
            dataOp.getDefiningOp())) {
      for (auto &u : devicePtrOp.getAccPtr().getUses()) {
        if (mlir::dyn_cast_or_null<acc::DeleteOp>(u.getOwner())) {
          deleteOperands.push_back(devicePtrOp.getVarPtr());
        } else if (mlir::dyn_cast_or_null<acc::CopyoutOp>(u.getOwner())) {
          // TODO copyout zero currenlty handled as copyout. Update when
          // extension available.
          copyout.push_back(devicePtrOp.getVarPtr());
        }
      }
    } else if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(
                   dataOp.getDefiningOp())) {
      // TODO copyin readonly currenlty handled as copyin. Update when extension
      // available.
      copyin.push_back(copyinOp.getVarPtr());
    } else if (auto createOp = mlir::dyn_cast_or_null<acc::CreateOp>(
                   dataOp.getDefiningOp())) {
      // TODO create zero currenlty handled as create. Update when extension
      // available.
      create.push_back(createOp.getVarPtr());
    } else if (auto presentOp = mlir::dyn_cast_or_null<acc::PresentOp>(
                   dataOp.getDefiningOp())) {
      present.push_back(createOp.getVarPtr());
    }
  }

  auto nbTotalOperands = copyin.size() + copyout.size() + create.size() +
                         present.size() + deleteOperands.size();

  // Copyin operands are handled as `to` call.
  if (failed(processOperands(builder, moduleTranslation, op, copyin,
                             nbTotalOperands, kDeviceCopyinFlag | kHoldFlag,
                             flags, names, index, mapperAllocas)))
    return failure();

  // Delete operands are handled as `delete` call.
  if (failed(processOperands(builder, moduleTranslation, op, deleteOperands,
                             nbTotalOperands, kDeleteFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  // Copyout operands are handled as `from` call.
  if (failed(processOperands(builder, moduleTranslation, op, copyout,
                             nbTotalOperands, kHostCopyoutFlag | kHoldFlag,
                             flags, names, index, mapperAllocas)))
    return failure();

  // Create operands are handled as `alloc` call.
  if (failed(processOperands(builder, moduleTranslation, op, create,
                             nbTotalOperands, kCreateFlag | kHoldFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  if (failed(processOperands(builder, moduleTranslation, op, present,
                             nbTotalOperands, kPresentFlag | kHoldFlag, flags,
                             names, index, mapperAllocas)))
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

  // Create call to start the data region.
  accBuilder->emitMapperCall(builder.saveIP(), beginMapperFunc, srcLocInfo,
                             maptypesArg, mapnamesArg, mapperAllocas,
                             kDefaultDevice, totalNbOperand);

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

  llvm::UncondBrInst *sourceTerminator = builder.CreateBr(entryBlock);

  builder.restoreIP(afterDataRegion);
  llvm::BasicBlock *endDataBlock = llvm::BasicBlock::Create(
      ctx, "acc.end_data", builder.GetInsertBlock()->getParent());

  SetVector<Block *> blocks = getBlocksSortedByDominance(op.getRegion());
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    if (bb->isEntryBlock()) {
      assert(sourceTerminator->getNumSuccessors() == 1 &&
             "provided entry block has multiple successors");
      sourceTerminator->setSuccessor(0, llvmBB);
    }

    if (failed(
            moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder))) {
      return failure();
    }

    if (isa<acc::TerminatorOp, acc::YieldOp>(bb->getTerminator()))
      builder.CreateBr(endDataBlock);
  }

  // Create call to end the data region.
  builder.SetInsertPoint(endDataBlock);
  accBuilder->emitMapperCall(builder.saveIP(), endMapperFunc, srcLocInfo,
                             maptypesArg, mapnamesArg, mapperAllocas,
                             kDefaultDevice, totalNbOperand);

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
  auto *mapperFunc = getAssociatedFunction(*accBuilder, op);

  // Number of arguments in the enter_data operation.
  unsigned totalNbOperand = op.getNumDataOperands();

  llvm::LLVMContext &ctx = builder.getContext();

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

  accBuilder->emitMapperCall(builder.saveIP(), mapperFunc, srcLocInfo,
                             maptypesArg, mapnamesArg, mapperAllocas,
                             kDefaultDevice, totalNbOperand);

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
      .Case([&](acc::EnterDataOp enterDataOp) {
        return convertStandaloneDataOp<acc::EnterDataOp>(enterDataOp, builder,
                                                         moduleTranslation);
      })
      .Case([&](acc::ExitDataOp exitDataOp) {
        return convertStandaloneDataOp<acc::ExitDataOp>(exitDataOp, builder,
                                                        moduleTranslation);
      })
      .Case([&](acc::UpdateOp updateOp) {
        return convertStandaloneDataOp<acc::UpdateOp>(updateOp, builder,
                                                      moduleTranslation);
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
      .Case<acc::TerminatorOp, acc::YieldOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        assert(op->getNumOperands() == 0 &&
               "unexpected OpenACC terminator with operands");
        return success();
      })
      .Case<acc::CreateOp, acc::CopyinOp, acc::CopyoutOp, acc::DeleteOp,
            acc::UpdateDeviceOp, acc::GetDevicePtrOp>([](auto op) {
        // NOP
        return success();
      })
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
