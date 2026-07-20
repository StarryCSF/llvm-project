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
#include "llvm/ADT/STLExtras.h"
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
      module.getOrInsertFunction(
          "__tgt_acc_data_enter",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
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
      module.getOrInsertFunction(
          "__tgt_acc_data_exit",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
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
      module.getOrInsertFunction(
          "__tgt_acc_data_begin",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
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
      module.getOrInsertFunction(
          "__tgt_acc_data_end",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
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
      module.getOrInsertFunction(
          "__tgt_acc_data_update",
          llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
              {identTy, i64Ty, i64Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty},
              false))
          .getCallee());
}

/// Get or create __tgt_acc_get_deviceptr used by host_data use_device.
static llvm::Function *getAccGetDevicePtrFunction(llvm::Module &module,
                                                    llvm::LLVMContext &ctx) {
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i64Ty = llvm::Type::getInt64Ty(ctx);
  return llvm::cast<llvm::Function>(
      module.getOrInsertFunction(
          "__tgt_acc_get_deviceptr",
          llvm::FunctionType::get(ptrTy, {ptrTy, ptrTy, i64Ty, ptrTy}, false))
          .getCallee());
}

/// Get or create acc_is_present used by host_data if_present.
static llvm::Function *getAccIsPresentFunction(llvm::Module &module,
                                                llvm::LLVMContext &ctx) {
  auto *ptrTy = llvm::PointerType::getUnqual(ctx);
  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  return llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction("acc_is_present",
                               llvm::FunctionType::get(i32Ty, {ptrTy}, false))
          .getCallee());
}

//===----------------------------------------------------------------------===//
// OpenACC Data Operation Call Helper
//===----------------------------------------------------------------------===//

/// Emit call to OpenACC data runtime function.
static void emitAccDataCall(llvm::IRBuilderBase &builder, llvm::Function *fn,
                            llvm::Value *srcLocInfo, llvm::Value *flagsArg,
                            llvm::Value *deviceTypeArg, llvm::Value *argNumVal,
                            llvm::Value *argsBasePtr, llvm::Value *argsPtr,
                            llvm::Value *argSizesPtr, llvm::Value *maptypesArg,
                            llvm::Value *mapnamesArg, llvm::Value *asyncVal) {
  auto *nullPtr = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(builder.getContext()));
  
  builder.CreateCall(fn, {srcLocInfo, flagsArg, deviceTypeArg, argNumVal,
                          argsBasePtr, argsPtr, argSizesPtr, maptypesArg,
                          mapnamesArg, nullPtr, nullPtr, asyncVal});
}

static llvm::Value *createSourceLocationInfo(OpenACCIRBuilder &builder,
                                             Operation *op);

/// Converts acc.use_device into a lookup of the device address for its host
/// variable. The result is used by operations inside the host_data region.
static LogicalResult
convertUseDeviceOp(acc::UseDeviceOp op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *varPtr = moduleTranslation.lookupValue(op.getVarPtr());
  if (!varPtr) {
    op.emitError("could not find LLVM value for varPtr");
    return failure();
  }

  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();
  llvm::Value *srcLocInfo = createSourceLocationInfo(
      *moduleTranslation.getOpenMPBuilder(), op);
  llvm::Value *nullPtr = llvm::ConstantPointerNull::get(
      llvm::PointerType::getUnqual(ctx));
  llvm::Value *devicePtr = builder.CreateCall(
      getAccGetDevicePtrFunction(*module, ctx),
      {srcLocInfo, nullPtr, builder.getInt64(0), varPtr});
  moduleTranslation.mapValue(op.getAccPtr(), devicePtr);
  return success();
}

/// Converts an acc.host_data region. The region itself does not create a new
/// mapping; use_device operations inside it retrieve pointers from the active
/// OpenACC data environment.
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

  if (cond)
    builder.CreateCondBr(cond, entryBlock, endBlock);
  else {
    builder.CreateBr(entryBlock);
  }

  SetVector<Block *> blocks = getBlocksSortedByDominance(op.getRegion());
  for (Block *bb : blocks) {
    if (failed(moduleTranslation.convertBlock(*bb, bb->isEntryBlock(),
                                              builder)))
      return failure();
    if (isa<acc::TerminatorOp, acc::YieldOp>(bb->getTerminator()))
      builder.CreateBr(endBlock);
  }

  builder.SetInsertPoint(endBlock);
  return success();
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Flag values are extracted from openmp/libomptarget/include/omptarget.h and
/// mapped to corresponding OpenACC flags.
/// TGT_ACC_MAPTYPE flags from Interface.h:
/// TO = 0x1, FROM = 0x2, ALLOC = 0x4, DELETE = 0x8, PTR_AND_OBJ = 0x10
static constexpr uint64_t kCreateFlag = 0x004;  // ALLOC
static constexpr uint64_t kDeviceCopyinFlag = 0x001;  // TO
static constexpr uint64_t kHostCopyoutFlag = 0x002;   // FROM
static constexpr uint64_t kDeleteFlag = 0x008;        // DELETE
static constexpr uint64_t kPtrAndObjFlag = 0x010;     // PTR_AND_OBJ
static constexpr uint64_t kPresentFlag = 0x1000;
// Runtime extension to implement the OpenACC second reference counter.
static constexpr uint64_t kHoldFlag = 0x2000;
// OMP_TGT_MAPTYPE_ATTACH from omptarget.h - used for attach/detach.
static constexpr uint64_t kAttachFlag = 0x4000;
// no_create: no ALLOC/TO/FROM flags, just PTR_AND_OBJ to register presence.
static constexpr uint64_t kNoCreateFlag = 0x0;
// Device pointer - already mapped on device.
static constexpr uint64_t kDevicePtrFlag = 0x400;  // DEVPTR

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
                                             Operation *op,
                                             llvm::Module &module,
                                             llvm::LLVMContext &ctx) {
  return llvm::TypeSwitch<Operation *, llvm::Function *>(op)
      .Case([&](acc::EnterDataOp) {
        return getAccDataEnterFunction(module, ctx);
      })
      .Case([&](acc::DeclareEnterOp) {
        return getAccDataEnterFunction(module, ctx);
      })
      .Case([&](acc::ExitDataOp) {
        return getAccDataExitFunction(module, ctx);
      })
      .Case([&](acc::DeclareExitOp) {
        return getAccDataExitFunction(module, ctx);
      })
      .Case([&](acc::UpdateOp) {
        return getAccDataUpdateFunction(module, ctx);
      });
  llvm_unreachable("Unknown OpenACC operation");
}

/// Return the data-clause result that produced `var` for an OpenACC
/// operation. The mapper receives the underlying FIR/LLVM pointer, while the
/// data-entry operation carries the original variable type needed to decide
/// whether PTR_AND_OBJ applies.
static ValueRange getDataOperandsForMapping(Operation *op) {
  if (auto kernelEnv = dyn_cast<acc::KernelEnvironmentOp>(op))
    return kernelEnv.getDataClauseOperands();
  return acc::getDataOperands(op);
}

static Operation *findDataEntryForMapping(Operation *op, Value var) {
  for (Value dataOperand : getDataOperandsForMapping(op)) {
    Operation *entry = dataOperand.getDefiningOp();
    if (entry && acc::getVar(entry) == var)
      return entry;
  }
  for (Operation *user : var.getUsers()) {
    if (acc::getVar(user) == var && acc::getVarType(user))
      return user;
  }
  return nullptr;
}

/// Scalar data operands are passed directly to the runtime. Arrays and
/// pointer-like objects retain PTR_AND_OBJ and use the runtime's dynamic size
/// calculation.
static bool getScalarMappingSize(Operation *parent, Value var,
                                 LLVM::ModuleTranslation &moduleTranslation,
                                 uint64_t &size) {
  Operation *entry = findDataEntryForMapping(parent, var);
  Type varType = entry ? acc::getVarType(entry) : Type();
  if (!varType) {
    // Recipe materialization can leave a scalar private temporary as an
    // LLVM alloca live into the launch. Such a temporary is mapped as a
    // short-lived create entry by the kernel environment lowering.
    if (auto alloca = var.getDefiningOp<LLVM::AllocaOp>())
      varType = alloca.getElemType();
  }
  if (!varType)
    return false;

  // Aggregate FIR types are sized dynamically by the mapper. Only builtin
  // scalar types can be converted directly at this stage.
  if (!isa<IntegerType, FloatType>(varType))
    return false;

  llvm::Type *llvmVarType = moduleTranslation.convertType(varType);
  if (!llvmVarType ||
      !(llvmVarType->isIntegerTy() || llvmVarType->isFloatingPointTy()))
    return false;

  llvm::TypeSize typeSize =
      moduleTranslation.getLLVMModule()->getDataLayout().getTypeAllocSize(
          llvmVarType);
  if (typeSize.isScalable())
    return false;
  size = typeSize.getFixedValue();
  return true;
}

/// Return the allocation size of the mapped object and its innermost element.
/// The data-clause varType attribute has already been converted to an LLVM
/// dialect type by FIR lowering, so the LLVM module data layout can size it.
static bool getMappingTypeSizes(Operation *parent, Value var,
                                LLVM::ModuleTranslation &moduleTranslation,
                                uint64_t &elementSize,
                                uint64_t &aggregateSize) {
  Type varType = acc::getVarType(parent);
  if (!varType) {
    Operation *entry = findDataEntryForMapping(parent, var);
    varType = entry ? acc::getVarType(entry) : Type();
  }
  if (!varType)
    return false;

  llvm::Type *llvmType = moduleTranslation.convertType(varType);
  if (!llvmType || !llvmType->isSized())
    return false;

  const llvm::DataLayout &dataLayout =
      moduleTranslation.getLLVMModule()->getDataLayout();
  llvm::Type *elementType = llvmType;
  while (auto *arrayType = llvm::dyn_cast<llvm::ArrayType>(elementType))
    elementType = arrayType->getElementType();
  llvm::TypeSize elementTypeSize = dataLayout.getTypeAllocSize(elementType);
  llvm::TypeSize objectTypeSize = dataLayout.getTypeAllocSize(llvmType);
  if (elementTypeSize.isScalable() || objectTypeSize.isScalable())
    return false;
  elementSize = elementTypeSize.getFixedValue();
  aggregateSize = objectTypeSize.getFixedValue();
  return elementSize != 0;
}

static llvm::Value *getI64BoundValue(llvm::IRBuilderBase &builder,
                                     LLVM::ModuleTranslation &moduleTranslation,
                                     Value value, int64_t defaultValue) {
  llvm::Value *result = value ? moduleTranslation.lookupValue(value) : nullptr;
  if (!result)
    return builder.getInt64(defaultValue);
  if (!result->getType()->isIntegerTy(64))
    result = builder.CreateIntCast(result, builder.getInt64Ty(), true);
  return result;
}

/// Form the host pointer used as the mapping key for a bounded data entry.
/// The data entry result is also passed to kernel argument lowering, so it
/// must identify the same section that processOperands maps in the runtime.
static llvm::Value *getBoundedMappingPointer(
    Operation *entry, llvm::Value *dataValue,
    LLVM::ModuleTranslation &moduleTranslation,
    llvm::IRBuilderBase &builder) {
  SmallVector<Value> bounds = acc::getBounds(entry);
  if (bounds.empty())
    return dataValue;

  uint64_t elementSize = 0;
  uint64_t aggregateSize = 0;
  if (!getMappingTypeSizes(entry, {}, moduleTranslation, elementSize,
                           aggregateSize))
    return dataValue;

  llvm::Value *offset = builder.getInt64(0);
  bool validBounds = true;
  for (Value bound : bounds) {
    auto boundsOp = bound.getDefiningOp<acc::DataBoundsOp>();
    if (!boundsOp) {
      validBounds = false;
      break;
    }

    llvm::Value *lower = getI64BoundValue(
        builder, moduleTranslation, boundsOp.getLowerbound(), 0);
    llvm::Value *stride = getI64BoundValue(
        builder, moduleTranslation, boundsOp.getStride(), 1);
    if (!boundsOp.getStrideInBytes())
      stride = builder.CreateMul(stride, builder.getInt64(elementSize));
    offset = builder.CreateAdd(offset, builder.CreateMul(lower, stride));
  }

  if (!validBounds)
    return dataValue;
  return builder.CreateGEP(builder.getInt8Ty(), dataValue, {offset});
}

/// A bounded section is passed to the runtime as an ordinary memory range.
/// PTR_AND_OBJ is reserved for mappings that attach a pointer field in a
/// parent object; using it for an array section makes the runtime interpret
/// the first array bytes as a pointer descriptor.
static uint64_t getMappingFlag(Operation *parent, Value var,
                               uint64_t flag, bool isScalar) {
  Operation *entry = findDataEntryForMapping(parent, var);
  bool hasBounds = entry && !acc::getBounds(entry).empty();
  return (isScalar || hasBounds) ? (flag & ~kPtrAndObjFlag) : flag;
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
    uint64_t scalarSize = 0;
    bool isScalar = getScalarMappingSize(op, data, moduleTranslation,
                                         scalarSize);
    uint64_t elementSize = 0;
    uint64_t aggregateSize = 0;
    bool hasTypeSizes = getMappingTypeSizes(op, data, moduleTranslation,
                                            elementSize, aggregateSize);
    Operation *entry = findDataEntryForMapping(op, data);
    SmallVector<Value> bounds = entry ? acc::getBounds(entry)
                                      : SmallVector<Value>();
    uint64_t mappingFlag = getMappingFlag(op, data, operandFlag, isScalar);

    if (isa<LLVM::LLVMPointerType>(data.getType())) {
      dataPtrBase = dataValue;
      dataPtr = dataValue;
      if (isScalar) {
        dataSize = builder.getInt64(scalarSize);
      } else if (!bounds.empty() && hasTypeSizes) {
        // Bounds are ordered from the innermost Fortran dimension outward.
        // Compute the byte offset of the first selected element and the byte
        // span through the last selected element. The latter is the range the
        // OpenACC runtime expects for a contiguous mapped section.
        llvm::Value *offset = builder.getInt64(0);
        llvm::Value *span = builder.getInt64(elementSize);
        bool validBounds = true;
        for (Value bound : bounds) {
          auto boundsOp = bound.getDefiningOp<acc::DataBoundsOp>();
          if (!boundsOp) {
            validBounds = false;
            break;
          }
          llvm::Value *lower = getI64BoundValue(
              builder, moduleTranslation, boundsOp.getLowerbound(), 0);
          llvm::Value *extent;
          if (boundsOp.getUpperbound()) {
            llvm::Value *upper = getI64BoundValue(
                builder, moduleTranslation, boundsOp.getUpperbound(), 0);
            extent = builder.CreateAdd(
                builder.CreateSub(upper, lower), builder.getInt64(1));
          } else {
            extent = getI64BoundValue(
                builder, moduleTranslation, boundsOp.getExtent(), 0);
          }
          llvm::Value *stride = getI64BoundValue(
              builder, moduleTranslation, boundsOp.getStride(), 1);
          if (!boundsOp.getStrideInBytes())
            stride = builder.CreateMul(stride, builder.getInt64(elementSize));
          offset = builder.CreateAdd(offset, builder.CreateMul(lower, stride));
          span = builder.CreateAdd(
              span, builder.CreateMul(
                        builder.CreateSub(extent, builder.getInt64(1)),
                        stride));
        }
        if (validBounds) {
          dataPtr = builder.CreateGEP(builder.getInt8Ty(), dataValue,
                                      {offset});
          dataSize = span;
        } else {
          dataSize = hasTypeSizes ? builder.getInt64(aggregateSize)
                                  : accBuilder->getSizeInBytes(dataValue);
        }
      } else if (hasTypeSizes) {
        dataSize = builder.getInt64(aggregateSize);
      } else {
        dataSize = accBuilder->getSizeInBytes(dataValue);
      }
      if (!(mappingFlag & kPtrAndObjFlag))
        dataPtrBase = llvm::ConstantPointerNull::get(
            llvm::PointerType::getUnqual(ctx));
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

    flags.push_back(mappingFlag);
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
  unsigned index = 0;

  // Create operands are handled as `alloc` call.
  // Copyin operands are handled as `to` call.
  // Attach operands are handled as `attach` call.
  // Create_zero operands are handled as `alloc` call with PTR_AND_OBJ.
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

  // Create operands are handled as `alloc` call.
  if (failed(processOperands(builder, moduleTranslation, op, create,
                             nbTotalOperands, kCreateFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Copyin operands are handled as `to` call.
  if (failed(processOperands(builder, moduleTranslation, op, copyin,
                             nbTotalOperands, kDeviceCopyinFlag | kPtrAndObjFlag, flags, names,
                             index, mapperAllocas)))
    return failure();

  // Attach operands are handled as `attach` call.
  if (failed(processOperands(builder, moduleTranslation, op, attachOperands,
                             nbTotalOperands, kAttachFlag | kPtrAndObjFlag, flags, names,
                             index, mapperAllocas)))
    return failure();

  // Create_zero operands are handled as `alloc` call with PTR_AND_OBJ.
  if (failed(processOperands(builder, moduleTranslation, op, createZeroOperands,
                             nbTotalOperands, kCreateFlag | kPtrAndObjFlag, flags, names,
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
  unsigned index = 0;

  llvm::SmallVector<mlir::Value> deleteOperands, copyoutOperands,
      detachOperands;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(
            dataOp.getDefiningOp())) {
      for (auto &u : devicePtrOp.getAccPtr().getUses()) {
        if (mlir::dyn_cast_or_null<acc::DeleteOp>(u.getOwner()))
          deleteOperands.push_back(devicePtrOp.getVarPtr());
        else if (mlir::dyn_cast_or_null<acc::CopyoutOp>(u.getOwner()))
          copyoutOperands.push_back(devicePtrOp.getVarPtr());
        else if (mlir::dyn_cast_or_null<acc::DetachOp>(u.getOwner()))
          detachOperands.push_back(devicePtrOp.getVarPtr());
      }
    }
  }

  auto nbTotalOperands = deleteOperands.size() + copyoutOperands.size() +
                         detachOperands.size();

  // Delete operands are handled as `delete` call.
  if (failed(processOperands(builder, moduleTranslation, op, deleteOperands,
                             nbTotalOperands, kDeleteFlag | kPtrAndObjFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  // Copyout operands are handled as `from` call.
  if (failed(processOperands(builder, moduleTranslation, op, copyoutOperands,
                             nbTotalOperands, kHostCopyoutFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Detach operands are handled as `detach` call (attach + delete + PTR_AND_OBJ).
  if (failed(processOperands(builder, moduleTranslation, op, detachOperands,
                             nbTotalOperands, kAttachFlag | kDeleteFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
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
                             kHostCopyoutFlag | kPtrAndObjFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  if (failed(processOperands(builder, moduleTranslation, op, to, to.size(),
                             kDeviceCopyinFlag | kPtrAndObjFlag, flags, names, index,
                             mapperAllocas)))
    return failure();
  return success();
}

/// Process data operands from acc::DeclareEnterOp. Declare uses the same
/// runtime entry point as enter_data, but its verifier permits a smaller set
/// of data entry operations and also includes device-resident/link entries.
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::DeclareEnterOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  unsigned index = 0;
  llvm::SmallVector<mlir::Value> copyin, create, present, deviceptr;

  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    Operation *entry = dataOp.getDefiningOp();
    if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(entry))
      copyin.push_back(copyinOp.getVarPtr());
    else if (auto createOp = mlir::dyn_cast_or_null<acc::CreateOp>(entry))
      create.push_back(createOp.getVarPtr());
    else if (auto presentOp = mlir::dyn_cast_or_null<acc::PresentOp>(entry))
      present.push_back(presentOp.getVarPtr());
    else if (mlir::isa_and_nonnull<acc::DeclareDeviceResidentOp>(entry))
      create.push_back(acc::getVarPtr(entry));
    else if (mlir::isa_and_nonnull<acc::DeclareLinkOp>(entry))
      copyin.push_back(acc::getVarPtr(entry));
    else if (mlir::isa_and_nonnull<acc::DevicePtrOp, acc::GetDevicePtrOp>(entry))
      deviceptr.push_back(acc::getVarPtr(entry));
  }

  unsigned totalNbOperand = copyin.size() + create.size() + present.size() +
                            deviceptr.size();
  if (failed(processOperands(builder, moduleTranslation, op, copyin,
                             totalNbOperand,
                             kDeviceCopyinFlag | kPtrAndObjFlag, flags, names,
                             index, mapperAllocas)))
    return failure();
  if (failed(processOperands(builder, moduleTranslation, op, create,
                             totalNbOperand, kCreateFlag | kPtrAndObjFlag,
                             flags, names, index, mapperAllocas)))
    return failure();
  if (failed(processOperands(builder, moduleTranslation, op, present,
                             totalNbOperand, kPresentFlag | kHoldFlag, flags,
                             names, index, mapperAllocas)))
    return failure();
  if (failed(processOperands(builder, moduleTranslation, op, deviceptr,
                             totalNbOperand, kDevicePtrFlag | kPtrAndObjFlag,
                             flags, names, index, mapperAllocas)))
    return failure();
  return success();
}

/// Process data operands from acc::DeclareExitOp. A direct declare data entry
/// is released as delete unless its acc pointer has an explicit exit action.
static LogicalResult
processDataOperands(llvm::IRBuilderBase &builder,
                    LLVM::ModuleTranslation &moduleTranslation,
                    acc::DeclareExitOp op, SmallVector<uint64_t> &flags,
                    SmallVectorImpl<llvm::Constant *> &names,
                    struct OpenACCIRBuilder::MapperAllocas &mapperAllocas) {
  unsigned index = 0;
  llvm::SmallVector<mlir::Value> deleteOperands, copyoutOperands,
      detachOperands;

  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    Operation *entry = dataOp.getDefiningOp();
    if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(entry)) {
      for (auto &u : devicePtrOp.getAccPtr().getUses()) {
        if (mlir::isa<acc::DeleteOp>(u.getOwner()))
          deleteOperands.push_back(devicePtrOp.getVarPtr());
        else if (mlir::isa<acc::CopyoutOp>(u.getOwner()))
          copyoutOperands.push_back(devicePtrOp.getVarPtr());
        else if (mlir::isa<acc::DetachOp>(u.getOwner()))
          detachOperands.push_back(devicePtrOp.getVarPtr());
      }
    } else if (entry && acc::getVarPtr(entry)) {
      bool hasExplicitAction = false;
      for (auto &u : acc::getAccPtr(entry).getUses()) {
        if (mlir::isa<acc::DeleteOp>(u.getOwner())) {
          deleteOperands.push_back(acc::getVarPtr(entry));
          hasExplicitAction = true;
        } else if (mlir::isa<acc::CopyoutOp>(u.getOwner())) {
          copyoutOperands.push_back(acc::getVarPtr(entry));
          hasExplicitAction = true;
        } else if (mlir::isa<acc::DetachOp>(u.getOwner())) {
          detachOperands.push_back(acc::getVarPtr(entry));
          hasExplicitAction = true;
        }
      }
      if (!hasExplicitAction)
        deleteOperands.push_back(acc::getVarPtr(entry));
    }
  }

  unsigned totalNbOperand = deleteOperands.size() + copyoutOperands.size() +
                            detachOperands.size();
  if (failed(processOperands(builder, moduleTranslation, op, deleteOperands,
                             totalNbOperand, kDeleteFlag | kPtrAndObjFlag,
                             flags, names, index, mapperAllocas)))
    return failure();
  if (failed(processOperands(builder, moduleTranslation, op, copyoutOperands,
                             totalNbOperand,
                             kHostCopyoutFlag | kPtrAndObjFlag, flags, names,
                             index, mapperAllocas)))
    return failure();
  if (failed(processOperands(builder, moduleTranslation, op, detachOperands,
                             totalNbOperand,
                             kAttachFlag | kDeleteFlag | kPtrAndObjFlag,
                             flags, names, index, mapperAllocas)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion functions for init/shutdown/set/wait
//===----------------------------------------------------------------------===//

/// Converts acc.init operation into LLVM IR.
static LogicalResult convertInitOp(acc::InitOp op,
                                    llvm::IRBuilderBase &builder,
                                    LLVM::ModuleTranslation &moduleTranslation) {
  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);
  auto *fn = getAccInitFunction(*module, ctx);

  // Map MLIR DeviceType enum to OpenACC runtime acc_device_t
  auto mapDeviceType = [](int64_t mlirType) -> int64_t {
    switch (mlirType) {
    case 0: return 0;  // None -> none
    case 1: return 0;  // Star -> none
    case 2: return 1;  // Default -> default
    case 3: return 2;  // Host -> host
    case 4: return 3;  // Multicore -> not_host
    case 5: return 4;  // Nvidia -> nvidia
    case 6: return 5;  // Radeon -> amd
    default: return 1; // Unknown -> default
    }
  };

  int64_t deviceType = 1; // default
  if (op.getDeviceTypes()) {
    auto dtypes = op.getDeviceTypes()->getValue();
    if (!dtypes.empty()) {
      int64_t mlirType = static_cast<int64_t>(
          mlir::cast<mlir::acc::DeviceTypeAttr>(dtypes[0]).getValue());
      deviceType = mapDeviceType(mlirType);
    }
  }

  llvm::Value *deviceNumVal = builder.getInt64(-1);
  if (op.getDeviceNum()) {
    deviceNumVal = moduleTranslation.lookupValue(op.getDeviceNum());
    if (deviceNumVal->getType() != llvm::Type::getInt64Ty(ctx))
      deviceNumVal = builder.CreateZExt(deviceNumVal, llvm::Type::getInt64Ty(ctx));
  }

  builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                          builder.getInt64(deviceType), deviceNumVal});
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

  // Map MLIR DeviceType enum to OpenACC runtime acc_device_t
  auto mapDeviceType = [](int64_t mlirType) -> int64_t {
    switch (mlirType) {
    case 0: return 0;  // None -> none
    case 1: return 0;  // Star -> none
    case 2: return 1;  // Default -> default
    case 3: return 2;  // Host -> host
    case 4: return 3;  // Multicore -> not_host
    case 5: return 4;  // Nvidia -> nvidia
    case 6: return 5;  // Radeon -> amd
    default: return 1; // Unknown -> default
    }
  };

  int64_t deviceType = 1; // default
  if (op.getDeviceTypes()) {
    auto dtypes = op.getDeviceTypes()->getValue();
    if (!dtypes.empty()) {
      int64_t mlirType = static_cast<int64_t>(
          mlir::cast<mlir::acc::DeviceTypeAttr>(dtypes[0]).getValue());
      deviceType = mapDeviceType(mlirType);
    }
  }

  llvm::Value *deviceNumVal = builder.getInt64(-1);
  if (op.getDeviceNum()) {
    deviceNumVal = moduleTranslation.lookupValue(op.getDeviceNum());
    if (deviceNumVal->getType() != llvm::Type::getInt64Ty(ctx))
      deviceNumVal = builder.CreateZExt(deviceNumVal, llvm::Type::getInt64Ty(ctx));
  }

  builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                          builder.getInt64(deviceType), deviceNumVal});
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

  // Map MLIR DeviceType enum to OpenACC runtime acc_device_t
  // MLIR: None=0, Star=1, Default=2, Host=3, Multicore=4, Nvidia=5, Radeon=6
  // Runtime: none=0, default=1, host=2, not_host=3, nvidia=4, amd=5, spirv=6
  auto mapDeviceType = [](int64_t mlirType) -> int64_t {
    switch (mlirType) {
    case 0: return 0;  // None -> none
    case 1: return 0;  // Star -> none (not used)
    case 2: return 1;  // Default -> default
    case 3: return 2;  // Host -> host
    case 4: return 3;  // Multicore -> not_host
    case 5: return 4;  // Nvidia -> nvidia
    case 6: return 5;  // Radeon -> amd
    default: return 1; // Unknown -> default
    }
  };

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
      deviceType = mapDeviceType(static_cast<int64_t>(*op.getDeviceType()));
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                           builder.getInt64(deviceType), deviceNumVal});
  } else if (op.getDeviceType()) {
    auto *fn = getAccSetDeviceTypeFunction(*module, ctx);
    int64_t rtDeviceType = mapDeviceType(static_cast<int64_t>(*op.getDeviceType()));
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                           builder.getInt64(rtDeviceType)});
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

  int64_t deviceType = 0;
  int32_t deviceNum = -1;
  if (op.getWaitDevnum()) {
    llvm::Value *devnumVal = moduleTranslation.lookupValue(op.getWaitDevnum());
    if (devnumVal->getType() != llvm::Type::getInt32Ty(ctx))
      devnumVal = builder.CreateZExt(devnumVal, llvm::Type::getInt32Ty(ctx));
  }

  uint32_t waitNum = op.getWaitOperands().size();

  llvm::Value *waitListPtr = llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx));
  if (waitNum > 0) {
    auto *arrTy = llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), waitNum);
    auto *waitList = builder.CreateAlloca(arrTy);
    for (uint32_t i = 0; i < waitNum; ++i) {
      llvm::Value *waitVal = moduleTranslation.lookupValue(op.getWaitOperands()[i]);
      if (waitVal->getType() != llvm::Type::getInt64Ty(ctx))
        waitVal = builder.CreateZExt(waitVal, llvm::Type::getInt64Ty(ctx));
      auto *gep = builder.CreateInBoundsGEP(arrTy, waitList,
                                            {builder.getInt32(0), builder.getInt32(i)});
      builder.CreateStore(waitVal, gep);
    }
    waitListPtr = waitList;
  }

  int64_t asyncVal = -1;
  if (op.getAsync() && op.getAsyncOperand()) {
    llvm::Value *asyncValL = moduleTranslation.lookupValue(op.getAsyncOperand());
    if (asyncValL->getType() != llvm::Type::getInt64Ty(ctx))
      asyncValL = builder.CreateZExt(asyncValL, llvm::Type::getInt64Ty(ctx));
    builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                           builder.getInt64(deviceType), builder.getInt32(deviceNum),
                           builder.getInt32(waitNum), waitListPtr, asyncValL});
    return success();
  }

  builder.CreateCall(fn, {srcLocInfo, builder.getInt64(0),
                         builder.getInt64(deviceType), builder.getInt32(deviceNum),
                         builder.getInt32(waitNum), waitListPtr,
                         builder.getInt64(asyncVal)});
  return success();
}

static LogicalResult
convertAccAtomicUpdate(acc::AtomicUpdateOp &opInst,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation);

/// Convert an OpenACC atomic read through the common LLVM atomic builder.
static LogicalResult
convertAccAtomicRead(acc::AtomicReadOp &opInst,
                     llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *llvmX = moduleTranslation.lookupValue(opInst.getX());
  llvm::Value *llvmV = moduleTranslation.lookupValue(opInst.getV());
  if (!llvmX || !llvmV)
    return opInst.emitError("could not find LLVM value for atomic read");
  llvm::Type *elementType =
      moduleTranslation.convertType(opInst.getElementType());
  if (!elementType)
    return opInst.emitError("could not convert atomic read element type");

  auto enclosingFuncOp =
      opInst.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());
  if (builder.GetInsertBlock() == &enclosingFunction->getEntryBlock() &&
      builder.GetInsertPoint() == builder.GetInsertBlock()->end()) {
    llvm::BasicBlock *bodyBlock = llvm::BasicBlock::Create(
        builder.getContext(), "acc.atomic.body", enclosingFunction,
        enclosingFunction->getEntryBlock().getNextNode());
    builder.CreateBr(bodyBlock);
    builder.SetInsertPoint(bodyBlock);
  }
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  llvm::OpenMPIRBuilder::LocationDescription location(builder);
  llvm::OpenMPIRBuilder::AtomicOpValue x = {
      llvmX, elementType, /*isSigned=*/false, /*isVolatile=*/false};
  llvm::OpenMPIRBuilder::AtomicOpValue v = {
      llvmV, elementType, /*isSigned=*/false, /*isVolatile=*/false};
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createAtomicRead(
      location, x, v, llvm::AtomicOrdering::Monotonic, allocaIP));
  return success();
}

/// Convert an OpenACC atomic write through the common LLVM atomic builder.
static LogicalResult
convertAccAtomicWrite(acc::AtomicWriteOp &opInst,
                      llvm::IRBuilderBase &builder,
                      LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Value *llvmX = moduleTranslation.lookupValue(opInst.getX());
  llvm::Value *llvmExpr = moduleTranslation.lookupValue(opInst.getExpr());
  if (!llvmX || !llvmExpr)
    return opInst.emitError("could not find LLVM value for atomic write");
  llvm::Type *elementType =
      moduleTranslation.convertType(opInst.getExpr().getType());
  if (!elementType)
    return opInst.emitError("could not convert atomic write element type");

  auto enclosingFuncOp =
      opInst.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());
  if (builder.GetInsertBlock() == &enclosingFunction->getEntryBlock() &&
      builder.GetInsertPoint() == builder.GetInsertBlock()->end()) {
    llvm::BasicBlock *bodyBlock = llvm::BasicBlock::Create(
        builder.getContext(), "acc.atomic.body", enclosingFunction,
        enclosingFunction->getEntryBlock().getNextNode());
    builder.CreateBr(bodyBlock);
    builder.SetInsertPoint(bodyBlock);
  }
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  llvm::OpenMPIRBuilder::LocationDescription location(builder);
  llvm::OpenMPIRBuilder::AtomicOpValue x = {
      llvmX, elementType, /*isSigned=*/false, /*isVolatile=*/false};
  builder.restoreIP(moduleTranslation.getOpenMPBuilder()->createAtomicWrite(
      location, x, llvmExpr, llvm::AtomicOrdering::Monotonic,
      allocaIP));
  return success();
}

/// Convert the supported OpenACC atomic capture forms by lowering the nested
/// atomic operations in their specified order.
static LogicalResult
convertAccAtomicCapture(acc::AtomicCaptureOp &opInst,
                        llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  bool converted = false;
  for (Operation &nestedOp : opInst.getRegion().front()) {
    if (auto readOp = dyn_cast<acc::AtomicReadOp>(&nestedOp)) {
      if (failed(convertAccAtomicRead(readOp, builder, moduleTranslation)))
        return failure();
    } else if (auto writeOp = dyn_cast<acc::AtomicWriteOp>(&nestedOp)) {
      if (failed(convertAccAtomicWrite(writeOp, builder, moduleTranslation)))
        return failure();
    } else if (auto updateOp = dyn_cast<acc::AtomicUpdateOp>(&nestedOp)) {
      if (failed(convertAccAtomicUpdate(updateOp, builder, moduleTranslation)))
        return failure();
    } else if (!isa<acc::TerminatorOp>(&nestedOp)) {
      return opInst.emitError("unsupported atomic capture operation");
    } else {
      continue;
    }
    converted = true;
  }
  return converted ? success()
                   : opInst.emitError("unsupported atomic capture form");
}

/// Convert an OpenACC atomic update through the common LLVM atomic builder.
/// The update region is also used for non-trivial expressions, in which case
/// the builder emits a compare-exchange loop instead of atomicrmw.
static LogicalResult
convertAccAtomicUpdate(acc::AtomicUpdateOp &opInst,
                       llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  auto &innerOpList = opInst.getRegion().front().getOperations();
  bool isXBinopExpr = false;
  llvm::AtomicRMWInst::BinOp binop = llvm::AtomicRMWInst::BinOp::BAD_BINOP;
  mlir::Value mlirExpr;
  llvm::Value *llvmExpr = nullptr;

  if (innerOpList.size() == 2) {
    Operation &innerOp = *opInst.getRegion().front().begin();
    BlockArgument regionArg = opInst.getRegion().front().getArgument(0);
    if (!llvm::is_contained(innerOp.getOperands(), regionArg))
      return opInst.emitError(
          "no atomic update operation with region argument as operand found");

    binop = llvm::TypeSwitch<Operation *, llvm::AtomicRMWInst::BinOp>(&innerOp)
                .Case([&](LLVM::AddOp) {
                  return llvm::AtomicRMWInst::BinOp::Add;
                })
                .Case([&](LLVM::SubOp) {
                  return llvm::AtomicRMWInst::BinOp::Sub;
                })
                .Case([&](LLVM::AndOp) {
                  return llvm::AtomicRMWInst::BinOp::And;
                })
                .Case([&](LLVM::OrOp) {
                  return llvm::AtomicRMWInst::BinOp::Or;
                })
                .Case([&](LLVM::XOrOp) {
                  return llvm::AtomicRMWInst::BinOp::Xor;
                })
                .Case([&](LLVM::UMaxOp) {
                  return llvm::AtomicRMWInst::BinOp::UMax;
                })
                .Case([&](LLVM::UMinOp) {
                  return llvm::AtomicRMWInst::BinOp::UMin;
                })
                .Case([&](LLVM::FAddOp) {
                  return llvm::AtomicRMWInst::BinOp::FAdd;
                })
                .Case([&](LLVM::FSubOp) {
                  return llvm::AtomicRMWInst::BinOp::FSub;
                })
                .Default(llvm::AtomicRMWInst::BinOp::BAD_BINOP);
    isXBinopExpr = innerOp.getOperand(0) == regionArg;
    mlirExpr = isXBinopExpr ? innerOp.getOperand(1) : innerOp.getOperand(0);
    llvmExpr = moduleTranslation.lookupValue(mlirExpr);
  }

  llvm::Value *llvmX = moduleTranslation.lookupValue(opInst.getX());
  if (!llvmX)
    return opInst.emitError("could not find LLVM value for atomic target");
  llvm::Type *llvmXElementType = moduleTranslation.convertType(
      opInst.getRegion().front().getArgument(0).getType());
  if (!llvmXElementType)
    return opInst.emitError("could not convert atomic element type");

  llvm::OpenMPIRBuilder::AtomicOpValue llvmAtomicX = {
      llvmX, llvmXElementType, /*isSigned=*/false, /*isVolatile=*/false};

  auto updateFn = [&opInst, &moduleTranslation](
                      llvm::Value *atomicx,
                      llvm::IRBuilder<> &nestedBuilder)
      -> llvm::Expected<llvm::Value *> {
    Block &block = opInst.getRegion().front();
    moduleTranslation.mapValue(block.getArgument(0), atomicx);
    moduleTranslation.mapBlock(&block, nestedBuilder.GetInsertBlock());
    if (failed(moduleTranslation.convertBlock(block, true, nestedBuilder)))
      return llvm::make_error<llvm::StringError>(
          "failed to convert atomic update region",
          llvm::inconvertibleErrorCode());
    auto yieldOp = dyn_cast<acc::YieldOp>(block.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return llvm::make_error<llvm::StringError>(
          "atomic update region must yield one value",
          llvm::inconvertibleErrorCode());
    return moduleTranslation.lookupValue(yieldOp.getOperand(0));
  };

  auto enclosingFuncOp =
      opInst.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());
  // OpenMPIRBuilder requires a dedicated insertion point for allocas. During
  // GPU module translation the current insertion point can be the end of the
  // entry block, so split it before passing the entry block to the builder.
  if (builder.GetInsertBlock() == &enclosingFunction->getEntryBlock() &&
      builder.GetInsertPoint() == builder.GetInsertBlock()->end()) {
    llvm::BasicBlock *bodyBlock = llvm::BasicBlock::Create(
        builder.getContext(), "acc.atomic.body", enclosingFunction,
        enclosingFunction->getEntryBlock().getNextNode());
    builder.CreateBr(bodyBlock);
    builder.SetInsertPoint(bodyBlock);
  }
  llvm::OpenMPIRBuilder::InsertPointTy allocaIP(
      &enclosingFunction->getEntryBlock(),
      enclosingFunction->getEntryBlock().getFirstInsertionPt());
  llvm::OpenMPIRBuilder::LocationDescription location(builder);
  llvm::OpenMPIRBuilder::InsertPointOrErrorTy afterIP =
      moduleTranslation.getOpenMPBuilder()->createAtomicUpdate(
          location, allocaIP, llvmAtomicX, llvmExpr,
          llvm::AtomicOrdering::Monotonic, binop, updateFn, isXBinopExpr);
  if (!afterIP) {
    llvm::handleAllErrors(afterIP.takeError(), [&](const llvm::ErrorInfoBase &e) {
      opInst.emitError() << e.message();
    });
    return failure();
  }
  builder.restoreIP(*afterIP);
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
  llvm::Module *module = moduleTranslation.getLLVMModule();
  auto enclosingFuncOp = op.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());

  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();

  llvm::Value *srcLocInfo = createSourceLocationInfo(*accBuilder, op);

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

  llvm::SmallVector<mlir::Value> copyin, copyout, create, present,
      deleteOperands, noCreateOperands, attachOperands, deviceptrOperands;
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
      deviceptrOperands.push_back(devicePtrOp.getVarPtr());
    } else if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::DevicePtrOp>(
                   dataOp.getDefiningOp())) {
      deviceptrOperands.push_back(devicePtrOp.getVarPtr());
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
      present.push_back(presentOp.getVarPtr());
    } else if (auto noCreateOp = mlir::dyn_cast_or_null<acc::NoCreateOp>(
                   dataOp.getDefiningOp())) {
      noCreateOperands.push_back(noCreateOp.getVarPtr());
    } else if (auto attachOp = mlir::dyn_cast_or_null<acc::AttachOp>(
                   dataOp.getDefiningOp())) {
      attachOperands.push_back(attachOp.getVarPtr());
    }
  }

  auto nbTotalOperands = copyin.size() + copyout.size() + create.size() +
                         present.size() + deleteOperands.size() +
                         noCreateOperands.size() + attachOperands.size() +
                         deviceptrOperands.size();

  // Copyin operands are handled as `to` call.
  if (failed(processOperands(builder, moduleTranslation, op, copyin,
                             nbTotalOperands, kDeviceCopyinFlag | kPtrAndObjFlag,
                             flags, names, index, mapperAllocas)))
    return failure();

  // Delete operands are handled as `delete` call.
    if (failed(processOperands(builder, moduleTranslation, op, deleteOperands,
                               nbTotalOperands, kDeleteFlag, flags, names, index,
                               mapperAllocas)))
      return failure();

    // Copyout operands are handled as `from` call.
    if (failed(processOperands(builder, moduleTranslation, op, copyout,
                               nbTotalOperands, kHostCopyoutFlag | kPtrAndObjFlag, flags,
                               names, index, mapperAllocas)))
      return failure();

    // Create operands are handled as `alloc` call.
    if (failed(processOperands(builder, moduleTranslation, op, create,
                               nbTotalOperands, kCreateFlag | kPtrAndObjFlag, flags,
                               names, index, mapperAllocas)))
      return failure();

  if (failed(processOperands(builder, moduleTranslation, op, present,
                             nbTotalOperands, kPresentFlag | kHoldFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // No_create operands - no ALLOC/TO/FROM flags, just PTR_AND_OBJ for presence.
  if (failed(processOperands(builder, moduleTranslation, op, noCreateOperands,
                             nbTotalOperands, kNoCreateFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Attach operands are handled as `attach` call.
  if (failed(processOperands(builder, moduleTranslation, op, attachOperands,
                             nbTotalOperands, kAttachFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Deviceptr operands are already device addresses and must not be copied.
  if (failed(processOperands(builder, moduleTranslation, op, deviceptrOperands,
                             nbTotalOperands, kDevicePtrFlag | kPtrAndObjFlag,
                             flags, names, index, mapperAllocas)))
    return failure();

  SmallVector<uint64_t> endFlags;
  auto appendEndFlag = [&](Value data, uint64_t flag) {
    uint64_t scalarSize = 0;
    bool isScalar = getScalarMappingSize(op, data, moduleTranslation,
                                         scalarSize);
    endFlags.push_back(getMappingFlag(op, data, flag, isScalar));
  };
  auto appendEndFlags = [&](ValueRange operands, uint64_t flag) {
    for (Value data : operands)
      appendEndFlag(data, flag);
  };
  for (Value data : copyin) {
    Operation *entry = findDataEntryForMapping(op, data);
    bool isCopyout = false;
    if (entry) {
      Value accPtr = acc::getAccPtr(entry);
      isCopyout = llvm::any_of(accPtr.getUses(), [](OpOperand &use) {
        return isa<acc::CopyoutOp>(use.getOwner());
      });
    }
    appendEndFlag(data, isCopyout ? (kHostCopyoutFlag | kPtrAndObjFlag)
                                  : kDeleteFlag);
  }
  appendEndFlags(deleteOperands, kDeleteFlag);
  appendEndFlags(copyout, kHostCopyoutFlag | kPtrAndObjFlag);
  for (Value data : create) {
    Operation *entry = findDataEntryForMapping(op, data);
    bool isCopyout = entry &&
                     cast<acc::CreateOp>(entry).getDataClause() ==
                         acc::DataClause::acc_copyout;
    appendEndFlag(data, isCopyout ? (kHostCopyoutFlag | kPtrAndObjFlag)
                                  : (kDeleteFlag | kPtrAndObjFlag));
  }
  appendEndFlags(present, kPresentFlag | kHoldFlag);
  appendEndFlags(noCreateOperands, kNoCreateFlag | kPtrAndObjFlag);
  appendEndFlags(attachOperands, kAttachFlag | kPtrAndObjFlag);
  appendEndFlags(deviceptrOperands, kDevicePtrFlag | kPtrAndObjFlag);

  llvm::GlobalVariable *endMaptypes =
      accBuilder->createOffloadMaptypes(endFlags, ".offload_maptypes_end");
  llvm::Value *endMaptypesArg = builder.CreateConstInBoundsGEP2_32(
      llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
      endMaptypes, /*Idx0=*/0, /*Idx1=*/0);

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
  auto *flagsVal = builder.getInt64(0);
  auto *deviceTypeVal = builder.getInt64(0);  // acc_device_default
  auto *argNumVal = builder.getInt32(totalNbOperand);
  auto *argsBasePtr = mapperAllocas.ArgsBase;
  auto *argsPtr = mapperAllocas.Args;
  auto *argSizesPtr = mapperAllocas.ArgSizes;
  
  emitAccDataCall(builder, beginMapperFunc, srcLocInfo, flagsVal, deviceTypeVal,
                  argNumVal, argsBasePtr, argsPtr, argSizesPtr,
                  maptypesArg, mapnamesArg, builder.getInt64(-1));  // async = -1 (sync)

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
  emitAccDataCall(builder, endMapperFunc, srcLocInfo, flagsVal, deviceTypeVal,
                  argNumVal, argsBasePtr, argsPtr, argSizesPtr,
                  endMaptypesArg, mapnamesArg,
                  builder.getInt64(-1));  // async = -1 (sync)

  return success();
}

/// Converts an OpenACC compute construct (parallel/serial) into LLVM IR.
/// Generates __tgt_acc_data_begin before the region and __tgt_acc_data_end
/// after, with data clause mapping identical to acc.data.
template <typename OpTy>
static LogicalResult convertComputeOp(OpTy &op,
                                      llvm::IRBuilderBase &builder,
                                      LLVM::ModuleTranslation &moduleTranslation) {
  llvm::LLVMContext &ctx = builder.getContext();
  llvm::Module *module = moduleTranslation.getLLVMModule();
  auto enclosingFuncOp = op.getOperation()->template getParentOfType<LLVM::LLVMFuncOp>();
  llvm::Function *enclosingFunction =
      moduleTranslation.lookupFunction(enclosingFuncOp.getName());

  OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
  llvm::Value *srcLocInfo = createSourceLocationInfo(*accBuilder, op);

  llvm::Function *beginMapperFunc = getAccDataBeginFunction(*module, ctx);
  llvm::Function *endMapperFunc = getAccDataEndFunction(*module, ctx);

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

  llvm::SmallVector<mlir::Value> copyin, copyout, create, present,
      deleteOperands, noCreateOperands, attachOperands, deviceptrOperands;
  for (mlir::Value dataOp : op.getDataClauseOperands()) {
    if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(
            dataOp.getDefiningOp())) {
      for (auto &u : devicePtrOp.getAccPtr().getUses()) {
        if (mlir::dyn_cast_or_null<acc::DeleteOp>(u.getOwner())) {
          deleteOperands.push_back(devicePtrOp.getVarPtr());
        } else if (mlir::dyn_cast_or_null<acc::CopyoutOp>(u.getOwner())) {
          copyout.push_back(devicePtrOp.getVarPtr());
        }
      }
      // deviceptr itself
      deviceptrOperands.push_back(devicePtrOp.getVarPtr());
    } else if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(
                   dataOp.getDefiningOp())) {
      copyin.push_back(copyinOp.getVarPtr());
    } else if (auto createOp = mlir::dyn_cast_or_null<acc::CreateOp>(
                   dataOp.getDefiningOp())) {
      create.push_back(createOp.getVarPtr());
    } else if (auto presentOp = mlir::dyn_cast_or_null<acc::PresentOp>(
                   dataOp.getDefiningOp())) {
      present.push_back(presentOp.getVarPtr());
    } else if (auto noCreateOp = mlir::dyn_cast_or_null<acc::NoCreateOp>(
                   dataOp.getDefiningOp())) {
      noCreateOperands.push_back(noCreateOp.getVarPtr());
    } else if (auto attachOp = mlir::dyn_cast_or_null<acc::AttachOp>(
                   dataOp.getDefiningOp())) {
      attachOperands.push_back(attachOp.getVarPtr());
    }
  }

  auto nbTotalOperands = copyin.size() + copyout.size() + create.size() +
                         present.size() + deleteOperands.size() +
                         noCreateOperands.size() + attachOperands.size() +
                         deviceptrOperands.size();

  // Copyin → TO
  if (failed(processOperands(builder, moduleTranslation, op, copyin,
                             nbTotalOperands, kDeviceCopyinFlag | kPtrAndObjFlag,
                             flags, names, index, mapperAllocas)))
    return failure();

  // Delete
  if (failed(processOperands(builder, moduleTranslation, op, deleteOperands,
                             nbTotalOperands, kDeleteFlag, flags, names, index,
                             mapperAllocas)))
    return failure();

  // Copyout → FROM
  if (failed(processOperands(builder, moduleTranslation, op, copyout,
                             nbTotalOperands, kHostCopyoutFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Create → ALLOC
  if (failed(processOperands(builder, moduleTranslation, op, create,
                             nbTotalOperands, kCreateFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Present
  if (failed(processOperands(builder, moduleTranslation, op, present,
                             nbTotalOperands, kPresentFlag | kHoldFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // No_create
  if (failed(processOperands(builder, moduleTranslation, op, noCreateOperands,
                             nbTotalOperands, kNoCreateFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Attach
  if (failed(processOperands(builder, moduleTranslation, op, attachOperands,
                             nbTotalOperands, kAttachFlag | kPtrAndObjFlag, flags,
                             names, index, mapperAllocas)))
    return failure();

  // Deviceptr
  if (failed(processOperands(builder, moduleTranslation, op, deviceptrOperands,
                             nbTotalOperands, kDevicePtrFlag | kPtrAndObjFlag, flags,
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

  auto *flagsVal = builder.getInt64(0);
  auto *deviceTypeVal = builder.getInt64(0);
  auto *argNumVal = builder.getInt32(totalNbOperand);
  auto *argsBasePtr = mapperAllocas.ArgsBase;
  auto *argsPtr = mapperAllocas.Args;
  auto *argSizesPtr = mapperAllocas.ArgSizes;

  // Begin data region
  emitAccDataCall(builder, beginMapperFunc, srcLocInfo, flagsVal, deviceTypeVal,
                  argNumVal, argsBasePtr, argsPtr, argSizesPtr,
                  maptypesArg, mapnamesArg, builder.getInt64(-1));

  // Convert the region body
  llvm::BasicBlock *entryBlock = nullptr;
  for (Block &bb : op.getRegion()) {
    llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
        ctx, "acc.compute", builder.GetInsertBlock()->getParent());
    if (entryBlock == nullptr)
      entryBlock = llvmBB;
    moduleTranslation.mapBlock(&bb, llvmBB);
  }

  auto afterRegion = builder.saveIP();
  llvm::UncondBrInst *sourceTerminator = builder.CreateBr(entryBlock);
  builder.restoreIP(afterRegion);

  llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
      ctx, "acc.end_compute", builder.GetInsertBlock()->getParent());

  SetVector<Block *> blocks = getBlocksSortedByDominance(op.getRegion());
  for (Block *bb : blocks) {
    llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
    if (bb->isEntryBlock()) {
      sourceTerminator->setSuccessor(0, llvmBB);
    }
    if (failed(moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder)))
      return failure();
    if (isa<acc::TerminatorOp, acc::YieldOp>(bb->getTerminator()))
      builder.CreateBr(endBlock);
  }

  // End data region
  builder.SetInsertPoint(endBlock);
  emitAccDataCall(builder, endMapperFunc, srcLocInfo, flagsVal, deviceTypeVal,
                  argNumVal, argsBasePtr, argsPtr, argSizesPtr,
                  maptypesArg, mapnamesArg, builder.getInt64(-1));

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
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::LLVMContext &ctx = builder.getContext();

  auto *srcLocInfo = createSourceLocationInfo(*accBuilder, op);
  auto *mapperFunc = getAssociatedFunction(*accBuilder, op, *module, ctx);

  // Number of arguments in the standalone data operation.
  unsigned totalNbOperand = op.getDataClauseOperands().size();

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

  // Prepare arguments for OpenACC data runtime call
  auto *flagsVal = builder.getInt64(0);
  auto *deviceTypeVal = builder.getInt64(0);  // acc_device_default
  auto *argNumVal = builder.getInt32(totalNbOperand);
  auto *argsBasePtr = mapperAllocas.ArgsBase;
  auto *argsPtr = mapperAllocas.Args;
  auto *argSizesPtr = mapperAllocas.ArgSizes;

  // Emit call to OpenACC data runtime function
  emitAccDataCall(builder, mapperFunc, srcLocInfo, flagsVal, deviceTypeVal,
                  argNumVal, argsBasePtr, argsPtr, argSizesPtr,
                  maptypesArg, mapnamesArg, builder.getInt64(-1));  // async = -1 (sync)

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
      .Case([&](acc::DeclareEnterOp declareEnterOp) {
        return convertStandaloneDataOp<acc::DeclareEnterOp>(
            declareEnterOp, builder, moduleTranslation);
      })
      .Case([&](acc::ExitDataOp exitDataOp) {
        return convertStandaloneDataOp<acc::ExitDataOp>(exitDataOp, builder,
                                                        moduleTranslation);
      })
      .Case([&](acc::DeclareExitOp declareExitOp) {
        return convertStandaloneDataOp<acc::DeclareExitOp>(
            declareExitOp, builder, moduleTranslation);
      })
      .Case([&](acc::UpdateOp updateOp) {
        return convertStandaloneDataOp<acc::UpdateOp>(updateOp, builder,
                                                      moduleTranslation);
      })
      .Case([&](acc::HostDataOp hostDataOp) {
        return convertHostDataOp(hostDataOp, builder, moduleTranslation);
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
      .Case([&](acc::AtomicUpdateOp atomicUpdateOp) {
        return convertAccAtomicUpdate(atomicUpdateOp, builder,
                                      moduleTranslation);
      })
      .Case([&](acc::AtomicReadOp atomicReadOp) {
        return convertAccAtomicRead(atomicReadOp, builder, moduleTranslation);
      })
      .Case([&](acc::AtomicWriteOp atomicWriteOp) {
        return convertAccAtomicWrite(atomicWriteOp, builder, moduleTranslation);
      })
      .Case([&](acc::AtomicCaptureOp atomicCaptureOp) {
        return convertAccAtomicCapture(atomicCaptureOp, builder,
                                       moduleTranslation);
      })
      .Case<acc::TerminatorOp, acc::YieldOp>([](auto op) {
        // `yield` and `terminator` can be just omitted. The block structure was
        // created in the function that handles their parent operation.
        // Atomic update regions use a value-bearing acc.yield; its converted
        // value is consumed by convertAccAtomicUpdate after this block is
        // translated, so it must also remain a translation NOP.
        return success();
      })
      .Case<acc::DataBoundsOp, acc::GetLowerboundOp, acc::GetUpperboundOp,
            acc::GetStrideOp, acc::GetExtentOp>([](auto op) {
        // Bounds are metadata consumed by processOperands above.
        return success();
      })
      .Case<acc::CreateOp, acc::CopyinOp, acc::CopyoutOp, acc::PresentOp,
            acc::NoCreateOp, acc::AttachOp, acc::DeclareDeviceResidentOp,
            acc::DeclareLinkOp, acc::FirstprivateMapInitialOp>(
      [&](auto op) {
            // Data entry ops: map their result (accPtr) to the varPtr's LLVM value.
            llvm::Value *varPtrVal = moduleTranslation.lookupValue(op.getVarPtr());
            if (!varPtrVal) {
              op.emitError("could not find LLVM value for varPtr");
              return failure();
            }
            varPtrVal = getBoundedMappingPointer(op.getOperation(), varPtrVal,
                                                 moduleTranslation, builder);
            if (!moduleTranslation.lookupValue(op.getAccPtr()))
              moduleTranslation.mapValue(op.getAccPtr(), varPtrVal);
            return success();
          })
      .Case<acc::DetachOp>([&](acc::DetachOp op) {
            // DetachOp has getAccVar() instead of getVarPtr()
            llvm::Value *accVarVal = moduleTranslation.lookupValue(op.getAccVar());
            if (!accVarVal) {
              op.emitError("could not find LLVM value for accVar");
              return failure();
            }
            return success();
          })
      .Case<acc::DevicePtrOp>([&](acc::DevicePtrOp op) {
            llvm::Value *varPtrVal = moduleTranslation.lookupValue(op.getVarPtr());
            if (!varPtrVal) {
              op.emitError("could not find LLVM value for varPtr");
              return failure();
            }
            if (!moduleTranslation.lookupValue(op.getAccPtr()))
              moduleTranslation.mapValue(op.getAccPtr(), varPtrVal);
            return success();
          })
      .Case<acc::UseDeviceOp>([&](acc::UseDeviceOp op) {
        return convertUseDeviceOp(op, builder, moduleTranslation);
      })
      .Case<acc::GetDevicePtrOp>([&](acc::GetDevicePtrOp op) {
            llvm::Value *varPtrVal = moduleTranslation.lookupValue(op.getVarPtr());
            if (!varPtrVal) {
              op.emitError("could not find LLVM value for varPtr");
              return failure();
            }
            if (!moduleTranslation.lookupValue(op.getAccPtr()))
              moduleTranslation.mapValue(op.getAccPtr(), varPtrVal);
            return success();
          })
      .Case<acc::DeleteOp, acc::UpdateDeviceOp, acc::UpdateHostOp,
            acc::PrivatizeOp,
            acc::UnwrapPrivateOp, acc::PrivateLocalOp,
            acc::ReductionInitOp,
            acc::ReductionCombineOp, acc::ReductionCombineRegionOp,
            acc::ReductionAccumulateOp, acc::ReductionAccumulateArrayOp,
            acc::GPUSharedMemoryOp>([](auto op) {
        // NOP - these ops are handled by the region body or are
        // intermediate representations consumed by later passes.
        return success();
      })
      .Case<acc::ParWidthOp>(
          [&](acc::ParWidthOp op) {
            // ParWidthOp returns a width value for a GPU dimension.
            // For sequential: return 1, otherwise return the launchArg value
            // or 0 if unknown.
            llvm::Value *result = nullptr;
            if (op.getLaunchArg()) {
              result = moduleTranslation.lookupValue(op.getLaunchArg());
              if (!result) {
                op.emitError("could not find LLVM value for launchArg");
                return failure();
              }
            } else if (op.getParDim().isSeq()) {
              result = builder.getInt64(1);
            } else {
              result = builder.getInt64(0);
            }
            moduleTranslation.mapValue(op.getResult(), result);
            return success();
          })
      .Case<acc::ComputeRegionOp>(
          [&](acc::ComputeRegionOp computeOp) {
            // Handle compute_region by inlining its region body into the
            // current function. The ins operands are mapped to their
            // converted LLVM values as block arguments.
            auto &region = computeOp.getRegion();
            if (region.empty())
              return success();

            llvm::LLVMContext &ctx = builder.getContext();
            llvm::BasicBlock *entryBlock = nullptr;

            // Create LLVM basic blocks for each block in the region.
            for (auto &bb : region) {
              auto *llvmBB = llvm::BasicBlock::Create(ctx, "acc.compute",
                                                      builder.GetInsertBlock()->getParent());
              if (entryBlock == nullptr)
                entryBlock = llvmBB;
              moduleTranslation.mapBlock(&bb, llvmBB);
            }

            auto afterRegion = builder.saveIP();
            llvm::UncondBrInst *sourceTerminator = builder.CreateBr(entryBlock);
            builder.restoreIP(afterRegion);

            llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
                ctx, "acc.end_compute", builder.GetInsertBlock()->getParent());

            // Map ins operands to entry block arguments.
            Block &regionEntry = region.front();
            auto launchArgs = computeOp.getLaunchArgs();
            auto inputArgs = computeOp.getInputArgs();
            unsigned totalArgs = launchArgs.size() + inputArgs.size();
            for (unsigned i = 0; i < totalArgs; ++i) {
              llvm::Value *val = moduleTranslation.lookupValue(
                  computeOp->getOperand(i));
              if (!val) {
                computeOp.emitError("could not find LLVM value for operand ") << i;
                return failure();
              }
              // Only map if not already mapped (e.g. by data entry op handler)
              if (!moduleTranslation.lookupValue(regionEntry.getArgument(i)))
                moduleTranslation.mapValue(regionEntry.getArgument(i), val);
            }

            SetVector<Block *> blocks =
                getBlocksSortedByDominance(region);
            for (Block *bb : blocks) {
              llvm::BasicBlock *llvmBB =
                  moduleTranslation.lookupBlock(bb);
              if (bb->isEntryBlock()) {
                sourceTerminator->setSuccessor(0, llvmBB);
              }

              if (failed(moduleTranslation.convertBlock(
                      *bb, bb->isEntryBlock(), builder)))
                return failure();

              if (isa<acc::YieldOp>(bb->getTerminator()))
                builder.CreateBr(endBlock);
            }

            builder.SetInsertPoint(endBlock);
            return success();
          })
      .Case<acc::PredicateRegionOp>(
          [&](acc::PredicateRegionOp predOp) {
            // PredicateRegion has no operands and NoTerminator.
            // Just inline the region body.
            auto &region = predOp.getRegion();
            if (region.empty())
              return success();

            llvm::LLVMContext &ctx = builder.getContext();
            llvm::BasicBlock *entryBlock = nullptr;

            for (auto &bb : region) {
              auto *llvmBB = llvm::BasicBlock::Create(ctx, "acc.predicate",
                                                      builder.GetInsertBlock()->getParent());
              if (entryBlock == nullptr)
                entryBlock = llvmBB;
              moduleTranslation.mapBlock(&bb, llvmBB);
            }

            auto afterRegion = builder.saveIP();
            llvm::UncondBrInst *sourceTerminator = builder.CreateBr(entryBlock);
            builder.restoreIP(afterRegion);

            llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
                ctx, "acc.end_predicate", builder.GetInsertBlock()->getParent());

            SetVector<Block *> blocks =
                getBlocksSortedByDominance(region);
            for (Block *bb : blocks) {
              llvm::BasicBlock *llvmBB =
                  moduleTranslation.lookupBlock(bb);
              if (bb->isEntryBlock()) {
                sourceTerminator->setSuccessor(0, llvmBB);
              }

              if (failed(moduleTranslation.convertBlock(
                      *bb, bb->isEntryBlock(), builder)))
                return failure();

              // NoTerminator - just branch to end block
              builder.CreateBr(endBlock);
            }

            builder.SetInsertPoint(endBlock);
            return success();
          })
      .Case<acc::KernelEnvironmentOp>(
          [&](acc::KernelEnvironmentOp kernelEnvOp) {
            // KernelEnvironmentOp wraps compute_region with data clauses.
            // Generate __tgt_acc_data_begin/end around the region body.
            llvm::LLVMContext &ctx = builder.getContext();
            llvm::Module *module = moduleTranslation.getLLVMModule();
            auto enclosingFuncOp = kernelEnvOp.getOperation()->getParentOfType<LLVM::LLVMFuncOp>();
            llvm::Function *enclosingFunction =
                moduleTranslation.lookupFunction(enclosingFuncOp.getName());

            OpenACCIRBuilder *accBuilder = moduleTranslation.getOpenMPBuilder();
            llvm::Value *srcLocInfo = createSourceLocationInfo(*accBuilder, kernelEnvOp);

            llvm::Function *beginMapperFunc = getAccDataBeginFunction(*module, ctx);
            llvm::Function *endMapperFunc = getAccDataEndFunction(*module, ctx);

            SmallVector<uint64_t> flags;
            SmallVector<llvm::Constant *> names;
            unsigned index = 0;

            llvm::SmallVector<mlir::Value> copyin, copyout, create, present,
                deleteOperands, noCreateOperands, attachOperands, deviceptrOperands,
                privateOperands;
            llvm::SmallVector<mlir::Value> mappedOperands(
                kernelEnvOp.getDataClauseOperands());
            for (mlir::Value dataOp : kernelEnvOp.getDataClauseOperands()) {
              if (auto devicePtrOp = mlir::dyn_cast_or_null<acc::GetDevicePtrOp>(
                      dataOp.getDefiningOp())) {
                for (auto &u : devicePtrOp.getAccPtr().getUses()) {
                  if (mlir::dyn_cast_or_null<acc::DeleteOp>(u.getOwner()))
                    deleteOperands.push_back(devicePtrOp.getVarPtr());
                  else if (mlir::dyn_cast_or_null<acc::CopyoutOp>(u.getOwner()))
                    copyout.push_back(devicePtrOp.getVarPtr());
                }
                deviceptrOperands.push_back(devicePtrOp.getVarPtr());
              } else if (auto copyinOp = mlir::dyn_cast_or_null<acc::CopyinOp>(
                             dataOp.getDefiningOp())) {
                copyin.push_back(copyinOp.getVarPtr());
              } else if (auto firstprivateOp =
                             mlir::dyn_cast_or_null<acc::FirstprivateMapInitialOp>(
                                 dataOp.getDefiningOp())) {
                // The initial value is copied to device memory and released
                // after the compute region, without a present-counter update.
                copyin.push_back(firstprivateOp.getVarPtr());
              } else if (auto createOp = mlir::dyn_cast_or_null<acc::CreateOp>(
                             dataOp.getDefiningOp())) {
                create.push_back(createOp.getVarPtr());
              } else if (auto presentOp = mlir::dyn_cast_or_null<acc::PresentOp>(
                             dataOp.getDefiningOp())) {
                present.push_back(presentOp.getVarPtr());
              } else if (auto noCreateOp = mlir::dyn_cast_or_null<acc::NoCreateOp>(
                             dataOp.getDefiningOp())) {
                noCreateOperands.push_back(noCreateOp.getVarPtr());
              } else if (auto attachOp = mlir::dyn_cast_or_null<acc::AttachOp>(
                             dataOp.getDefiningOp())) {
                attachOperands.push_back(attachOp.getVarPtr());
              }
            }

            // Private recipe storage is represented by an alloca that is not
            // itself an OpenACC data entry. Map it only for the lifetime of
            // this launch so gpu.launch_func receives a device pointer.
            kernelEnvOp.walk([&](gpu::LaunchFuncOp launchOp) {
              for (mlir::Value arg : launchOp.getKernelOperands()) {
                if (!isa<LLVM::LLVMPointerType>(arg.getType()) ||
                    !arg.getDefiningOp<LLVM::AllocaOp>())
                  continue;
                bool isMapped = llvm::any_of(
                    mappedOperands, [&](mlir::Value dataOp) {
                      return acc::getVar(dataOp.getDefiningOp()) == arg;
                    });
                if (!isMapped && !llvm::is_contained(privateOperands, arg))
                  privateOperands.push_back(arg);
              }
            });

            unsigned totalNbOperand =
                kernelEnvOp.getDataClauseOperands().size() +
                privateOperands.size();
            struct OpenACCIRBuilder::MapperAllocas mapperAllocas;
            OpenACCIRBuilder::InsertPointTy allocaIP(
                &enclosingFunction->getEntryBlock(),
                enclosingFunction->getEntryBlock().getFirstInsertionPt());
            accBuilder->createMapperAllocas(builder.saveIP(), allocaIP,
                                            totalNbOperand, mapperAllocas);

            auto nbTotalOperands = copyin.size() + copyout.size() + create.size() +
                                   present.size() + deleteOperands.size() +
                                   noCreateOperands.size() + attachOperands.size() +
                                   deviceptrOperands.size() + privateOperands.size();

            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, copyin,
                                       nbTotalOperands, kDeviceCopyinFlag | kPtrAndObjFlag,
                                       flags, names, index, mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, deleteOperands,
                                       nbTotalOperands, kDeleteFlag, flags, names, index,
                                       mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, copyout,
                                       nbTotalOperands, kHostCopyoutFlag | kPtrAndObjFlag, flags,
                                       names, index, mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, create,
                                       nbTotalOperands, kCreateFlag | kPtrAndObjFlag, flags,
                                       names, index, mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, present,
                                       nbTotalOperands, kPresentFlag | kHoldFlag, flags,
                                       names, index, mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, noCreateOperands,
                                       nbTotalOperands, kNoCreateFlag | kPtrAndObjFlag, flags,
                                       names, index, mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, attachOperands,
                                       nbTotalOperands, kAttachFlag | kPtrAndObjFlag, flags,
                                       names, index, mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp, deviceptrOperands,
                                       nbTotalOperands, kDevicePtrFlag | kPtrAndObjFlag, flags,
                                       names, index, mapperAllocas)))
              return failure();
            if (failed(processOperands(builder, moduleTranslation, kernelEnvOp,
                                       privateOperands, nbTotalOperands,
                                       kCreateFlag, flags, names, index,
                                       mapperAllocas)))
              return failure();

            SmallVector<uint64_t> endFlags;
            auto appendEndFlags = [&](ValueRange operands, uint64_t flag) {
              for (Value data : operands) {
                uint64_t scalarSize = 0;
                bool isScalar = getScalarMappingSize(
                    kernelEnvOp, data, moduleTranslation, scalarSize);
                endFlags.push_back(getMappingFlag(kernelEnvOp, data, flag,
                                                  isScalar));
              }
            };
            for (Value data : copyin) {
              Operation *entry = findDataEntryForMapping(kernelEnvOp, data);
              bool isCopyout = false;
              if (entry) {
                Value accPtr = acc::getAccPtr(entry);
                isCopyout = llvm::any_of(accPtr.getUses(), [](OpOperand &use) {
                  return isa<acc::CopyoutOp>(use.getOwner());
                });
              }
              uint64_t flag = isCopyout
                                  ? (kHostCopyoutFlag | kPtrAndObjFlag)
                                  : kDeleteFlag;
              uint64_t scalarSize = 0;
              bool isScalar = getScalarMappingSize(
                  kernelEnvOp, data, moduleTranslation, scalarSize);
              endFlags.push_back(
                  getMappingFlag(kernelEnvOp, data, flag, isScalar));
            }
            appendEndFlags(deleteOperands, kDeleteFlag);
            appendEndFlags(copyout, kHostCopyoutFlag | kPtrAndObjFlag);
            for (Value data : create) {
              auto *createOp = findDataEntryForMapping(kernelEnvOp, data);
              bool isCopyout = createOp &&
                               (cast<acc::CreateOp>(createOp).getDataClause() ==
                                    acc::DataClause::acc_copyout ||
                                cast<acc::CreateOp>(createOp).getDataClause() ==
                                    acc::DataClause::acc_copyout_zero);
              uint64_t flag = isCopyout
                                  ? (kHostCopyoutFlag | kPtrAndObjFlag)
                                  : (kDeleteFlag | kPtrAndObjFlag);
              uint64_t scalarSize = 0;
              bool isScalar = getScalarMappingSize(
                  kernelEnvOp, data, moduleTranslation, scalarSize);
              endFlags.push_back(
                  getMappingFlag(kernelEnvOp, data, flag, isScalar));
            }
            appendEndFlags(privateOperands, kDeleteFlag);
            appendEndFlags(present, kPresentFlag | kHoldFlag);
            appendEndFlags(noCreateOperands, kNoCreateFlag | kPtrAndObjFlag);
            appendEndFlags(attachOperands, kAttachFlag | kPtrAndObjFlag);
            appendEndFlags(deviceptrOperands, kDevicePtrFlag | kPtrAndObjFlag);

            llvm::GlobalVariable *maptypes =
                accBuilder->createOffloadMaptypes(flags, ".offload_maptypes");
            llvm::Value *maptypesArg = builder.CreateConstInBoundsGEP2_32(
                llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
                maptypes, 0, 0);

            llvm::GlobalVariable *mapnames =
                accBuilder->createOffloadMapnames(names, ".offload_mapnames");
            llvm::Value *mapnamesArg = builder.CreateConstInBoundsGEP2_32(
                llvm::ArrayType::get(llvm::PointerType::getUnqual(ctx), totalNbOperand),
                mapnames, 0, 0);
            llvm::GlobalVariable *endMaptypes =
                accBuilder->createOffloadMaptypes(endFlags, ".offload_maptypes_end");
            llvm::Value *endMaptypesArg = builder.CreateConstInBoundsGEP2_32(
                llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), totalNbOperand),
                endMaptypes, 0, 0);

            auto *flagsVal = builder.getInt64(0);
            auto *deviceTypeVal = builder.getInt64(0);
            auto *argNumVal = builder.getInt32(totalNbOperand);
            auto *argsBasePtr = mapperAllocas.ArgsBase;
            auto *argsPtr = mapperAllocas.Args;
            auto *argSizesPtr = mapperAllocas.ArgSizes;

            // Determine async value for runtime calls
            llvm::Value *asyncArg = builder.getInt64(-1); // default: synchronous
            if (kernelEnvOp.getAsyncOperand()) {
              llvm::Value *asyncV = moduleTranslation.lookupValue(
                  kernelEnvOp.getAsyncOperand());
              if (!asyncV) {
                kernelEnvOp.emitError("could not find LLVM value for async operand");
                return failure();
              }
              // Convert to i64 if needed
              if (asyncV->getType() != llvm::Type::getInt64Ty(ctx))
                asyncV = builder.CreateZExt(asyncV, llvm::Type::getInt64Ty(ctx));
              asyncArg = asyncV;
            } else if (kernelEnvOp.getAsyncOnly()) {
              // acc parallel async (no arg) = use default async queue
              // __tgt_acc_set_default_async has already been called,
              // pass acc_async_noval (-2) to indicate async without explicit value
              asyncArg = builder.getInt64(-2);
            }

            // Generate wait calls before the data region if wait operands exist
            if (!kernelEnvOp.getWaitOperands().empty()) {
              llvm::Function *waitFn = getAccWaitFunction(*module, ctx);
              for (mlir::Value waitOp : kernelEnvOp.getWaitOperands()) {
                llvm::Value *waitV = moduleTranslation.lookupValue(waitOp);
                if (!waitV) {
                  kernelEnvOp.emitError("could not find LLVM value for wait operand");
                  return failure();
                }
                if (waitV->getType() != llvm::Type::getInt64Ty(ctx))
                  waitV = builder.CreateZExt(waitV, llvm::Type::getInt64Ty(ctx));
                builder.CreateCall(waitFn, {srcLocInfo, waitV});
              }
            }

            // Begin data region
            emitAccDataCall(builder, beginMapperFunc, srcLocInfo, flagsVal,
                            deviceTypeVal, argNumVal, argsBasePtr, argsPtr,
                            argSizesPtr, maptypesArg, mapnamesArg, asyncArg);

            // Convert the region body
            llvm::BasicBlock *entryBlock = nullptr;
            for (Block &bb : kernelEnvOp.getRegion()) {
              llvm::BasicBlock *llvmBB = llvm::BasicBlock::Create(
                  ctx, "acc.kernel_env", builder.GetInsertBlock()->getParent());
              if (entryBlock == nullptr)
                entryBlock = llvmBB;
              moduleTranslation.mapBlock(&bb, llvmBB);
            }

            auto afterRegion = builder.saveIP();
            llvm::UncondBrInst *sourceTerminator = builder.CreateBr(entryBlock);
            builder.restoreIP(afterRegion);

            llvm::BasicBlock *endBlock = llvm::BasicBlock::Create(
                ctx, "acc.end_kernel_env", builder.GetInsertBlock()->getParent());

            SetVector<Block *> blocks = getBlocksSortedByDominance(kernelEnvOp.getRegion());
            for (Block *bb : blocks) {
              llvm::BasicBlock *llvmBB = moduleTranslation.lookupBlock(bb);
              if (bb->isEntryBlock())
                sourceTerminator->setSuccessor(0, llvmBB);
              if (failed(moduleTranslation.convertBlock(*bb, bb->isEntryBlock(), builder)))
                return failure();
              // KernelEnvironmentOp has NoTerminator - just branch to end block
              builder.CreateBr(endBlock);
            }

            // End data region
            builder.SetInsertPoint(endBlock);
            emitAccDataCall(builder, endMapperFunc, srcLocInfo, flagsVal,
                            deviceTypeVal, argNumVal, argsBasePtr, argsPtr,
                            argSizesPtr, endMaptypesArg, mapnamesArg, asyncArg);

            return success();
          })
      .Case<acc::ParallelOp, acc::SerialOp>(
          [&](auto computeOp) {
            return convertComputeOp(computeOp, builder, moduleTranslation);
          })
      .Case<acc::LoopOp>(
          [&](auto computeOp) {
            // Loop op - NOP for now, region body handled by enclosing function.
            return success();
          })
      .Case<acc::RoutineOp>([](auto) {
        // The routine declaration is consumed by the OpenACC routine passes;
        // it carries no runtime behavior into LLVM IR.
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
