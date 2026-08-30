//===- FIROpenACCTypeInterfaces.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains external dialect interfaces for FIR.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_
#define FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_

#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace fir::acc {

template <typename T>
struct OpenACCPointerLikeModel
    : public mlir::acc::PointerLikeType::ExternalModel<
          OpenACCPointerLikeModel<T>, T> {
  mlir::Type getElementType(mlir::Type pointer) const {
    // For types like !fir.ref<!fir.box<...>>, we need to unwrap all the
    // layers to get the actual element type.
    mlir::Type eleTy = mlir::cast<T>(pointer).getElementType();
    // Keep unwrapping while the element type is still pointer-like or a box.
    while (true) {
      if (mlir::isa<mlir::acc::PointerLikeType>(eleTy)) {
        eleTy = mlir::cast<mlir::acc::PointerLikeType>(eleTy).getElementType();
      } else if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(eleTy)) {
        // For box types, extract the element type from the box content.
        eleTy = fir::unwrapRefType(boxTy.getEleTy());
      } else {
        break;
      }
    }
    return eleTy;
  }
  mlir::acc::VariableTypeCategory
  getPointeeTypeCategory(mlir::Type pointer,
                         mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
                         mlir::Type varType) const;

  mlir::Value genAllocate(mlir::Type pointer, mlir::OpBuilder &builder,
                          mlir::Location loc, llvm::StringRef varName,
                          mlir::Type varType, mlir::Value originalVar,
                          bool &needsFree) const;

  bool genFree(mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
               mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
               mlir::Value allocRes, mlir::Type varType) const;

  bool genCopy(mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
               mlir::TypedValue<mlir::acc::PointerLikeType> destination,
               mlir::TypedValue<mlir::acc::PointerLikeType> source,
               mlir::Type varType) const;

  mlir::Value genLoad(mlir::Type pointer, mlir::OpBuilder &builder,
                      mlir::Location loc,
                      mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
                      mlir::Type valueType) const;

  bool genStore(mlir::Type pointer, mlir::OpBuilder &builder,
                mlir::Location loc, mlir::Value valueToStore,
                mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const;

  mlir::Value genCast(mlir::Type pointer, mlir::OpBuilder &builder,
                      mlir::Location loc, mlir::Value value,
                      mlir::Type resultType) const;

  mlir::MemRefType getAsMemRefType(mlir::Type pointer,
                                   mlir::ModuleOp module) const;

  bool isDeviceData(mlir::Type pointer, mlir::Value var) const;
};

template <typename T>
struct OpenACCMappableModel
    : public mlir::acc::MappableType::ExternalModel<OpenACCMappableModel<T>,
                                                    T> {
  mlir::TypedValue<mlir::acc::PointerLikeType> getVarPtr(::mlir::Type type,
                                                         mlir::Value var) const;

  std::optional<llvm::TypeSize>
  getSizeInBytes(mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
                 const mlir::DataLayout &dataLayout) const;

  std::optional<int64_t>
  getOffsetInBytes(mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
                   const mlir::DataLayout &dataLayout) const;

  bool hasUnknownDimensions(mlir::Type type) const;

  llvm::SmallVector<mlir::Value>
  generateAccBounds(mlir::Type type, mlir::Value var,
                    mlir::OpBuilder &builder) const;

  mlir::acc::VariableTypeCategory getTypeCategory(mlir::Type type,
                                                  mlir::Value var) const;

  mlir::acc::VariableInfoAttr
  genPrivateVariableInfo(mlir::Type type,
                         mlir::TypedValue<mlir::acc::MappableType> var) const;

  mlir::Value generatePrivateInit(mlir::Type type, mlir::OpBuilder &builder,
                                  mlir::Location loc,
                                  mlir::TypedValue<mlir::acc::MappableType> var,
                                  llvm::StringRef varName,
                                  mlir::ValueRange extents, mlir::Value initVal,
                                  mlir::acc::VariableInfoAttr varInfo,
                                  bool &needsDestroy) const;

  bool generatePrivateDestroy(mlir::Type type, mlir::OpBuilder &builder,
                              mlir::Location loc, mlir::Value privatized,
                              mlir::ValueRange bounds,
                              mlir::acc::VariableInfoAttr varInfo) const;

  bool generateCopy(mlir::Type type, mlir::OpBuilder &mlirBuilder,
                    mlir::Location loc,
                    mlir::TypedValue<mlir::acc::MappableType> source,
                    mlir::TypedValue<mlir::acc::MappableType> dest,
                    mlir::ValueRange bounds,
                    mlir::acc::VariableInfoAttr varInfo) const;

  bool generateCombiner(mlir::Type type, mlir::OpBuilder &mlirBuilder,
                        mlir::Location loc,
                        mlir::TypedValue<mlir::acc::MappableType> dest,
                        mlir::TypedValue<mlir::acc::MappableType> source,
                        mlir::ValueRange bounds,
                        mlir::acc::ReductionOperator op,
                        mlir::Attribute fastmathFlags) const;

  bool isDeviceData(mlir::Type type, mlir::Value var) const;
};

struct OpenACCReducibleLogicalModel
    : public mlir::acc::ReducibleType::ExternalModel<
          OpenACCReducibleLogicalModel, fir::LogicalType> {
  std::optional<mlir::arith::AtomicRMWKind>
  getAtomicRMWKind(mlir::Type type, mlir::acc::ReductionOperator redOp) const;
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_
