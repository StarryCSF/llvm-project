//===-- CodeGenOpenACC.h - OpenACC codegen -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_CODEGENOPENACC_H
#define FORTRAN_OPTIMIZER_CODEGEN_CODEGENOPENACC_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir {
class LLVMTypeConverter;

void populateOpenACCFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

} // namespace fir

namespace mlir {

class ConversionTarget;
class LLVMTypeConverter;

void configureOpenACCToLLVMConversionLegality(
    ConversionTarget &target, const LLVMTypeConverter &typeConverter);

void populateOpenACCToLLVMConversionPatterns(fir::LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);

} // namespace mlir

#endif // FORTRAN_OPTIMIZER_CODEGEN_CODEGENOPENACC_H
