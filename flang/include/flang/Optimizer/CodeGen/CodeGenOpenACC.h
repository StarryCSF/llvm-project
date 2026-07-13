//===------- Optimizer/CodeGen/CodeGenOpenACC.h - OpenACC codegen -*- C++ -*-===//
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
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
class LLVMTypeConverter;

/// Configure the legality of OpenACC operations for conversion to LLVM.
/// OpenACC data clause operations are legal when their operand and result
/// types are LLVM types.
void configureOpenACCToLLVMConversionLegality(
    mlir::ConversionTarget &target, const LLVMTypeConverter &typeConverter);

/// Specialised conversion patterns of OpenACC operations for FIR to LLVM
/// dialect, utilised in cases where the default OpenACC dialect handling cannot
/// handle all cases for intermingled fir types and operations.
void populateOpenACCFIRToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_CODEGENOPENACC_H