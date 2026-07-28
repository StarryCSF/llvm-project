//===- ACCPipeline.cpp - OpenACC flang pass pipelines ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace fir::acc {

void populateFIRCodeGenOpenACCPassPipeline(mlir::PassManager &pm) {
  pm.addPass(createACCRecipeBufferizationPass());
  pm.addPass(mlir::acc::createACCRecipeMaterialization());
}

} // namespace fir::acc
