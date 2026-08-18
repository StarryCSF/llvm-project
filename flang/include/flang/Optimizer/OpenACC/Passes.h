//===- Passes.h - OpenACC pass entry points -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the OpenACC passes specific to Fortran and FIR.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENACC_PASSES_H
#define FORTRAN_OPTIMIZER_OPENACC_PASSES_H

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"

#include <memory>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace fir {
namespace acc {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/OpenACC/Passes.h.inc"

std::unique_ptr<mlir::Pass> createACCInitializeFIRAnalysesPass();
std::unique_ptr<mlir::Pass> createACCOptimizeFirstprivateMapPass();
std::unique_ptr<mlir::Pass> createACCRecipeBufferizationPass();
std::unique_ptr<mlir::Pass> createACCUseDeviceCanonicalizerPass();

/// Options for the Flang OpenACC pipeline.
struct OpenACCFlangPipelineOptions
    : public mlir::PassPipelineOptions<OpenACCFlangPipelineOptions> {
  PassOptions::Option<mlir::acc::DeviceType> deviceType{
      *this, "device-type", llvm::cl::desc("Target OpenACC device type"),
      llvm::cl::init(mlir::acc::DeviceType::Nvidia),
      llvm::cl::values(
          clEnumValN(mlir::acc::DeviceType::Nvidia, "nvidia", "NVIDIA GPU"))};

  PassOptions::Option<bool> emitRemarks{
      *this, "emit-remarks",
      llvm::cl::desc("Emit OpenACC private and loop mapping remarks"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> enableImplicitReductionCopy{
      *this, "enable-implicit-reduction-copy",
      llvm::cl::desc("Use implicit copy for reduction variables"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> ignoreDefaultNone{
      *this, "ignore-default-none",
      llvm::cl::desc("Generate implicit data under verified default(none)"),
      llvm::cl::init(false)};

  PassOptions::Option<int64_t> maxWorkgroupSharedMemory{
      *this, "max-workgroup-shared-memory",
      llvm::cl::desc("Maximum workgroup shared memory budget in bytes"),
      llvm::cl::init(49152)};
  PassOptions::Option<int64_t> maxThreadPrivateStack{
      *this, "max-thread-private-stack",
      llvm::cl::desc("Maximum thread-private stack allocation in bytes"),
      llvm::cl::init(16384)};
  PassOptions::Option<int64_t> subgroupSize{
      *this, "subgroup-size",
      llvm::cl::desc("Subgroup size used for GPU dimension alignment"),
      llvm::cl::init(32)};
  PassOptions::Option<std::string> dataLayoutStr{
      *this, "data-layout-str",
      llvm::cl::desc("Data layout attached to outlined GPU modules"),
      llvm::cl::init("")};
};

/// Build the Flang OpenACC pipeline.
void buildOpenACCFlangPipeline(mlir::OpPassManager &pm,
                               const OpenACCFlangPipelineOptions &options);

/// Register the Flang OpenACC pipeline.
void registerOpenACCFlangPipelines();

} // namespace acc
} // namespace fir

#endif // FORTRAN_OPTIMIZER_OPENACC_PASSES_H
