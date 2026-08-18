//===- ACCPipeline.cpp - OpenACC flang pass pipelines ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

mlir::acc::ACCCGToGPUOptions
getCodegenOptions(const fir::acc::OpenACCFlangPipelineOptions &options) {
  mlir::acc::ACCCGToGPUOptions codegenOptions;
  codegenOptions.deviceType = options.deviceType;
  codegenOptions.maxWorkgroupSharedMemory = options.maxWorkgroupSharedMemory;
  codegenOptions.maxThreadPrivateStack = options.maxThreadPrivateStack;
  codegenOptions.subgroupSize = options.subgroupSize;
  return codegenOptions;
}

} // namespace

namespace fir::acc {

void buildOpenACCFlangPipeline(OpPassManager &pm,
                               const OpenACCFlangPipelineOptions &options) {
  pm.addPass(createACCInitializeFIRAnalysesPass());
  pm.addPass(createACCDeclareActionConversion());

  mlir::acc::ACCImplicitRoutineOptions implicitRoutineOptions;
  implicitRoutineOptions.deviceType = options.deviceType;
  pm.addPass(mlir::acc::createACCImplicitRoutine(implicitRoutineOptions));

  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createOffloadLiveInValueCanonicalization());
  pm.addPass(mlir::acc::createACCImplicitDeclare());
  pm.addNestedPass<func::FuncOp>(mlir::acc::createACCIfClauseLowering());
  pm.addNestedPass<func::FuncOp>(createACCUseDeviceCanonicalizerPass());

  // Refresh FIR alias analysis after use_device canonicalization and
  // immediately before implicit data generation.
  pm.addPass(createACCInitializeFIRAnalysesPass());

  mlir::acc::ACCImplicitDataOptions implicitDataOptions;
  implicitDataOptions.enableImplicitReductionCopy =
      options.enableImplicitReductionCopy;
  implicitDataOptions.ignoreDefaultNone = options.ignoreDefaultNone;
  pm.addPass(mlir::acc::createACCImplicitData(implicitDataOptions));

  mlir::acc::LegalizeDataValuesInRegionOptions legalizeDataOptions;
  legalizeDataOptions.hostToDevice = true;
  legalizeDataOptions.applyToAccDataConstruct = true;
  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createLegalizeDataValuesInRegion(legalizeDataOptions));

  // Verify while data, private, and reduction clause operations still carry
  // the mapping evidence consumed by the verifier. Recipe materialization
  // removes that evidence as it inlines the recipes.
  mlir::acc::OffloadTargetVerifierOptions verifierOptions;
  verifierOptions.deviceType = options.deviceType;
  verifierOptions.softCheck = false;
  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createOffloadTargetVerifier(verifierOptions));

  if (options.emitRemarks)
    pm.addNestedPass<func::FuncOp>(mlir::acc::createACCEmitRemarksPrivate());

  // Recipe materialization only supports reductions attached to parallel and
  // loop constructs. Normalize serial before consuming recipes.
  pm.addNestedPass<func::FuncOp>(mlir::acc::createACCLegalizeSerial());
  pm.addPass(createACCRecipeBufferizationPass());
  pm.addPass(mlir::acc::createACCRecipeMaterialization());
  pm.addNestedPass<func::FuncOp>(createACCOptimizeFirstprivateMapPass());

  pm.addNestedPass<func::FuncOp>(mlir::acc::createACCLoopTiling());

  mlir::acc::ACCRoutineLoweringOptions routineLoweringOptions;
  routineLoweringOptions.deviceType = options.deviceType;
  pm.addPass(mlir::acc::createACCRoutineLowering(routineLoweringOptions));

  mlir::acc::ACCSpecializeForHostOptions hostSpecializationOptions;
  hostSpecializationOptions.enableHostFallback = false;
  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createACCSpecializeForHost(hostSpecializationOptions));

  mlir::acc::ACCSpecializeForDeviceOptions deviceSpecializationOptions;
  // Public OpenACC device codes: acc_device_not_host and acc_device_nvidia.
  deviceSpecializationOptions.theDeviceTypes = {3, 4};
  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createACCSpecializeForDevice(deviceSpecializationOptions));

  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createOffloadLiveInValueCanonicalization());

  mlir::acc::ACCComputeLoweringOptions computeOptions;
  computeOptions.deviceType = options.deviceType;
  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createACCComputeLowering(computeOptions));

  if (options.emitRemarks)
    pm.addNestedPass<func::FuncOp>(mlir::acc::createACCEmitRemarksLoop());

  mlir::acc::ACCBindRoutineOptions bindOptions;
  bindOptions.deviceType = options.deviceType;
  pm.addNestedPass<func::FuncOp>(mlir::acc::createACCBindRoutine(bindOptions));

  OpPassManager &deviceBindPM = pm.nest<gpu::GPUModuleOp>();
  deviceBindPM.addNestedPass<gpu::GPUFuncOp>(
      mlir::acc::createACCBindRoutine(bindOptions));

  pm.addPass(mlir::acc::createACCDeclareGPUModuleInsertion());

  mlir::acc::ACCRoutineToGPUFuncOptions placementOptions;
  placementOptions.deviceType = options.deviceType;
  pm.addPass(mlir::acc::createACCRoutineToGPUFunc(placementOptions));

  // Specialized routines are regular gpu.func operations after placement and
  // must be lowered in place before outlining.
  OpPassManager &deviceCodegenPM = pm.nest<gpu::GPUModuleOp>();
  deviceCodegenPM.addNestedPass<gpu::GPUFuncOp>(
      mlir::acc::createACCCGToGPU(getCodegenOptions(options)));

  // Lower host compute regions only after routine placement, so staged
  // routines are not mistaken for ordinary host compute constructs.
  pm.addNestedPass<func::FuncOp>(
      mlir::acc::createACCCGToGPU(getCodegenOptions(options)));

  mlir::GpuKernelOutliningPassOptions outliningOptions;
  outliningOptions.dataLayoutStr = options.dataLayoutStr;
  pm.addPass(mlir::createGpuKernelOutliningPass(outliningOptions));

  pm.addPass(mlir::createConvertOpenACCToSCFPass());
}

void registerOpenACCFlangPipelines() {
  PassPipelineRegistration<OpenACCFlangPipelineOptions>(
      "fir-acc-pipeline",
      "Prepare Flang OpenACC constructs and lower them to the current "
      "intermediate device representation",
      buildOpenACCFlangPipeline);
}

} // namespace fir::acc
