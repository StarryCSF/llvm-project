! Verify that -fopenacc enables the native OpenACC pipeline by
! default and that --disable-acc-pipeline disables all OpenACC passes.

! RUN: %flang_fc1 -fopenacc -emit-llvm \
! RUN:   -mmlir --mlir-print-ir-before=acc-declare-action-conversion,acc-implicit-routine,fir-acc-recipe-bufferization,acc-recipe-materialization,acc-compute-lowering,acc-cg-to-gpu,gpu-kernel-outlining,convert-openacc-to-scf \
! RUN:   -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -fopenacc -emit-llvm -mmlir --disable-acc-pipeline \
! RUN:   -mmlir --mlir-print-ir-before=acc-initialize-fir-analyses,acc-declare-action-conversion,fir-acc-recipe-bufferization,acc-recipe-materialization,acc-compute-lowering,acc-cg-to-gpu,gpu-kernel-outlining,convert-openacc-to-scf \
! RUN:   -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=DISABLED --allow-empty
! RUN: %flang_fc1 -emit-llvm \
! RUN:   -mmlir --mlir-print-ir-before=acc-initialize-fir-analyses,acc-declare-action-conversion,fir-acc-recipe-bufferization,acc-recipe-materialization,acc-compute-lowering,acc-cg-to-gpu,gpu-kernel-outlining,convert-openacc-to-scf \
! RUN:   -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NO-OPENACC --allow-empty

! DEFAULT: IR Dump Before ACCDeclareActionConversion
! DEFAULT: IR Dump Before ACCImplicitRoutine
! DEFAULT: IR Dump Before ACCRecipeBufferization
! DEFAULT: IR Dump Before ACCRecipeMaterialization
! DEFAULT: IR Dump Before ACCComputeLowering
! DEFAULT: IR Dump Before ACCCGToGPU
! DEFAULT: IR Dump Before GpuKernelOutliningPass
! DEFAULT: IR Dump Before ConvertOpenACCToSCFPass
! DEFAULT-NOT: ACCDeclareActionConversion
! DEFAULT-NOT: ACCRecipeBufferization
! DEFAULT-NOT: ACCRecipeMaterialization
! DEFAULT-NOT: ConvertOpenACCToSCFPass

! DISABLED-NOT: ACCInitializeFIRAnalyses
! DISABLED-NOT: ACCDeclareActionConversion
! DISABLED-NOT: ACCRecipeBufferization
! DISABLED-NOT: ACCRecipeMaterialization
! DISABLED-NOT: ACCComputeLowering
! DISABLED-NOT: ACCCGToGPU
! DISABLED-NOT: GpuKernelOutliningPass
! DISABLED-NOT: ConvertOpenACCToSCFPass

! NO-OPENACC-NOT: ACCInitializeFIRAnalyses
! NO-OPENACC-NOT: ACCDeclareActionConversion
! NO-OPENACC-NOT: ACCRecipeBufferization
! NO-OPENACC-NOT: ACCRecipeMaterialization
! NO-OPENACC-NOT: ACCComputeLowering
! NO-OPENACC-NOT: ACCCGToGPU
! NO-OPENACC-NOT: GpuKernelOutliningPass
! NO-OPENACC-NOT: ConvertOpenACCToSCFPass

end program
