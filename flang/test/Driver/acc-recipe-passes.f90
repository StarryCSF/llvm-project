! Verify that both recipe passes are enabled only for OpenACC compilation and
! that the disable flag works.

! RUN: %flang_fc1 -fopenacc -emit-llvm -mmlir --mlir-print-ir-before=fir-acc-recipe-bufferization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BUFFERIZATION
! RUN: %flang_fc1 -fopenacc -emit-llvm -mmlir --mlir-print-ir-before=acc-recipe-materialization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=MATERIALIZATION
! RUN: %flang_fc1 -emit-llvm -mmlir --mlir-print-ir-before=fir-acc-recipe-bufferization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NO-OPENACC-BUFFERIZATION --allow-empty
! RUN: %flang_fc1 -emit-llvm -mmlir --mlir-print-ir-before=acc-recipe-materialization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NO-OPENACC-MATERIALIZATION --allow-empty
! RUN: %flang_fc1 -fopenacc -emit-llvm -mmlir --disable-recipe -mmlir --mlir-print-ir-before=fir-acc-recipe-bufferization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=DISABLED-BUFFERIZATION --allow-empty
! RUN: %flang_fc1 -fopenacc -emit-llvm -mmlir --disable-recipe -mmlir --mlir-print-ir-before=acc-recipe-materialization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=DISABLED-MATERIALIZATION --allow-empty

! BUFFERIZATION: IR Dump Before ACCRecipeBufferization
! MATERIALIZATION: IR Dump Before ACCRecipeMaterialization
! NO-OPENACC-BUFFERIZATION-NOT: ACCRecipeBufferization
! NO-OPENACC-MATERIALIZATION-NOT: ACCRecipeMaterialization
! DISABLED-BUFFERIZATION-NOT: ACCRecipeBufferization
! DISABLED-MATERIALIZATION-NOT: ACCRecipeMaterialization

end program
