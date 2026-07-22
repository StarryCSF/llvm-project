! Verify that both recipe passes are registered and the disable flag works.

! RUN: %flang_fc1 -emit-llvm -mmlir --mlir-print-ir-before=fir-acc-recipe-bufferization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=BUFFERIZATION
! RUN: %flang_fc1 -emit-llvm -mmlir --mlir-print-ir-before=acc-recipe-materialization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=MATERIALIZATION
! RUN: %flang_fc1 -emit-llvm -mmlir --disable-recipe -mmlir --mlir-print-ir-before=fir-acc-recipe-bufferization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NO-BUFFERIZATION --allow-empty
! RUN: %flang_fc1 -emit-llvm -mmlir --disable-recipe -mmlir --mlir-print-ir-before=acc-recipe-materialization -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=NO-MATERIALIZATION --allow-empty

! BUFFERIZATION: IR Dump Before ACCRecipeBufferization
! MATERIALIZATION: IR Dump Before ACCRecipeMaterialization
! NO-BUFFERIZATION-NOT: ACCRecipeBufferization
! NO-MATERIALIZATION-NOT: ACCRecipeMaterialization

end program
