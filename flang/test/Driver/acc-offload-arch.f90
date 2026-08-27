! REQUIRES: x86-registered-target, nvptx-registered-target
! UNSUPPORTED: system-windows
!
! RUN: rm -rf %t && split-file %s %t
! RUN: chmod +x %t/single/offload-arch %t/duplicate/offload-arch
! RUN: chmod +x %t/empty/offload-arch %t/whitespace/offload-arch
! RUN: chmod +x %t/fail/offload-arch
! RUN: chmod +x %t/mixed/offload-arch
! RUN: touch %t/input.o
!
! An explicit architecture is forwarded without running the detection tool.
! RUN: %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   --offload-arch=sm_86 -B%t/fail %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ARCH-SM86
!
! An omitted architecture and `native` use offload-arch --only=nvptx.
! RUN: %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   -B%t/single %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ARCH-SM86
! RUN: %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   --offload-arch=native -B%t/single %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ARCH-SM86
!
! Repeated reports of the same architecture are accepted after deduplication.
! RUN: %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   -B%t/duplicate %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ARCH-SM86
!
! Empty output, tool failure, and different detected architectures are errors.
! RUN: not %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   -B%t/empty %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DETECT-ERROR
! RUN: not %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   -B%t/whitespace %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DETECT-ERROR
! RUN: not %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   -B%t/fail %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DETECT-ERROR
! RUN: not %flang -### -c --target=x86_64-unknown-linux-gnu -fopenacc \
! RUN:   -B%t/mixed %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=MULTIPLE
!
! OpenACC currently accepts one explicit NVIDIA architecture only.
! RUN: not %flang -### -c -fopenacc --offload-arch=sm_80,sm_86 \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=MULTIPLE
! RUN: not %flang -### -c -fopenacc \
! RUN:   --offload-arch=sm_86 --offload-arch=sm_86 \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=MULTIPLE
! RUN: not %flang -### -c -fopenacc --offload-arch=gfx90a \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=BAD-GFX
! RUN: not %flang -### -c -fopenacc --offload-arch=sm_999 \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=BAD-UNKNOWN
! RUN: not %flang -### -c -fopenacc --offload-arch= \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=BAD-EMPTY
! RUN: not %flang -### -c -fopenacc --no-offload-arch=sm_86 \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=NO-ARCH
! RUN: not %flang -### -c -fopenacc --cuda-gpu-arch=sm_86 \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=CUDA-ALIAS
! RUN: not %flang -### -c -fopenacc --no-cuda-gpu-arch=sm_86 \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=NO-CUDA-ALIAS
!
! Language-only actions do not require or detect an architecture.
! RUN: %flang -### -fsyntax-only -fopenacc -B%t/fail %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=SYNTAX \
! RUN:       --implicit-check-not=-openacc-target-arch
! RUN: %flang -### -fsyntax-only -fopenacc --offload-arch=native \
! RUN:   -B%t/fail %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=SYNTAX \
! RUN:     --implicit-check-not=-openacc-target-arch
! RUN: %flang -### -fopenacc -B%t/fail %t/input.o
! RUN: %flang_fc1 -fsyntax-only -fopenacc %t/input.f90
! RUN: %flang_fc1 -emit-fir -fopenacc -o /dev/null %t/input.f90
! RUN: %flang_fc1 -emit-hlfir -fopenacc -o /dev/null %t/input.f90
!
! Direct FC1 code generation requires one concrete architecture.
! RUN: %flang_fc1 -emit-llvm -fopenacc -openacc-target-arch=sm_86 \
! RUN:   -mmlir --disable-acc-pipeline -o /dev/null %t/input.f90
! RUN: not %flang_fc1 -emit-llvm -fopenacc -o /dev/null \
! RUN:   %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FC1-MISSING
! RUN: not %flang_fc1 -emit-llvm -fopenacc \
! RUN:   -openacc-target-arch=sm_86 -openacc-target-arch=sm_86 \
! RUN:   -o /dev/null %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FC1-MULTIPLE
! RUN: not %flang_fc1 -emit-llvm -fopenacc \
! RUN:   -openacc-target-arch=gfx90a -o /dev/null %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FC1-BAD
! RUN: not %flang_fc1 -emit-llvm -fopenacc \
! RUN:   -openacc-target-arch=native -o /dev/null %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FC1-NATIVE
! RUN: not %flang_fc1 -emit-llvm -fopenacc \
! RUN:   -openacc-target-arch= -o /dev/null %t/input.f90 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FC1-EMPTY
! RUN: not %flang_fc1 -fsyntax-only -openacc-target-arch=sm_86 \
! RUN:   %t/input.f90 2>&1 | FileCheck %s --check-prefix=FC1-NO-OPENACC
!
! ARCH-SM86: "{{[^"]*}}flang{{[^"]*}}" "-fc1"
! ARCH-SM86-SAME: "-fopenacc"
! ARCH-SM86-SAME: "-openacc-target-arch=sm_86"
! ARCH-SM86-NOT: "{{[^"]*}}flang{{[^"]*}}" "-fc1"
! ARCH-SM86-NOT: warning: argument unused during compilation
! DETECT-ERROR: error: cannot determine OpenACC architecture: {{.*}}; consider passing it via '--offload-arch'
! MULTIPLE: error: only one OpenACC gpu architecture is supported
! BAD-GFX: error: unsupported OpenACC gpu architecture: gfx90a
! BAD-UNKNOWN: error: unsupported OpenACC gpu architecture: sm_999
! BAD-EMPTY: error: unsupported OpenACC gpu architecture:
! NO-ARCH: error: invalid argument '--no-offload-arch=sm_86' not allowed with '-fopenacc'
! CUDA-ALIAS: error: unknown argument: '--cuda-gpu-arch=sm_86'
! NO-CUDA-ALIAS: error: unknown argument: '--no-cuda-gpu-arch=sm_86'
! SYNTAX: "{{[^"]*}}flang{{[^"]*}}" "-fc1"
! SYNTAX-SAME: "-fsyntax-only"
! SYNTAX-SAME: "-fopenacc"
! FC1-MISSING: error: must pass in an explicit OpenACC gpu architecture
! FC1-MULTIPLE: error: only one OpenACC gpu architecture is supported
! FC1-BAD: error: unsupported OpenACC gpu architecture: gfx90a
! FC1-NATIVE: error: unsupported OpenACC gpu architecture: native
! FC1-EMPTY: error: unsupported OpenACC gpu architecture:
! FC1-NO-OPENACC: error: invalid argument '-openacc-target-arch=sm_86' only allowed with '-fopenacc'
!

!--- input.f90
end

!--- single/offload-arch
#!/bin/sh
[ "$1" = "--only=nvptx" ] || exit 1
echo sm_86

!--- duplicate/offload-arch
#!/bin/sh
[ "$1" = "--only=nvptx" ] || exit 1
echo sm_86
echo sm_86

!--- empty/offload-arch
#!/bin/sh
[ "$1" = "--only=nvptx" ] || exit 1
exit 0

!--- whitespace/offload-arch
#!/bin/sh
[ "$1" = "--only=nvptx" ] || exit 1
printf ' \t \n'

!--- fail/offload-arch
#!/bin/sh
exit 1

!--- mixed/offload-arch
#!/bin/sh
[ "$1" = "--only=nvptx" ] || exit 1
echo sm_80
echo sm_86
