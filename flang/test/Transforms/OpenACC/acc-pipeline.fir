// RUN: fir-opt %s --fir-acc-pipeline | FileCheck %s --check-prefix=LOWERING
// RUN: fir-opt /dev/null --fir-acc-pipeline --dump-pass-pipeline \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=PIPELINE \
// RUN:   --implicit-check-not=acc-emit-remarks-private \
// RUN:   --implicit-check-not=acc-emit-remarks-loop
// RUN: fir-opt /dev/null \
// RUN:   --fir-acc-pipeline='emit-remarks=true' \
// RUN:   --dump-pass-pipeline -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REMARKS
// RUN: fir-opt /dev/null \
// RUN:   --fir-acc-pipeline='enable-implicit-reduction-copy=false ignore-default-none=true' \
// RUN:   --dump-pass-pipeline -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DATA-OPTIONS
// RUN: not fir-opt /dev/null \
// RUN:   --fir-acc-pipeline='device-type=radeon' \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-TARGET

// PIPELINE: Pass Manager with 26 passes:
// PIPELINE-NEXT: builtin.module(
// PIPELINE-NEXT: acc-initialize-fir-analyses,
// PIPELINE-NEXT: acc-declare-action-conversion,
// PIPELINE-NEXT: acc-implicit-routine{device-type=nvidia},
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: offload-livein-value-canonicalization
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: acc-implicit-declare,
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-if-clause-lowering
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-use-device-canonicalizer
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: acc-initialize-fir-analyses,
// PIPELINE-NEXT: acc-implicit-data{enable-implicit-reduction-copy=true ignore-default-none=false},
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: openacc-legalize-data-values{apply-to-acc-data-construct=true host-to-device=true}
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: offload-target-verifier{device-type=nvidia soft-check=false}
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-legalize-serial
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: fir-acc-recipe-bufferization,
// PIPELINE-NEXT: acc-recipe-materialization,
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-optimize-firstprivate-map
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-loop-tiling{default-tile-size=32}
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: acc-routine-lowering{device-type=nvidia},
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-specialize-for-host{enable-host-fallback=false}
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-specialize-for-device{the-device-types={3,4}}
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: offload-livein-value-canonicalization
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-compute-lowering{device-type=nvidia}
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: func.func(
// PIPELINE-NEXT: acc-bind-routine{device-type=nvidia}
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: gpu.module(
// PIPELINE-NEXT: gpu.func(
// PIPELINE-NEXT: acc-bind-routine{device-type=nvidia}
// PIPELINE-NEXT: )
// PIPELINE-NEXT: ),
// PIPELINE-NEXT: acc-declare-gpu-module-insertion,
// PIPELINE-NEXT: acc-routine-to-gpu-func{device-type=nvidia},
// PIPELINE-NEXT: convert-openacc-to-scf
// PIPELINE-NEXT: )

// REMARKS: offload-target-verifier{device-type=nvidia soft-check=false}
// REMARKS: acc-emit-remarks-private
// REMARKS: acc-legalize-serial
// REMARKS: acc-compute-lowering{device-type=nvidia}
// REMARKS: acc-emit-remarks-loop
// REMARKS: acc-bind-routine{device-type=nvidia}

// DATA-OPTIONS: acc-implicit-data{enable-implicit-reduction-copy=false ignore-default-none=true}

// UNSUPPORTED-TARGET: for the --device-type option
// UNSUPPORTED-TARGET-SAME: radeon

// LOWERING-LABEL: func.func @host_compute
// LOWERING-NOT: acc.serial
// LOWERING: acc.kernel_environment
// LOWERING: acc.compute_region
// LOWERING: func.call @callee
// LOWERING: gpu.module @acc_gpu_module
// LOWERING: gpu.func @callee

func.func @callee() {
  return
}

func.func @host_compute() {
  acc.serial {
    func.call @callee() : () -> ()
    acc.yield
  }
  return
}
