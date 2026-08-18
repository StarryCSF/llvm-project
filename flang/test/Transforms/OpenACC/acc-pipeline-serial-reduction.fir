// RUN: fir-opt %s --fir-acc-pipeline | FileCheck %s \
// RUN:   --implicit-check-not=acc.compute_region \
// RUN:   --implicit-check-not=acc.offload_id

// Verify that serial legalization runs before recipe materialization. The
// reduction recipe must be consumed successfully before GPU lowering.
// CHECK-NOT: acc.reduction.recipe
// CHECK-LABEL: func.func @serial_reduction
// CHECK-NOT: acc.serial
// CHECK: acc.kernel_environment
// CHECK: gpu.launch_func @serial_reduction_kernel::@serial_reduction_kernel
// CHECK: gpu.module @serial_reduction_kernel
// CHECK: gpu.func @serial_reduction_kernel
// CHECK: fir.load
// CHECK: arith.addf
// CHECK: fir.store

acc.reduction.recipe @reduction_add_ref_f64 : !fir.ref<f64> reduction_operator <add> init {
^bb0(%arg0: !fir.ref<f64>):
  %cst = arith.constant 0.000000e+00 : f64
  %0 = fir.alloca f64
  %1 = fir.declare %0 {uniq_name = "acc.reduction.init"} : (!fir.ref<f64>) -> !fir.ref<f64>
  fir.store %cst to %1 : !fir.ref<f64>
  acc.yield %1 : !fir.ref<f64>
} combiner {
^bb0(%arg0: !fir.ref<f64>, %arg1: !fir.ref<f64>):
  %0 = fir.load %arg0 : !fir.ref<f64>
  %1 = fir.load %arg1 : !fir.ref<f64>
  %2 = arith.addf %0, %1 fastmath<contract> : f64
  fir.store %2 to %arg0 : !fir.ref<f64>
  acc.yield %arg0 : !fir.ref<f64>
}

func.func @serial_reduction(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) {
  %cst = arith.constant 1.000000e+00 : f64
  %scope = fir.dummy_scope : !fir.dscope
  %decl = fir.declare %arg0 dummy_scope %scope {uniq_name = "_QFserial_reductionEtmp"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
  %red = acc.reduction varPtr(%decl : !fir.ref<f64>) recipe(@reduction_add_ref_f64) -> !fir.ref<f64> {name = "tmp"}
  acc.serial reduction(%red : !fir.ref<f64>) {
    %val = fir.load %red : !fir.ref<f64>
    %sum = arith.addf %val, %cst fastmath<contract> : f64
    fir.store %sum to %red : !fir.ref<f64>
    acc.yield
  }
  return
}
