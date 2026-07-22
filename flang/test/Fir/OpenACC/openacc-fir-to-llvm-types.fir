// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --disable-fir-type --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=DISABLED

func.func @enter_data_ref(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %copyin = acc.copyin varPtr(%arg0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>>
  acc.enter_data dataOperands(%copyin : !fir.ref<!fir.array<10xf32>>)
  return
}

// CHECK-LABEL: llvm.func @enter_data_ref
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARG0]] : !llvm.ptr) varType(!llvm.array<10 x f32>) -> !llvm.ptr
// CHECK: acc.enter_data dataOperands(%[[COPYIN]] : !llvm.ptr)
// CHECK-NOT: !fir.ref
// CHECK-NOT: unrealized_conversion_cast

// DISABLED-LABEL: llvm.func @enter_data_ref
// DISABLED: builtin.unrealized_conversion_cast
// DISABLED: acc.copyin
// DISABLED-SAME: !fir.ref<!fir.array<10xf32>>

// -----

func.func @data_region_ref(%arg0: !fir.ref<f32>) {
  %copyin = acc.copyin varPtr(%arg0 : !fir.ref<f32>) -> !fir.ref<f32>
  acc.data dataOperands(%copyin : !fir.ref<f32>) {
    %0 = fir.load %arg0 : !fir.ref<f32>
    acc.terminator
  }
  return
}

// CHECK-LABEL: llvm.func @data_region_ref
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr)
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[ARG0]] : !llvm.ptr) varType(f32) -> !llvm.ptr
// CHECK: acc.data dataOperands(%[[COPYIN]] : !llvm.ptr)
// CHECK: llvm.load %[[ARG0]] : !llvm.ptr -> f32
// CHECK-NOT: !fir.ref
// CHECK-NOT: unrealized_conversion_cast

// -----

func.func @copyin_box(%arg0: !fir.box<!fir.array<?xf32>>) {
  %copyin = acc.copyin var(%arg0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  acc.enter_data dataOperands(%copyin : !fir.box<!fir.array<?xf32>>)
  return
}

// CHECK-LABEL: llvm.func @copyin_box
// CHECK: acc.copyin varPtr(%{{.*}} : !llvm.ptr) varType(!llvm.struct<{{.*}}>) -> !llvm.ptr
// CHECK: acc.enter_data dataOperands(%{{.*}} : !llvm.ptr)
// CHECK-NOT: !fir.box
// CHECK-NOT: !fir.ref
// CHECK-NOT: unrealized_conversion_cast

// -----

func.func @exit_data_copyout_delete(%arg0: !fir.ref<f32>) {
  %devptr = acc.getdeviceptr varPtr(%arg0 : !fir.ref<f32>) -> !fir.ref<f32>
  acc.exit_data dataOperands(%devptr : !fir.ref<f32>)
  acc.delete accPtr(%devptr : !fir.ref<f32>)
  acc.copyout accPtr(%devptr : !fir.ref<f32>) to varPtr(%arg0 : !fir.ref<f32>) varType(f32)
  return
}

// CHECK-LABEL: llvm.func @exit_data_copyout_delete
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr)
// CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[ARG0]] : !llvm.ptr) varType(f32) -> !llvm.ptr
// CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.delete accPtr(%[[DEVPTR]] : !llvm.ptr)
// CHECK: acc.copyout accPtr(%[[DEVPTR]] : !llvm.ptr) to varPtr(%[[ARG0]] : !llvm.ptr) varType(f32)
// CHECK-NOT: !fir.ref
// CHECK-NOT: unrealized_conversion_cast

// -----

func.func @update_device_and_update(%arg0: !fir.ref<f32>) {
  %devptr = acc.update_device varPtr(%arg0 : !fir.ref<f32>) -> !fir.ref<f32>
  acc.update dataOperands(%devptr : !fir.ref<f32>)
  return
}

// CHECK-LABEL: llvm.func @update_device_and_update
// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr)
// CHECK: %[[DEVPTR:.*]] = acc.update_device varPtr(%[[ARG0]] : !llvm.ptr) varType(f32) -> !llvm.ptr
// CHECK: acc.update dataOperands(%[[DEVPTR]] : !llvm.ptr)
// CHECK-NOT: !fir.ref
// CHECK-NOT: unrealized_conversion_cast
