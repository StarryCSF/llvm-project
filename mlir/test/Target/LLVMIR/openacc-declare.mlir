// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Mapping globals are emitted before all function definitions.
// CHECK-DAG: @.offload_maptypes_declare{{(\.[0-9]+)?}} = private unnamed_addr constant [1 x i64] {{(zeroinitializer|\[i64 0\])}}
// CHECK-DAG: @.offload_maptypes_declare{{(\.[0-9]+)?}} = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK-DAG: @.offload_maptypes_declare{{(\.[0-9]+)?}} = private unnamed_addr constant [1 x i64] [i64 1056768]
// CHECK-DAG: @.offload_maptypes_declare{{(\.[0-9]+)?}} = private unnamed_addr constant [1 x i64] [i64 262144]
// CHECK-DAG: @.offload_maptypes_declare_end = private unnamed_addr constant [1 x i64] [i64 8]

//===----------------------------------------------------------------------===//
// declare_enter with single data operand
//===----------------------------------------------------------------------===//

llvm.func @declare_create(%a: !llvm.ptr) {
  %0 = acc.create varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0 : !llvm.ptr)
  // CHECK-LABEL: define void @declare_create
  // CHECK: call void @__tgt_acc_declare
  llvm.return
}
// CHECK-LABEL: define void @declare_copyin
llvm.func @declare_copyin(%a: !llvm.ptr) {
  %0 = acc.copyin varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0 : !llvm.ptr)
  llvm.return
}
// CHECK-LABEL: define void @declare_present
llvm.func @declare_present(%a: !llvm.ptr) {
  %0 = acc.present varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @declare_device_resident
llvm.func @declare_device_resident(%a: !llvm.ptr) {
  %0 = acc.declare_device_resident varPtr(%a : !llvm.ptr)
      varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0 : !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// declare_enter with multiple data operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_multiple_operands
// CHECK: call void @__tgt_acc_declare
llvm.func @declare_multiple_operands(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {
  %0 = acc.create varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  %1 = acc.copyin varPtr(%b : !llvm.ptr) varType(i32) -> !llvm.ptr
  %2 = acc.present varPtr(%c : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0, %1, %2 : !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// declare_enter with token result and declare_exit using token
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_enter_exit_with_token
// CHECK: call void @__tgt_acc_declare
// CHECK: call void @__tgt_acc_data_end
llvm.func @declare_enter_exit_with_token(%a: !llvm.ptr) {
  %0 = acc.create varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  %token = acc.declare_enter dataOperands(%0 : !llvm.ptr)
  acc.declare_exit token(%token) dataOperands(%0 : !llvm.ptr)
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// declare_exit with token only (no dataOperands) - falls back to enter operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_exit_token_only
// CHECK: call void @__tgt_acc_declare
// CHECK: call void @__tgt_acc_data_end
llvm.func @declare_exit_token_only(%a: !llvm.ptr) {
  %0 = acc.create varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  %token = acc.declare_enter dataOperands(%0 : !llvm.ptr)
  acc.declare_exit token(%token)
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// declare_exit without token (direct data operand)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_exit_direct
// CHECK: call void @__tgt_acc_declare
// CHECK: call void @__tgt_acc_data_end
llvm.func @declare_exit_direct(%a: !llvm.ptr) {
  %0 = acc.create varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0 : !llvm.ptr)
  acc.declare_exit dataOperands(%0 : !llvm.ptr)
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// declare_exit with getdeviceptr operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_exit_getdeviceptr
// CHECK: call void @__tgt_acc_declare
// CHECK: call void @__tgt_acc_data_end
llvm.func @declare_exit_getdeviceptr(%a: !llvm.ptr) {
  %0 = acc.create varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0 : !llvm.ptr)
  %1 = acc.getdeviceptr varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr {dataClause = #acc<data_clause acc_create>}
  acc.declare_exit dataOperands(%1 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// deviceptr - no runtime mapping
//===----------------------------------------------------------------------===//

// deviceptr does not create a runtime mapping.
// CHECK-LABEL: define void @declare_deviceptr
// CHECK-NOT: call void @__tgt_acc_declare
llvm.func @declare_deviceptr(%a: !llvm.ptr) {
  %0 = acc.deviceptr varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare_enter dataOperands(%0 : !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// acc.declare implicit region with single operand
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_implicit_region
// CHECK: call void @__tgt_acc_declare
// CHECK: call void @__tgt_acc_data_end
llvm.func @declare_implicit_region(%a: !llvm.ptr) {
  %0 = acc.present varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare dataOperands(%0 : !llvm.ptr) {
    // implicit region - function body would go here
  }
  llvm.return
}

//===----------------------------------------------------------------------===//
// acc.declare implicit region with multiple operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_implicit_region_multiple
// CHECK: call void @__tgt_acc_declare
// CHECK: call void @__tgt_acc_data_end
llvm.func @declare_implicit_region_multiple(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {
  %0 = acc.present varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  %1 = acc.copyin varPtr(%b : !llvm.ptr) varType(i32) -> !llvm.ptr
  %2 = acc.create varPtr(%c : !llvm.ptr) varType(i32) -> !llvm.ptr
  acc.declare dataOperands(%0, %1, %2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    // implicit region
  }
  llvm.return
}

//===----------------------------------------------------------------------===//
// acc.declare with multiple llvm.ptr operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_multiple_ptrs
// CHECK: call void @__tgt_acc_declare
llvm.func @declare_multiple_ptrs(%a: !llvm.ptr, %b: !llvm.ptr) {
  %0 = acc.present varPtr(%a : !llvm.ptr) varType(i32) -> !llvm.ptr
  %1 = acc.copyin varPtr(%b : !llvm.ptr) varType(!llvm.array<10 x f32>) -> !llvm.ptr
  acc.declare_enter dataOperands(%0, %1 : !llvm.ptr, !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// declare with array bounds
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_with_bounds
// CHECK: call void @__tgt_acc_declare
llvm.func @declare_with_bounds(%a: !llvm.ptr, %lb: i64, %extent: i64, %stride: i64) {
  %0 = acc.bounds lowerbound(%lb : i64) extent(%extent : i64) stride(%stride : i64) startIdx(%lb : i64)
  %1 = acc.create varPtr(%a : !llvm.ptr) varType(!llvm.array<100 x i32>) bounds(%0) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  llvm.return
}
//===----------------------------------------------------------------------===//
// declare_device_resident with bounds
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define void @declare_device_resident_with_bounds
// CHECK: call void @__tgt_acc_declare
llvm.func @declare_device_resident_with_bounds(%a: !llvm.ptr, %lb: i64, %extent: i64) {
  %0 = acc.bounds lowerbound(%lb : i64) extent(%extent : i64) startIdx(%lb : i64)
  %1 = acc.declare_device_resident varPtr(%a : !llvm.ptr) varType(!llvm.array<50 x f32>) bounds(%0) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  llvm.return
}

//===----------------------------------------------------------------------===//
// Global variable declare - acc.global_ctor/dtor translation
//===----------------------------------------------------------------------===//

// Test global variable with acc.declare attribute and ctor/dtor functions.
// This models the OpenACC declare directive on module-level variables.
// Note: acc.global_ctor/dtor are converted to llvm.func by ACCDeclareCtorDtorConversion
// pass before this translation stage.
llvm.mlir.global external @global_arr() {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.array<10 x i32> {
  %0 = llvm.mlir.zero : !llvm.array<10 x i32>
  llvm.return %0 : !llvm.array<10 x i32>
}

// acc.global_ctor is converted to llvm.func by ACCDeclareCtorDtorConversion pass.
// Here we test the translation of the converted ctor function.
// CHECK-LABEL: define internal void @global_arr_acc_ctor
// CHECK: call void @__tgt_acc_declare
// CHECK: ret void
llvm.func internal @global_arr_acc_ctor() {
  %0 = llvm.mlir.addressof @global_arr {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.create varPtr(%0 : !llvm.ptr) varType(!llvm.array<10 x i32>) -> !llvm.ptr
  acc.declare_enter dataOperands(%1 : !llvm.ptr)
  llvm.return
}

// acc.global_dtor is converted to llvm.func by ACCDeclareCtorDtorConversion pass.
// Here we test the translation of the converted dtor function.
// CHECK-LABEL: define internal void @global_arr_acc_dtor
// CHECK: call void @__tgt_acc_data_end
// CHECK: ret void
llvm.func internal @global_arr_acc_dtor() {
  %0 = llvm.mlir.addressof @global_arr {acc.declare = #acc.declare<dataClause = acc_create>} : !llvm.ptr
  %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr) varType(!llvm.array<10 x i32>) -> !llvm.ptr {dataClause = #acc<data_clause acc_create>}
  acc.declare_exit dataOperands(%1 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr)
  llvm.return
}
