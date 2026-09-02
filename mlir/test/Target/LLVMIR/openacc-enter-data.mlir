// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// enter_data if clause.
llvm.func @testenterdata_if(%arg0: !llvm.ptr, %cond: i1) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data if(%cond) dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testenterdata_if
// CHECK: br i1 %[[COND:.*]], label %acc.standalone.then, label %acc.standalone.end
// CHECK: call void @__tgt_acc_data_enter(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// enter_data async clause.
llvm.func @testenterdata_async(%arg0: !llvm.ptr, %async: i64) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data async(%async : i64) dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testenterdata_async
// CHECK: call void @__tgt_acc_data_enter(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 %[[ASYNC:.*]])

// -----

// enter_data wait clause.
llvm.func @testenterdata_wait(%arg0: !llvm.ptr, %wait: i64) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data wait(%wait : i64) dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testenterdata_wait
// CHECK: store i64 %[[WAIT:.*]], ptr %{{.*}}
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 1, ptr {{.*}}, i64 -1)
// CHECK: call void @__tgt_acc_data_enter(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// enter_data copyin clause.
llvm.func @testenterdata_copyin(%arg0: !llvm.ptr) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 17]
// CHECK-LABEL: define void @testenterdata_copyin
// CHECK: call void @__tgt_acc_data_enter(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// enter_data create clause.
llvm.func @testenterdata_create(%arg0: !llvm.ptr) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testenterdata_create
// CHECK: call void @__tgt_acc_data_enter(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// enter_data create(zero:) clause.
llvm.func @testenterdata_create_zero(%arg0: !llvm.ptr) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
      {dataClause = #acc<data_clause acc_create_zero>}
  acc.enter_data dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 131088]
// CHECK-LABEL: define void @testenterdata_create_zero
// CHECK: call void @__tgt_acc_data_enter(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// enter_data attach clause.
llvm.func @testenterdata_attach(%arg0: !llvm.ptr) {
  %0 = acc.attach varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data dataOperands(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testenterdata_attach
// CHECK: call void @__tgt_acc_data_enter(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// exit_data if clause.
llvm.func @testexitdata_if(%arg0: !llvm.ptr, %cond: i1) {
  %0 = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.exit_data if(%cond) dataOperands(%0 : !llvm.ptr)
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testexitdata_if
// CHECK: br i1 %[[COND:.*]], label %acc.standalone.then, label %acc.standalone.end
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// exit_data async clause.
llvm.func @testexitdata_async(%arg0: !llvm.ptr, %async: i64) {
  %0 = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.exit_data async(%async : i64) dataOperands(%0 : !llvm.ptr)
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testexitdata_async
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 %[[ASYNC:.*]])

// -----

// exit_data wait clause.
llvm.func @testexitdata_wait(%arg0: !llvm.ptr, %wait: i64) {
  %0 = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.exit_data wait(%wait : i64) dataOperands(%0 : !llvm.ptr)
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testexitdata_wait
// CHECK: store i64 %[[WAIT:.*]], ptr %{{.*}}
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 1, ptr {{.*}}, i64 -1)
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// exit_data copyout clause.
llvm.func @testexitdata_copyout(%arg0: !llvm.ptr) {
  %0 = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.exit_data dataOperands(%0 : !llvm.ptr)
  acc.copyout accPtr(%0 : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr)
      varType(f32)
  llvm.return
}

// FROM|PTR_AND_OBJ = 18
// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 18]
// CHECK-LABEL: define void @testexitdata_copyout
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// exit_data delete clause.
llvm.func @testexitdata_delete(%arg0: !llvm.ptr) {
  %0 = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32)
      -> !llvm.ptr {dataClause = #acc<data_clause acc_delete>}
  acc.exit_data dataOperands(%0 : !llvm.ptr)
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// DELETE|PTR_AND_OBJ = 16
// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testexitdata_delete
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// exit_data detach clause.
llvm.func @testexitdata_detach(%arg0: !llvm.ptr) {
  %0 = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32)
      -> !llvm.ptr {dataClause = #acc<data_clause acc_detach>}
  acc.exit_data dataOperands(%0 : !llvm.ptr)
  acc.detach accPtr(%0 : !llvm.ptr)
  llvm.return
}

// DETACH|PTR_AND_OBJ = 16
// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 16]
// CHECK-LABEL: define void @testexitdata_detach
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// exit_data finalize clause.
llvm.func @testexitdata_finalize(%arg0: !llvm.ptr) {
  %0 = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32)
      -> !llvm.ptr {dataClause = #acc<data_clause acc_delete>}
  acc.exit_data dataOperands(%0 : !llvm.ptr) attributes {finalize}
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// DELETE|PTR_AND_OBJ|FINALIZE = 24
// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 24]
// CHECK-LABEL: define void @testexitdata_finalize
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)
