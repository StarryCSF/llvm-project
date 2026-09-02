// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-DAG: @[[MAPTYPES:.*]] = private unnamed_addr constant [2 x i64] [i64 18, i64 17]
// CHECK-DAG: @[[MAPTYPES_IF_PRESENT:.*]] = private unnamed_addr constant [1 x i64] [i64 524306]
// CHECK-DAG: @[[MAPTYPES_HOST_SELF:.*]] = private unnamed_addr constant [2 x i64] [i64 18, i64 18]
// CHECK-DAG: @[[MAPTYPES_ASYNC_BARE:.*]] = private unnamed_addr constant [1 x i64] [i64 18]
// CHECK-DAG: @[[MAPTYPES_WAIT_BARE:.*]] = private unnamed_addr constant [1 x i64] [i64 17]
// CHECK-DAG: @[[MAPTYPES_WAIT_DEVNUM:.*]] = private unnamed_addr constant [2 x i64] [i64 17, i64 17]
// CHECK: define void @test_update(ptr %[[HOST:.*]], ptr %[[DEVICE:.*]], i1 %[[COND:.*]], i64 %[[ASYNC:.*]], i64 %[[WAIT:.*]])
// CHECK: br i1 %[[COND]], label %acc.update.then, label %acc.update.end
// CHECK: %[[WAITLIST:.*]] = alloca [1 x i64]
// CHECK: store i64 %[[WAIT]], ptr {{.*}}
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 1, i32 -1, i32 1, ptr %[[WAITLIST]], i64 %[[ASYNC]])
// CHECK: call void @__tgt_acc_data_update(ptr {{.*}}, i64 0, i64 1, i32 2, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[MAPTYPES]], ptr {{.*}}, ptr null, ptr null, i64 %[[ASYNC]])

llvm.func @test_update(%host: !llvm.ptr, %device: !llvm.ptr, %cond: i1,
                       %async: i64, %wait: i64) {
  %host_dev = acc.getdeviceptr varPtr(%host : !llvm.ptr) varType(f32)
      -> !llvm.ptr
  %device_dev = acc.update_device varPtr(%device : !llvm.ptr) varType(f32)
      -> !llvm.ptr
  acc.update if(%cond) async(%async : i64) wait({%wait : i64})
      dataOperands(%device_dev, %host_dev : !llvm.ptr, !llvm.ptr)
  llvm.return
}

// -----

// if_present sets the TGT_ACC_MAPTYPE_IF_PRESENT (0x80000) bit on the
// maptypes so that the runtime skips variables that are not present on the
// device instead of terminating.
// NOTE: the UnitAttr's textual name is camelCase (as declared in ODS).
// CHECK: define void @test_update_if_present(ptr %[[HOST_IF:.*]])
// CHECK: call void @__tgt_acc_data_update(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[MAPTYPES_IF_PRESENT]], ptr {{.*}}, ptr null, ptr null, i64 -1)
llvm.func @test_update_if_present(%host: !llvm.ptr) {
  %host_dev = acc.getdeviceptr varPtr(%host : !llvm.ptr) varType(f32)
      -> !llvm.ptr
  acc.update dataOperands(%host_dev : !llvm.ptr) attributes {ifPresent}
  llvm.return
}

// -----

// OpenACC 3.3 2.14.4: host is a synonym for self; both copy from device to
// local memory, so both are passed to the runtime with the FROM maptype.
llvm.func @test_update_host_self(%h: !llvm.ptr, %s: !llvm.ptr) {
  %h_dev = acc.getdeviceptr varPtr(%h : !llvm.ptr) varType(f32) -> !llvm.ptr
      {dataClause = #acc<data_clause acc_update_host>}
  %s_dev = acc.getdeviceptr varPtr(%s : !llvm.ptr) varType(f32) -> !llvm.ptr
      {dataClause = #acc<data_clause acc_update_self>}
  acc.update dataOperands(%h_dev, %s_dev : !llvm.ptr, !llvm.ptr)
  acc.update_host accPtr(%s_dev : !llvm.ptr) to varPtr(%s : !llvm.ptr)
      varType(f32) {dataClause = #acc<data_clause acc_update_self>}
  llvm.return
}
// CHECK: define void @test_update_host_self(ptr %{{.*}}, ptr %{{.*}})
// CHECK: call void @__tgt_acc_data_update(ptr {{.*}}, i64 0, i64 1, i32 2, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[MAPTYPES_HOST_SELF]], ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// async without argument is currently translated to -2.
llvm.func @test_update_async_bare(%a: !llvm.ptr) {
  %a_dev = acc.getdeviceptr varPtr(%a : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.update async dataOperands(%a_dev : !llvm.ptr)
  llvm.return
}
// CHECK: define void @test_update_async_bare(ptr %{{.*}})
// CHECK: call void @__tgt_acc_data_update(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[MAPTYPES_ASYNC_BARE]], ptr {{.*}}, ptr null, ptr null, i64 -2)

// -----

// wait without argument waits for all queues; no wait list is passed.
llvm.func @test_update_wait_bare(%a: !llvm.ptr) {
  %a_dev = acc.update_device varPtr(%a : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.update wait dataOperands(%a_dev : !llvm.ptr)
  llvm.return
}
// CHECK: define void @test_update_wait_bare(ptr %{{.*}})
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 1, i32 -1, i32 0, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_update(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[MAPTYPES_WAIT_BARE]], ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// wait with devnum and multiple queues, combined with async with an int32
// argument (cast to int64).
llvm.func @test_update_wait_devnum(%a: !llvm.ptr, %b: !llvm.ptr,
                                   %devnum: i32, %a32: i32, %w1: i64,
                                   %w2: i64) {
  %a_dev = acc.update_device varPtr(%a : !llvm.ptr) varType(f32) -> !llvm.ptr
  %b_dev = acc.update_device varPtr(%b : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.update async(%a32 : i32)
      wait({devnum: %devnum : i32, %w1 : i64, %w2 : i64})
      dataOperands(%a_dev, %b_dev : !llvm.ptr, !llvm.ptr)
  llvm.return
}
// CHECK: define void @test_update_wait_devnum(ptr %{{.*}}, ptr %{{.*}}, i32 %[[DEVNUM:.*]], i32 %[[A32:.*]], i64 %[[W1:.*]], i64 %[[W2:.*]])
// CHECK: %[[A64:.*]] = sext i32 %[[A32]] to i64
// CHECK: %[[WAITLIST2:.*]] = alloca [2 x i64]
// CHECK: store i64 %[[W1]], ptr {{.*}}
// CHECK: store i64 %[[W2]], ptr {{.*}}
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 1, i32 %[[DEVNUM]], i32 2, ptr %[[WAITLIST2]], i64 %[[A64]])
// CHECK: call void @__tgt_acc_data_update(ptr {{.*}}, i64 0, i64 1, i32 2, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[MAPTYPES_WAIT_DEVNUM]], ptr {{.*}}, ptr null, ptr null, i64 %[[A64]])

// CHECK: declare void @__tgt_acc_data_update(ptr, i64, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)
