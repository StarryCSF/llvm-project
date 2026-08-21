// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @consume(!llvm.ptr)

// Check host_data use_device lowers to a device-pointer lookup and that the
// host_data body is emitted inline on the host.
llvm.func @test_host_data_use_device(%arg0: !llvm.ptr) {
  %dev = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.host_data dataOperands(%dev : !llvm.ptr) {
    llvm.call @consume(%dev) : (!llvm.ptr) -> ()
    acc.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @test_host_data_use_device
// CHECK: call ptr @__tgt_acc_get_deviceptr(ptr {{.*}}, ptr null, i64 0, ptr %{{.*}})
// CHECK: call void @consume(ptr {{.*}})
// CHECK: br label %acc.end_host_data

// -----

// Check the host_data if clause.
llvm.func @test_host_data_if(%arg0: !llvm.ptr, %cond: i1) {
  %dev = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.host_data if(%cond) dataOperands(%dev : !llvm.ptr) {
    llvm.call @consume(%dev) : (!llvm.ptr) -> ()
    acc.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @test_host_data_if
// CHECK: %[[SELECT:.*]] = select i1 %{{.*}}, ptr {{.*}}, ptr %{{.*}}
// CHECK: call void @consume(ptr %[[SELECT]])

// -----

// Check host_data if_present and its conjunction with if.
llvm.func @test_host_data_if_present(%arg0: !llvm.ptr, %cond: i1) {
  %dev = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.host_data if(%cond) dataOperands(%dev : !llvm.ptr) {
    llvm.call @consume(%dev) : (!llvm.ptr) -> ()
    acc.terminator
  } attributes {ifPresent}
  llvm.return
}

// CHECK-LABEL: define void @test_host_data_if_present
// CHECK: call i32 @acc_is_present(ptr %{{.*}})
// CHECK: %{{.*}} = icmp ne i32 %{{.*}}, 0
// CHECK: %{{.*}} = and i1 %{{.*}}, %{{.*}}
// CHECK: %{{.*}} = select i1 %{{.*}}, ptr {{.*}}, ptr %{{.*}}
