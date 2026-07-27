// Test OpenACC device management ops translation: init, shutdown, set, wait
// RUN: fir-opt --fir-to-llvm-ir %s | mlir-translate --mlir-to-llvmir | FileCheck %s

module {
  func.func @test_acc_ops() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    
    // Test acc.init
    // CHECK-DAG: call void @__tgt_acc_init(ptr {{.*}}, i64 0, i64 1, i64 -1)
    acc.init
    
    // CHECK-DAG: call void @__tgt_acc_init(ptr {{.*}}, i64 0, i64 1, i64 0)
    acc.init device_num(%c0_i32 : i32)
    
    // CHECK-DAG: call void @__tgt_acc_init(ptr {{.*}}, i64 0, i64 4, i64 -1)
    acc.init attributes {device_types = [#acc.device_type<nvidia>]}
    
    // CHECK-DAG: call void @__tgt_acc_init(ptr {{.*}}, i64 0, i64 4, i64 0)
    acc.init device_num(%c0_i32 : i32) attributes {device_types = [#acc.device_type<nvidia>]}
    
    // CHECK-DAG: call void @__tgt_acc_init(ptr {{.*}}, i64 0, i64 0, i64 -1)
    acc.init attributes {device_types = [#acc.device_type<star>]}
    
    // CHECK-NOT: acc.init
    // acc.init if(%true) generates conditional branch
    // CHECK: br i1 {{true|%.+}}, label %acc.init.then, label %acc.init.end
    // CHECK: acc.init.then:
    // CHECK: call void @__tgt_acc_init(ptr {{.*}}, i64 0, i64 1, i64 -1)
    // CHECK: br label %acc.init.end
    // CHECK: acc.init.end:
    acc.init if(%true)
    
    // Test acc.shutdown
    // CHECK-DAG: call void @__tgt_acc_shutdown(ptr {{.*}}, i64 0, i64 1, i64 -1)
    acc.shutdown
    
    // CHECK-DAG: call void @__tgt_acc_shutdown(ptr {{.*}}, i64 0, i64 1, i64 0)
    acc.shutdown device_num(%c0_i32 : i32)
    
    // CHECK-DAG: call void @__tgt_acc_shutdown(ptr {{.*}}, i64 0, i64 4, i64 -1)
    acc.shutdown attributes {device_types = [#acc.device_type<nvidia>]}
    
    // CHECK-NOT: acc.shutdown
    // acc.shutdown if(%true) generates conditional branch
    // CHECK: br i1 {{true|%.+}}, label %acc.shutdown.then, label %acc.shutdown.end
    // CHECK: acc.shutdown.then:
    // CHECK: call void @__tgt_acc_shutdown(ptr {{.*}}, i64 0, i64 1, i64 -1)
    // CHECK: br label %acc.shutdown.end
    // CHECK: acc.shutdown.end:
    acc.shutdown if(%true)
    
    // Test acc.set
    // CHECK-DAG: call void @__tgt_acc_set_device_num(ptr {{.*}}, i64 0, i64 1, i64 0)
    acc.set device_num(%c0_i32 : i32)
    
    // CHECK-DAG: call void @__tgt_acc_set_device_type(ptr {{.*}}, i64 0, i64 4)
    acc.set attributes {device_type = #acc.device_type<nvidia>}
    
    // CHECK-DAG: call void @__tgt_acc_set_default_async(ptr {{.*}}, i64 1)
    acc.set default_async(%c1_i32 : i32)
    
    // CHECK-DAG: call void @__tgt_acc_set_device_num(ptr {{.*}}, i64 0, i64 4, i64 0)
    acc.set device_num(%c0_i32 : i32) attributes {device_type = #acc.device_type<nvidia>}
    
    // CHECK-DAG: call void @__tgt_acc_set_default_async(ptr {{.*}}, i64 1)
    // CHECK-DAG: call void @__tgt_acc_set_device_num(ptr {{.*}}, i64 0, i64 1, i64 0)
    acc.set default_async(%c1_i32 : i32) device_num(%c0_i32 : i32)
    
    // CHECK-NOT: acc.set
    // acc.set if(%true) generates conditional branch
    // CHECK: br i1 {{true|%.+}}, label %acc.set.then, label %acc.set.end
    // CHECK: acc.set.then:
    // CHECK: call void @__tgt_acc_set_device_num
    // CHECK: br label %acc.set.end
    // CHECK: acc.set.end:
    acc.set device_num(%c0_i32 : i32) if(%true)
    
    // Test acc.wait
    // CHECK-DAG: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 0, ptr null, i64 -1)
    acc.wait
    
    // CHECK-DAG: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 1, {{.*}}, i64 -1)
    acc.wait(%c1_i32 : i32)
    
    // CHECK-DAG: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 3, {{.*}}, i64 -1)
    acc.wait(%c1_i32, %c1_i32, %c1_i32 : i32, i32, i32)
    
    // CHECK-DAG: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 0, ptr null, i64 -2)
    acc.wait async
    
    // CHECK-DAG: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 1, {{.*}}, i64 -2)
    acc.wait(%c1_i32 : i32) async
    
    // CHECK-DAG: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 1, {{.*}}, i64 1)
    acc.wait(%c1_i32 : i32) async(%c1_i32 : i32)
    
    // CHECK-NOT: acc.wait
    // acc.wait if(%true) generates conditional branch
    // CHECK: br i1 {{true|%.+}}, label %acc.wait.then, label %acc.wait.end
    // CHECK: acc.wait.then:
    // CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 0, i32 -1, i32 0, ptr null, i64 -1)
    // CHECK: br label %acc.wait.end
    // CHECK: acc.wait.end:
    acc.wait if(%true)
    
    // checkICVs
    // CHECK: call void @__tgt_acc_set_device_type(ptr {{.*}}, i64 0, i64 2)
    acc.set attributes {device_type = #acc.device_type<host>}
    return
  }
}
