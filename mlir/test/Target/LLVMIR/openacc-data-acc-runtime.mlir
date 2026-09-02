// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Cover all structured data-entry classifications in one data construct.
llvm.func @testdataop_all_clauses(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %copyin = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %copyout = acc.create varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr {dataClause = #acc<data_clause acc_copyout>}
  %create = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %present = acc.present varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %nocreate = acc.nocreate varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %deviceptr = acc.deviceptr varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %attach = acc.attach varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data dataOperands(%copyin, %copyout, %create, %present, %nocreate,
                        %deviceptr, %attach : !llvm.ptr, !llvm.ptr, !llvm.ptr,
                        !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%copyin : !llvm.ptr)
  acc.copyout accPtr(%copyout : !llvm.ptr) to varPtr(%arg1 : !llvm.ptr) varType(f32)
  acc.delete accPtr(%create : !llvm.ptr)
  acc.delete accPtr(%present : !llvm.ptr)
  acc.delete accPtr(%nocreate : !llvm.ptr)
  acc.detach accPtr(%attach : !llvm.ptr)
  llvm.return
}

// Entry flags strip PTR_AND_OBJ for scalar f32: copyin TO, copyout/create none,
// present PRESENT|NO_CREATE, nocreate NO_CREATE, deviceptr DEVPTR, attach none.
// Exit: delete, copyout FROM|PTR_AND_OBJ, delete, present, delete, deviceptr, detach.
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [7 x i64] [i64 1, i64 0, i64 0, i64 1056768, i64 8192, i64 1024, i64 0]
// CHECK: @[[END_MAPTYPES:.*]] = private unnamed_addr constant [7 x i64] [i64 8, i64 2, i64 8, i64 1056768, i64 8, i64 1040, i64 24]
// CHECK-LABEL: define void @testdataop_all_clauses
// CHECK: alloca [7 x ptr]
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 7, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 7, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes_end, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Check async propagation and conditional data-region control flow.
llvm.func @testdataop_async_if(%arg0: !llvm.ptr, %async: i32, %cond: i1) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data if(%cond) dataOperands(%0 : !llvm.ptr) async(%async : i32) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @testdataop_async_if
// CHECK: br i1 %{{.*}}, label %acc.data, label %acc.data.skip
// CHECK: acc.data:
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 %{{.*}})
// CHECK: br label %acc.end_data
// CHECK: acc.data.skip:
// CHECK: br label %acc.data.continue
// CHECK: acc.end_data:
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 %{{.*}})
// CHECK: br label %acc.data.continue
// CHECK: acc.data.continue:
// CHECK: ret void

// -----

// Check wait emission and the device_type association on async and wait.
llvm.func @testdataop_wait_device_type(%arg0: !llvm.ptr, %async: i32,
                                        %wait: i32, %devnum: i32) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data async(%async : i32 [#acc.device_type<none>])
      wait({devnum: %devnum : i32, %wait : i32} [#acc.device_type<none>])
      dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @testdataop_wait_device_type
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 1, i32 %{{.*}}, i32 1, ptr %{{.*}}, i64 %{{.*}})
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 %{{.*}})
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 %{{.*}})

// -----

// default is parsed and retained by acc.data. Runtime flag lowering is not
// implemented yet, so the current ABI flags argument remains zero.
llvm.func @testdataop_default(%arg0: !llvm.ptr) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue none>}
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @testdataop_default
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Check upperbound-derived extent and ordinary bounded-range mapping.
llvm.func @testdata_bounds_upper(%arg0: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %bounds = acc.bounds lowerbound(%c1 : i64) upperbound(%c4 : i64) stride(%c1 : i64)
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) bounds(%bounds) -> !llvm.ptr
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// Upperbound wins over a possible extent operand; 1..4 selects 4 f32 values.
// CHECK: @[[UPPER_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @[[UPPER_END_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 8]
// CHECK-LABEL: define void @testdata_bounds_upper
// CHECK: %[[UPPER_PTR:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store ptr null, ptr %{{.*}}
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes_end, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Check the explicit-extent fallback, byte strides, and multidimensional
// offset/span accumulation.
llvm.func @testdata_bounds_extent_and_stride(%arg0: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %c8 = llvm.mlir.constant(8 : i64) : i64
  %extent_bounds = acc.bounds lowerbound(%c2 : i64) extent(%c4 : i64)
  %stride_bounds = acc.bounds lowerbound(%c1 : i64) extent(%c3 : i64)
      stride(%c8 : i64) {strideInBytes = true}
  %dim0 = acc.bounds lowerbound(%c1 : i64) upperbound(%c4 : i64) stride(%c1 : i64)
  %dim1 = acc.bounds lowerbound(%c2 : i64) extent(%c3 : i64) stride(%c8 : i64)
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32)
      bounds(%extent_bounds) -> !llvm.ptr
  %1 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32)
      bounds(%stride_bounds) -> !llvm.ptr
  %2 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32)
      bounds(%dim0, %dim1) -> !llvm.ptr
  acc.data dataOperands(%0, %1, %2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  acc.delete accPtr(%1 : !llvm.ptr)
  acc.delete accPtr(%2 : !llvm.ptr)
  llvm.return
}

// Explicit extent: offset 8, span 16. Byte stride: offset 8, span 20.
// Multidimensional: offset 68, span 80.
// CHECK-LABEL: define void @testdata_bounds_extent_and_stride
// CHECK: %[[EXTENT_PTR:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: %[[BYTE_STRIDE_PTR:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK: store i64 20, ptr %{{.*}}
// CHECK: %[[TWO_D_PTR:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 68
// CHECK: store i64 80, ptr %{{.*}}

// -----

// Check a bounded copy entry and its bounded copyout exit flag.
llvm.func @testdata_bounds_copy(%arg0: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %bounds = acc.bounds lowerbound(%c1 : i64) upperbound(%c4 : i64) stride(%c1 : i64)
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) bounds(%bounds) -> !llvm.ptr
      {dataClause = #acc<data_clause acc_copy>}
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.copyout accPtr(%0 : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr) varType(f32)
  llvm.return
}

// Entry TO and bounded exit FROM both omit PTR_AND_OBJ.
// CHECK: @[[COPY_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @[[COPY_END_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 2]
// CHECK-LABEL: define void @testdata_bounds_copy
// CHECK: %[[COPY_PTR:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store ptr null, ptr %{{.*}}
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes_end, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Unbounded copy: entry copyin + exit copyout, distinct from copyin's delete.
llvm.func @testdataop_copy(%arg0: !llvm.ptr) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
      {dataClause = #acc<data_clause acc_copy>}
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.copyout accPtr(%0 : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr) varType(f32)
  llvm.return
}

// CHECK: @[[UB_COPY_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @[[UB_COPY_END_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 18]
// CHECK-LABEL: define void @testdataop_copy
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes_end, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// async without a queue id lowers to the async-only sentinel.
llvm.func @testdataop_async_only(%arg0: !llvm.ptr) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data async dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @testdataop_async_only
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -2)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -2)

// -----

// wait without values still emits a wait call before data_begin.
llvm.func @testdataop_wait_only(%arg0: !llvm.ptr) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data wait dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @testdataop_wait_only
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 1, i32 -1, i32 0, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end

// -----

// wait with multiple queues and no device_type association.
llvm.func @testdataop_wait_values(%arg0: !llvm.ptr, %w1: i32, %w2: i32) {
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data wait({%w1 : i32, %w2 : i32}) dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @testdataop_wait_values
// CHECK: store i64 %{{.*}}, ptr %{{.*}}
// CHECK: store i64 %{{.*}}, ptr %{{.*}}
// CHECK: call i32 @__tgt_acc_wait(ptr {{.*}}, i64 0, i64 1, i32 -1, i32 2, ptr %{{.*}}, i64 -1)
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// default(present) is retained; runtime flags are still zero.
llvm.func @testdataop_default_present(%arg0: !llvm.ptr) {
  %0 = acc.present varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue present>}
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @testdataop_default_present
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Omitted lowerbound is treated as 0, so upperbound 3 selects 4 f32 values
// at offset 0.
llvm.func @testdata_bounds_default_lb(%arg0: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %bounds = acc.bounds upperbound(%c3 : i64) stride(%c1 : i64)
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) bounds(%bounds) -> !llvm.ptr
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.delete accPtr(%0 : !llvm.ptr)
  llvm.return
}

// CHECK: @[[LB0_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 1]
// CHECK: @[[LB0_END_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 8]
// CHECK-LABEL: define void @testdata_bounds_default_lb
// CHECK: store ptr null, ptr %{{.*}}
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes_end, ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Bounds apply to every structured data-entry kind, not only copy/copyin.
llvm.func @testdata_bounds_clauses(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %bounds = acc.bounds lowerbound(%c1 : i64) upperbound(%c4 : i64) stride(%c1 : i64)
  %copyout = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) bounds(%bounds)
      -> !llvm.ptr {dataClause = #acc<data_clause acc_copyout>}
  %create = acc.create varPtr(%arg1 : !llvm.ptr) varType(f32) bounds(%bounds)
      -> !llvm.ptr
  %present = acc.present varPtr(%arg0 : !llvm.ptr) varType(f32) bounds(%bounds)
      -> !llvm.ptr
  %nocreate = acc.nocreate varPtr(%arg1 : !llvm.ptr) varType(f32) bounds(%bounds)
      -> !llvm.ptr
  %deviceptr = acc.deviceptr varPtr(%arg0 : !llvm.ptr) varType(f32)
      bounds(%bounds) -> !llvm.ptr
  %attach = acc.attach varPtr(%arg1 : !llvm.ptr) varType(f32) bounds(%bounds)
      -> !llvm.ptr
  acc.data dataOperands(%copyout, %create, %present, %nocreate, %deviceptr,
                        %attach : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
                        !llvm.ptr, !llvm.ptr) {
    acc.terminator
  }
  acc.copyout accPtr(%copyout : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr) varType(f32)
  acc.delete accPtr(%create : !llvm.ptr)
  acc.delete accPtr(%present : !llvm.ptr)
  acc.delete accPtr(%nocreate : !llvm.ptr)
  acc.detach accPtr(%attach : !llvm.ptr)
  llvm.return
}

// Bounded 1..4 f32: offset 4, span 16. PTR_AND_OBJ stripped on bounded entries.
// Exit: copyout FROM, create/present/nocreate delete, deviceptr, attach detach.
// CHECK: @[[CLAUSE_MAPTYPES:.*]] = private unnamed_addr constant [6 x i64] [i64 0, i64 0, i64 1056768, i64 8192, i64 1024, i64 0]
// CHECK: @[[CLAUSE_END_MAPTYPES:.*]] = private unnamed_addr constant [6 x i64] [i64 2, i64 8, i64 1056768, i64 8, i64 1040, i64 8]
// CHECK-LABEL: define void @testdata_bounds_clauses
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK: store i64 16, ptr %{{.*}}
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 6, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes, ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 6, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @.offload_maptypes_end, ptr {{.*}}, ptr null, ptr null, i64 -1)
