// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @testenterdataop(%arg0: !llvm.ptr, %arg1 : !llvm.ptr) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %1 = acc.copyin varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data dataOperands(%0, %1 : !llvm.ptr, !llvm.ptr)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }

// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testenterdataop;{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }, align 8
// CHECK: @[[MAPNAME1:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";unknown;{{.*}};{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: @[[MAPNAME2:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";unknown;{{.*}};{{[0-9]*}};{{[0-9]*}};;\00", align 1
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 16, i64 17]
// CHECK: @[[MAPNAMES:.*]] = private constant [{{[0-9]*}} x ptr] [ptr @[[MAPNAME1]], ptr @[[MAPNAME2]]]

// CHECK: define void @testenterdataop(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]])
// CHECK: %[[OFFLOAD_BASEPTR:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAD_PTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAD_SIZES:.*]] = alloca [{{[0-9]*}} x i64]

// CHECK: %[[OFFLOAD_BASEPTR_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTR]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_BASEPTR_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAD_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAD_SIZES]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAD_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTR_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTR]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_BASEPTR_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAD_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAD_SIZES]], i32 0, i32 1
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAD_SIZES_GEP]]



// CHECK: call void @__tgt_acc_data_enter(ptr @[[LOCGLOBAL]], i64 0, i64 1, i32 2, ptr %[[OFFLOAD_BASEPTR]], ptr %[[OFFLOAD_PTRS]], ptr %[[OFFLOAD_SIZES]], ptr @[[MAPTYPES]], ptr @[[MAPNAMES]], ptr null, ptr null, i64 -1)

// CHECK: declare void @__tgt_acc_data_enter(ptr, i64, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)

// -----


llvm.func @testexitdataop(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %arg0_devptr = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %1 = acc.getdeviceptr varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.exit_data dataOperands(%arg0_devptr, %1 : !llvm.ptr, !llvm.ptr)
  acc.delete accPtr(%arg0_devptr : !llvm.ptr)
  acc.copyout accPtr(%1 : !llvm.ptr) to varPtr(%arg1 : !llvm.ptr) varType(f32)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }

// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testexitdataop;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }
// CHECK: @[[MAPNAME1:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";unknown;{{.*}};{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[MAPNAME2:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";unknown;{{.*}};{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 16, i64 18]
// CHECK: @[[MAPNAMES:.*]] = private constant [{{[0-9]*}} x ptr] [ptr @[[MAPNAME1]], ptr @[[MAPNAME2]]]

// CHECK: define void @testexitdataop(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]])
// CHECK: %[[OFFLOAD_BASEPTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAD_PTRS:.*]] = alloca [{{[0-9]*}} x ptr]
// CHECK: %[[OFFLOAS_SIZES:.*]] = alloca [{{[0-9]*}} x i64]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 1
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]


// CHECK: call void @__tgt_acc_data_exit(ptr @[[LOCGLOBAL]], i64 0, i64 1, i32 2, ptr %[[OFFLOAD_BASEPTRS]], ptr %[[OFFLOAD_PTRS]], ptr %[[OFFLOAS_SIZES]], ptr @[[MAPTYPES]], ptr @[[MAPNAMES]], ptr null, ptr null, i64 -1)

// CHECK: declare void @__tgt_acc_data_exit(ptr, i64, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)

// -----

// enter_data with attach, create_zero, if, wait and async clauses.
llvm.func @testenterdataop_clauses(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %cond: i1, %aid: i64) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr {dataClause = #acc<data_clause acc_create_zero>}
  %1 = acc.attach varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.enter_data if(%cond) wait(%aid : i64) async(%aid : i64) dataOperands(%0, %1 : !llvm.ptr, !llvm.ptr)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }

// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testenterdataop_clauses;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [2 x i64] [i64 16, i64 131088]

// CHECK: define void @testenterdataop_clauses(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]], i1 %[[COND:.*]], i64 %[[ASYNC:.*]])

// CHECK: %[[WAITLIST:.*]] = alloca [1 x i64]
// CHECK: store i64 %[[ASYNC]], ptr %[[WAITLIST_GEP:.*]]
// CHECK: call i32 @__tgt_acc_wait(ptr @[[LOCGLOBAL]], i64 0, i64 0, i32 -1, i32 1, ptr %[[WAITLIST]], i64 %[[ASYNC]])

// CHECK: br i1 %[[COND]], label %acc.standalone.then, label %acc.standalone.end

// CHECK: call void @__tgt_acc_data_enter(ptr @[[LOCGLOBAL]], i64 0, i64 1, i32 2, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr @[[MAPTYPES]], ptr @{{.*}}, ptr null, ptr null, i64 %[[ASYNC]])

// CHECK: declare void @__tgt_acc_data_enter(ptr, i64, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)

// CHECK: declare i32 @__tgt_acc_wait(ptr, i64, i64, i32, i32, ptr, i64)

// -----

// exit_data with detach, finalize, if, wait and async clauses.
llvm.func @testexitdataop_clauses(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %cond: i1, %aid: i64) {
  %arg0_devptr = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr {dataClause = #acc<data_clause acc_delete>}
  %arg1_devptr = acc.getdeviceptr varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr {dataClause = #acc<data_clause acc_detach>}
  acc.exit_data if(%cond) wait(%aid : i64) async(%aid : i64) dataOperands(%arg0_devptr, %arg1_devptr : !llvm.ptr, !llvm.ptr) attributes {finalize}
  acc.delete accPtr(%arg0_devptr : !llvm.ptr)
  acc.detach accPtr(%arg1_devptr : !llvm.ptr)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }

// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testexitdataop_clauses;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [2 x i64] [i64 24, i64 24]

// CHECK: define void @testexitdataop_clauses(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]], i1 %[[COND:.*]], i64 %[[ASYNC:.*]])

// CHECK: %[[WAITLIST:.*]] = alloca [1 x i64]
// CHECK: store i64 %[[ASYNC]], ptr %[[WAITLIST_GEP:.*]]
// CHECK: call i32 @__tgt_acc_wait(ptr @[[LOCGLOBAL]], i64 0, i64 0, i32 -1, i32 1, ptr %[[WAITLIST]], i64 %[[ASYNC]])

// CHECK: br i1 %[[COND]], label %acc.standalone.then, label %acc.standalone.end

// CHECK: call void @__tgt_acc_data_exit(ptr @[[LOCGLOBAL]], i64 0, i64 1, i32 2, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr @[[MAPTYPES]], ptr @{{.*}}, ptr null, ptr null, i64 %[[ASYNC]])

// CHECK: declare void @__tgt_acc_data_exit(ptr, i64, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)

// CHECK: declare i32 @__tgt_acc_wait(ptr, i64, i64, i32, i32, ptr, i64)

// -----

// TODO: acc.update is not lowered to LLVM IR .
// once acc.update is lowered to __tgt_acc_data_update;
// call void @__tgt_acc_data_update(ptr @{{.*}}, i64 0, i64 1, i32 1,
//       ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr @{{.*}}, ptr @{{.*}},
//       ptr null, ptr null, i64 -1)
//
// llvm.func @testupdateop(%arg1: !llvm.ptr) {
//   %0 = acc.update_device varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
//   acc.update dataOperands(%0 : !llvm.ptr)
//   llvm.return
// }

// -----

llvm.func @testdataop(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  
  %0 = acc.copyin varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  %1 = acc.create varPtr(%arg1 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data dataOperands(%0, %1 : !llvm.ptr, !llvm.ptr) {
    %9 = llvm.mlir.constant(2 : i32) : i32
    llvm.store %9, %arg2 : i32, !llvm.ptr
    acc.terminator
  }
  acc.copyout accPtr(%0 : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr) varType(f32)
  acc.copyout accPtr(%1 : !llvm.ptr) to varPtr(%arg1 : !llvm.ptr) varType(f32)
  llvm.return
}

// CHECK: %struct.ident_t = type { i32, i32, i32, i32, ptr }
// CHECK: @[[LOCSTR:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";{{.*}};testdataop;{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[LOCGLOBAL:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]*}}, ptr @[[LOCSTR]] }
// CHECK: @[[MAPNAME1:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";unknown;{{.*}};{{[0-9]*}};{{[0-9]*}};;\00"
// CHECK: @[[MAPNAME2:.*]] = private unnamed_addr constant [{{[0-9]*}} x i8] c";unknown;{{.*}};{{[0-9]*}};{{[0-9]*}};;\00"
// Entry flags: scalar copyin TO = 0x1, scalar create none = 0x0
// CHECK: @[[MAPTYPES:.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 1, i64 0]
// Exit flags: delete for both = FINALIZE = 0x8
// CHECK: @[[MAPTYPES_END:.*]] = private unnamed_addr constant [{{[0-9]*}} x i64] [i64 8, i64 8]
// CHECK: @[[MAPNAMES:.*]] = private constant [{{[0-9]*}} x ptr] [ptr @[[MAPNAME1]], ptr @[[MAPNAME2]]]

// CHECK: define void @testdataop(ptr %[[PTR0:.*]], ptr %[[PTR1:.*]], ptr %[[PTR2:.*]])
// CHECK: %[[OFFLOAD_BASEPTRS:.*]] = alloca [{{[0-9]*}} x ptr], align 8
// CHECK: %[[OFFLOAD_PTRS:.*]] = alloca [{{[0-9]*}} x ptr], align 8
// CHECK: %[[OFFLOAS_SIZES:.*]] = alloca [{{[0-9]*}} x i64], align 8

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 0
// CHECK: store ptr null, ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 0
// CHECK: store ptr %[[PTR0]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]

// CHECK: %[[OFFLOAD_BASEPTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_BASEPTRS]], i32 0, i32 1
// CHECK: store ptr null, ptr %[[OFFLOAD_BASEPTRS_GEP]]
// CHECK: %[[OFFLOAD_PTRS_GEP:.*]] = getelementptr inbounds [2 x ptr], ptr %[[OFFLOAD_PTRS]], i32 0, i32 1
// CHECK: store ptr %[[PTR1]], ptr %[[OFFLOAD_PTRS_GEP]]
// CHECK: %[[OFFLOAS_SIZES_GEP:.*]] = getelementptr inbounds [2 x i64], ptr %[[OFFLOAS_SIZES]], i32 0, i32 1
// CHECK: store i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64), ptr %[[OFFLOAS_SIZES_GEP]]

// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 2, ptr %[[OFFLOAD_BASEPTRS]], ptr %[[OFFLOAD_PTRS]], ptr %[[OFFLOAS_SIZES]], ptr @[[MAPTYPES]], ptr @[[MAPNAMES]], ptr null, ptr null, i64 -1)
// CHECK: br label %acc.data

// CHECK:      acc.data:
// CHECK-NEXT:   store i32 2, ptr %{{.*}}
// CHECK-NEXT:   br label %acc.end_data

// CHECK: acc.end_data:
// CHECK:   call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 2, ptr %[[OFFLOAD_BASEPTRS]], ptr %[[OFFLOAD_PTRS]], ptr %[[OFFLOAS_SIZES]], ptr @[[MAPTYPES_END]], ptr @[[MAPNAMES]], ptr null, ptr null, i64 -1)

// CHECK: declare void @__tgt_acc_data_begin(ptr, i64, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)
// CHECK: declare void @__tgt_acc_data_end(ptr, i64, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)

// -----

llvm.func @testpresentop(%arg0: !llvm.ptr) {
  %0 = acc.present varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  llvm.return
}

// Entry flags: present -> PRESENT | NO_CREATE = 0x102000 (1056768)
// CHECK: @[[PRESENT_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] [i64 1056768]
// CHECK: define void @testpresentop(ptr %[[PRESENT_PTR:.*]])
// CHECK: store ptr %[[PRESENT_PTR]], ptr %{{.*}}
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[PRESENT_MAPTYPES]], ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Test exit_data with copyout(zero) modifier.
// The copyout(zero) should have INIT_ZERO flag (0x20000) in addition to FROM | PTR_AND_OBJ.
// FROM = 0x2, PTR_AND_OBJ = 0x10, INIT_ZERO = 0x20000
// Expected: 0x20012 = 131090
llvm.func @testexitdata_copyout_zero(%arg0: !llvm.ptr) {
  %arg0_devptr = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr
  acc.exit_data dataOperands(%arg0_devptr : !llvm.ptr)
  acc.copyout accPtr(%arg0_devptr : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr) varType(f32) {dataClause = #acc<data_clause acc_copyout_zero>}
  llvm.return
}

// CHECK: @[[MAPTYPES_EXIT_ZERO:.*]] = private unnamed_addr constant [1 x i64] [i64 131090]
// CHECK: define void @testexitdata_copyout_zero(ptr %[[PTR0:.*]])
// CHECK: call void @__tgt_acc_data_exit(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[MAPTYPES_EXIT_ZERO]], ptr {{.*}}, ptr null, ptr null, i64 -1)

// -----

// Test data/end_data with copyout(zero) modifier.
// Entry: create with copyout_zero clause
// Exit: FROM | PTR_AND_OBJ | INIT_ZERO = 0x20012 = 131090
// For scalar types, PTR_AND_OBJ is stripped, so entry = 0, exit = FROM | INIT_ZERO = 0x20002 = 131074
llvm.func @testdata_copyout_zero(%arg0: !llvm.ptr) {
  %0 = acc.create varPtr(%arg0 : !llvm.ptr) varType(f32) -> !llvm.ptr {dataClause = #acc<data_clause acc_copyout_zero>}
  acc.data dataOperands(%0 : !llvm.ptr) {
    acc.terminator
  }
  acc.copyout accPtr(%0 : !llvm.ptr) to varPtr(%arg0 : !llvm.ptr) varType(f32) {dataClause = #acc<data_clause acc_copyout_zero>}
  llvm.return
}

// Entry flags: scalar create -> PTR_AND_OBJ stripped = 0
// CHECK: @[[DATA_ZERO_MAPTYPES:.*]] = private unnamed_addr constant [1 x i64] zeroinitializer
// Exit flags: copyout(zero) -> FROM | INIT_ZERO = 0x20002 = 131074
// CHECK: @[[DATA_ZERO_MAPTYPES_END:.*]] = private unnamed_addr constant [1 x i64] [i64 131074]
// CHECK: define void @testdata_copyout_zero(ptr %[[PTR0:.*]])
// CHECK: call void @__tgt_acc_data_begin(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[DATA_ZERO_MAPTYPES]], ptr {{.*}}, ptr null, ptr null, i64 -1)
// CHECK: call void @__tgt_acc_data_end(ptr {{.*}}, i64 0, i64 1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr @[[DATA_ZERO_MAPTYPES_END]], ptr {{.*}}, ptr null, ptr null, i64 -1)
