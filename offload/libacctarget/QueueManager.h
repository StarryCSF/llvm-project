//===- QueueManager.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _QUEUEMANAGER_H_
#define _QUEUEMANAGER_H_

#include "omptarget.h"

namespace llvm::acc::target {

using QueueIdTy = uint32_t;

class QueueManagerTy {
public:
  QueueManagerTy();
  ~QueueManagerTy();

  void init() {}
  void deinit() {}

  enum class StatusTy { READY = 0, NOT_READY = 1 };

  AsyncInfoTy *get(DeviceTy &Device, QueueIdTy QueueId);

  void synchronize(DeviceTy &Device, QueueIdTy Queue);
  void synchronize(DeviceTy &Device);
  void synchronize();

  StatusTy query(DeviceTy &Device, QueueIdTy Queue);
  StatusTy query(DeviceTy &Device);
  StatusTy query();

  /// Asynchronously insert a wait dependency from \p SrcAsyncInfo into
  /// \p DstAsyncInfo using an event. This is non-blocking for the host:
  /// subsequent work enqueued on \p DstAsyncInfo will not start until the
  /// pending operations on \p SrcAsyncInfo complete. The temporary event is
  /// destroyed once \p DstAsyncInfo is later synchronized.
  void waitAsync(DeviceTy &Device, AsyncInfoTy &SrcAsyncInfo,
                 AsyncInfoTy &DstAsyncInfo);

  /// Asynchronously insert wait dependencies from every stream of \p Device
  /// (except \p DstAsyncInfo itself) into \p DstAsyncInfo.
  void waitAllAsync(DeviceTy &Device, AsyncInfoTy &DstAsyncInfo);

  /// Asynchronously insert wait dependencies from every stream of every
  /// device that currently has streams into each device's own \p DstQueue
  /// stream. Avoids cross-device event dependencies.
  void waitAllAsyncAllDevices(QueueIdTy DstQueue);

private:
  std::map<std::pair<DeviceTy *, QueueIdTy>, std::unique_ptr<AsyncInfoTy>>
      QueueMap;
};

extern QueueManagerTy *QueueManager;

class QueueAsyncInfoWrapperTy {
  AsyncInfoTy *AsyncInfo;

public:
  QueueAsyncInfoWrapperTy(DeviceTy &Device, QueueIdTy QueueId) {
    AsyncInfo = QueueManager->get(Device, QueueId);
  }

  ~QueueAsyncInfoWrapperTy() {}

  operator AsyncInfoTy &() { return *AsyncInfo; }
};

extern QueueManagerTy *QueueManager;

void accAsyncWait(ident_t *Loc, int64_t DeviceId, int64_t WaitArg);
void accAsyncWait(ident_t *Loc, int64_t DeviceId, uint32_t WaitNum,
                  int64_t *WaitList);
void accAsyncWaitAll(ident_t *Loc, int64_t DeviceId);
void accAsyncWaitAll(ident_t *Loc);
void accAsyncWaitAsync(ident_t *Loc, int64_t DeviceId, int64_t WaitArg,
                       int64_t AsyncArg);
void accAsyncWaitAsync(ident_t *Loc, int64_t DeviceId, uint32_t WaitNum,
                       int64_t *WaitList, int64_t AsyncArg);
void accAsyncWaitAllAsync(ident_t *Loc, int64_t DeviceId, int64_t AsyncArg);
void accAsyncWaitAllAsync(ident_t *Loc, int64_t AsyncArg);
int accAsyncTest(ident_t *Loc, int64_t DeviceId, int64_t TestArg);
int accAsyncTest(ident_t *Loc, int64_t DeviceId, uint32_t TestNum,
                 int64_t *TestList);
int accAsyncTestAll(ident_t *Loc, int64_t DeviceId);
int accAsyncTestAll(ident_t *Loc);
} // namespace llvm::acc::target

namespace llvm::acc::target::icv {
// acc-default-async-var
extern thread_local int32_t AccDefaultAsyncVar;
} // namespace llvm::acc::target::icv

#endif // _QUEUEMANAGER_H_
