//===-- FIRToSCF.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_FIRTOSCFPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;

namespace {
class FIRToSCFPass : public fir::impl::FIRToSCFPassBase<FIRToSCFPass> {
public:
  void runOnOperation() override;
};

struct DoLoopConversion : public OpRewritePattern<fir::DoLoopOp> {
  using OpRewritePattern<fir::DoLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(fir::DoLoopOp doLoopOp,
                                PatternRewriter &rewriter) const override {
    auto loc = doLoopOp.getLoc();
    bool hasFinalValue = doLoopOp.getFinalValue().has_value();

    // Get loop values from the DoLoopOp
    auto low = doLoopOp.getLowerBound();
    auto high = doLoopOp.getUpperBound();
    assert(low && high && "must be a Value");
    auto step = doLoopOp.getStep();
    llvm::SmallVector<Value> iterArgs;
    if (hasFinalValue)
      iterArgs.push_back(low);
    iterArgs.append(doLoopOp.getIterOperands().begin(),
                    doLoopOp.getIterOperands().end());

    // fir.do_loop iterates over the interval [%l, %u], and the step may be
    // negative. But scf.for iterates over the interval [%l, %u), and the step
    // must be a positive value.
    // For easier conversion, we calculate the trip count and use a canonical
    // induction variable.
    auto diff = rewriter.create<arith::SubIOp>(loc, high, low);
    auto distance = rewriter.create<arith::AddIOp>(loc, diff, step);
    auto tripCount = rewriter.create<arith::DivSIOp>(loc, distance, step);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto scfForOp =
        rewriter.create<scf::ForOp>(loc, zero, tripCount, one, iterArgs);

    auto &loopOps = doLoopOp.getBody()->getOperations();
    auto resultOp = cast<fir::ResultOp>(doLoopOp.getBody()->getTerminator());
    auto results = resultOp.getOperands();
    Block *loweredBody = scfForOp.getBody();

    loweredBody->getOperations().splice(loweredBody->begin(), loopOps,
                                        loopOps.begin(),
                                        std::prev(loopOps.end()));

    rewriter.setInsertionPointToStart(loweredBody);
    Value iv =
        rewriter.create<arith::MulIOp>(loc, scfForOp.getInductionVar(), step);
    iv = rewriter.create<arith::AddIOp>(loc, low, iv);

    if (!results.empty()) {
      rewriter.setInsertionPointToEnd(loweredBody);
      rewriter.create<scf::YieldOp>(resultOp->getLoc(), results);
    }
    doLoopOp.getInductionVar().replaceAllUsesWith(iv);
    rewriter.replaceAllUsesWith(doLoopOp.getRegionIterArgs(),
                                hasFinalValue
                                    ? scfForOp.getRegionIterArgs().drop_front()
                                    : scfForOp.getRegionIterArgs());

    // Copy all the attributes from the old to new op.
    scfForOp->setAttrs(doLoopOp->getAttrs());
    rewriter.replaceOp(doLoopOp, scfForOp);
    return success();
  }
};
struct ResultConversion : public OpRewritePattern<fir::ResultOp> {
  using OpRewritePattern<fir::ResultOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::ResultOp resultOp,
                                PatternRewriter &rewriter) const override {
    auto loc = resultOp.getLoc();
    //  create scf.yieldOp
    auto scfYieldOp = rewriter.create<scf::YieldOp>(loc, resultOp.getOperands());
    // copy fir.result attributes to scf.yield
    scfYieldOp->setAttrs(resultOp->getAttrs());
    rewriter.replaceOp(resultOp, scfYieldOp.getResults()); 
    return success();
  }
};

struct IfConversion : public OpRewritePattern<fir::IfOp> {
  using OpRewritePattern<fir::IfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(fir::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto loc = ifOp.getLoc();
    Value condition = ifOp.getCondition();
    TypeRange resultTypes = ifOp.getResultTypes();
    // create ifOp
    auto scfIfOp = rewriter.create<scf::IfOp>(loc, resultTypes, condition,/*withElseRegion=*/!ifOp.getElseRegion().empty()); 
    // Helper function
    auto handleRegion = [&](Region &sourceRegion, Region &targetRegion)  {
      if (sourceRegion.empty())return failure();
      Block &sourceBlock = sourceRegion.front();
      if (sourceBlock.empty() || !sourceBlock.back().mightHaveTrait<OpTrait::IsTerminator>()) {
        return rewriter.notifyMatchFailure(ifOp, "Region block does not have a terminator");
      }
      // Get Terminate operation,such as fir.result
      Operation *terminator = sourceBlock.getTerminator();
      SmallVector<Value> terminatorOperands(terminator->getOperands());
      if (terminatorOperands.size() != resultTypes.size()) {
        return rewriter.notifyMatchFailure(
            ifOp, "Mismatch in number of result operands");
      }
      // Move all operations ,except the terminator
      Block &targetBlock = targetRegion.front();
      rewriter.setInsertionPointToStart(&targetBlock);
      auto &sourceOps = sourceBlock.getOperations();
      // Remove the terminator operation
      sourceOps.pop_back(); 
      targetBlock.getOperations().splice(targetBlock.begin(), sourceOps);
      // scf.yield
      rewriter.setInsertionPointToEnd(&targetBlock);
      if (!targetBlock.empty() && isa<scf::YieldOp>(targetBlock.back())) {
        rewriter.eraseOp(&targetBlock.back());
      }
      rewriter.create<scf::YieldOp>(loc, terminatorOperands);
      return success();
    };

    // then Region
    if (failed(handleRegion(ifOp.getThenRegion(), scfIfOp.getThenRegion())))
      return failure();

    // else Region
    if (!ifOp.getElseRegion().empty()) {
      if (failed(handleRegion(ifOp.getElseRegion(), scfIfOp.getElseRegion())))
        return failure();
    }

    // Copy all the attributes from the old to new op
    scfIfOp->setAttrs(ifOp->getAttrs());

    // returns ,at most two
    auto results = scfIfOp.getResults();
    if (results.size() == 2) {
      rewriter.replaceOp(ifOp, {results[0], results[1]});
    } else {
      rewriter.replaceOp(ifOp, scfIfOp.getResults());
    }
    
    return success();
  }
};
} // namespace

void FIRToSCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<DoLoopConversion>(patterns.getContext());
  ConversionTarget target(getContext());
  target.addIllegalOp<fir::DoLoopOp>();
  patterns.add<IfConversion>(patterns.getContext());
  target.addIllegalOp<fir::IfOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> fir::createFIRToSCFPass() {
  return std::make_unique<FIRToSCFPass>();
}
