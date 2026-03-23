// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// Stage 4 – tilelangir-insert-vid (spike / 穿刺)
//
// Write-guard mode: both Vector Sub-Blocks redundantly compute the same
// softmax. Only VID==0 executes workspace writes and GM write-back.
// This avoids cross-sub-block reduce-merge synchronization.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-insert-vid"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINSERTVID
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

static bool isVectorScope(CVScopeOp sc) {
  if (auto a = sc->getAttrOfType<IntegerAttr>("tcore_type"))
    return a.getInt() == static_cast<int64_t>(hivm::TCoreType::VECTOR);
  return false;
}

static bool isWriteOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "hivm.hir.store")
    return true;
  if (name == "memref.copy") {
    if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      auto srcAS = hivm::getHIVMAddressSpace(copyOp.getSource().getType());
      if (srcAS == hivm::AddressSpace::GM)
        return false;
    }
    return true;
  }
  return false;
}

/// Wrap a single write operation with `scf.if %cond { <op> }`.
/// If \p addLimitAttr is true, also add `{limit_sub_block_id0}`.
static void wrapWithVIDGuard(OpBuilder &b, Operation *writeOp,
                             Value isVid0, bool addLimitAttr) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(writeOp);
  Location loc = writeOp->getLoc();

  auto ifOp = b.create<scf::IfOp>(loc, TypeRange{}, isVid0,
                                   /*withElseRegion=*/false);
  if (addLimitAttr)
    ifOp->setAttr("limit_sub_block_id0", b.getUnitAttr());

  Block *thenBlock = &ifOp.getThenRegion().front();
  writeOp->moveBefore(thenBlock->getTerminator());
}

/// Insert VID computation at the start of a Vector scope body:
///   %vid = hivm.hir.get_sub_block_idx -> i64
///   %vid_idx = arith.index_cast %vid : i64 to index
///   %c0 = arith.constant 0 : index
///   %is_vid0 = arith.cmpi eq, %vid_idx, %c0 : index
static Value insertVIDCheck(OpBuilder &b, Block &body, Location loc) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&body);

  OperationState s(loc, "hivm.hir.get_sub_block_idx");
  s.addTypes({b.getI64Type()});
  Operation *getVid = b.create(s);
  Value vid = getVid->getResult(0);

  Value vidIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), vid);
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value isVid0 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          vidIdx, c0);
  return isVid0;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct TileLangIRInsertVID
    : impl::TileLangIRInsertVIDBase<TileLangIRInsertVID> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr("hivm.func_core_type"))
      return;

    OpBuilder b(func.getContext());

    SmallVector<CVScopeOp> vectorScopes;
    func.walk([&](CVScopeOp sc) {
      if (isVectorScope(sc))
        vectorScopes.push_back(sc);
    });

    for (auto sc : vectorScopes)
      processVectorScope(b, sc);

    LLVM_DEBUG(llvm::dbgs() << "[vid] insert_vid done\n");
  }

  void processVectorScope(OpBuilder &b, CVScopeOp sc) {
    Block &body = sc.getBody().front();
    Location loc = sc.getLoc();

    SmallVector<Operation *> writes;
    for (auto &op : body)
      if (isWriteOp(&op))
        writes.push_back(&op);

    if (writes.empty())
      return;

    Value isVid0 = insertVIDCheck(b, body, loc);

    bool isPostLoop = isPostLoopScope(sc);
    for (auto *writeOp : writes) {
      bool addLimit = isPostLoop && writeOp->getName().getStringRef() ==
                                        "memref.copy";
      wrapWithVIDGuard(b, writeOp, isVid0, addLimit);
      LLVM_DEBUG(llvm::dbgs()
                 << "[vid] wrapped " << writeOp->getName() << "\n");
    }
  }

  bool isPostLoopScope(CVScopeOp sc) {
    Operation *parent = sc->getParentOp();
    return !isa<scf::ForOp>(parent);
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
