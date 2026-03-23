// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// tilelangir-inject-block-sync
//
// Injects cross-core synchronization (sync_block_set / sync_block_wait)
// between CUBE and VECTOR scopes.
//
// Supports two layouts:
//   1. Single-buffer: scopes directly in a for-loop body
//   2. Multi-buffer (unrolled): N copies of each scope with version_id attr
//
// For multi-buffer, each (transition, version) pair gets its own static
// flag pair (data-ready + done-reading).  This enables true N-version
// pipeline concurrency across CUBE and VECTOR cores.
// Flag IDs must be < 16 (hardware limit on Ascend NPU).
//
// Pipe barrier: we insert pipe_barrier(PIPE_ALL) around sync signals in
// VECTOR scopes to ensure DMA completion before cross-core signaling.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-inject-block-sync"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINJECTBLOCKSYNC
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

using CT = hivm::TCoreType;
using PP = hivm::PIPE;

// ---------------------------------------------------------------------------
// Emit helpers — static flag only (no dynamic flags needed with unrolling)
// ---------------------------------------------------------------------------

static void emitSyncSet(OpBuilder &b, Location loc,
                        CT coreType, PP tpipe, int flag) {
  auto *ctx = b.getContext();
  OperationState s(loc, "hivm.hir.sync_block_set");
  s.addAttribute("tcore_type", hivm::TCoreTypeAttr::get(ctx, coreType));
  s.addAttribute("tpipe", hivm::PipeAttr::get(ctx, tpipe));
  s.addAttribute("pipe", hivm::PipeAttr::get(ctx, PP::PIPE_S));
  s.addAttribute("static_flag_id", b.getI64IntegerAttr(flag));
  s.addAttribute("operandSegmentSizes",
                  DenseI32ArrayAttr::get(ctx, {0, 0}));
  s.addAttribute("tsync_instr_mode",
                  hivm::SyncBlockInstrModeAttr::get(
                      ctx,
                      hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION));
  b.create(s);
}

static void emitSyncWait(OpBuilder &b, Location loc,
                         CT coreType, PP waitPipe, int flag) {
  auto *ctx = b.getContext();
  OperationState s(loc, "hivm.hir.sync_block_wait");
  s.addAttribute("tcore_type", hivm::TCoreTypeAttr::get(ctx, coreType));
  s.addAttribute("tpipe", hivm::PipeAttr::get(ctx, PP::PIPE_S));
  s.addAttribute("pipe", hivm::PipeAttr::get(ctx, waitPipe));
  s.addAttribute("static_flag_id", b.getI64IntegerAttr(flag));
  b.create(s);
}

static void emitPipeBarrier(OpBuilder &b, Location loc) {
  auto *ctx = b.getContext();
  OperationState s(loc, "hivm.hir.pipe_barrier");
  s.addAttribute("pipe", hivm::PipeAttr::get(ctx, PP::PIPE_ALL));
  b.create(s);
}

static CT getCoreType(CVScopeOp sc) {
  auto a = sc->getAttrOfType<IntegerAttr>("tcore_type");
  return static_cast<CT>(a.getInt());
}

static PP producerPipe(CT core) {
  return core == CT::CUBE ? PP::PIPE_FIX : PP::PIPE_MTE3;
}

static CVScopeOp createSyncScope(OpBuilder &b, Location loc, CT coreType) {
  return b.create<CVScopeOp>(
      loc, ValueRange{},
      static_cast<int32_t>(coreType), /*stageId=*/0, /*scopeId=*/-1);
}

// ---------------------------------------------------------------------------
// Sync unit — a cv_scope with optional version metadata
// ---------------------------------------------------------------------------

struct SyncUnit {
  CVScopeOp scope;
  CT coreType;
  int versionId;    // -1 = not versioned
  int versionCount; // 1 = single-buffer
};

static SmallVector<SyncUnit> collectSyncUnits(Block &body) {
  SmallVector<SyncUnit> units;
  for (auto &op : body) {
    if (auto sc = dyn_cast<CVScopeOp>(&op)) {
      int vid = -1, vcnt = 1;
      if (auto a = sc->getAttrOfType<IntegerAttr>("version_id"))
        vid = a.getInt();
      if (auto a = sc->getAttrOfType<IntegerAttr>("version_count"))
        vcnt = a.getInt();
      units.push_back({sc, getCoreType(sc), vid, vcnt});
    }
  }
  return units;
}

// ---------------------------------------------------------------------------
// Version group: a run of same-coreType scopes with sequential version_ids.
// ---------------------------------------------------------------------------

struct VersionGroup {
  CT coreType;
  SmallVector<SyncUnit *> versions; // ordered by version_id
};

static SmallVector<VersionGroup> buildVersionGroups(
    SmallVector<SyncUnit> &units) {
  SmallVector<VersionGroup> groups;
  size_t i = 0;
  while (i < units.size()) {
    VersionGroup g;
    g.coreType = units[i].coreType;
    g.versions.push_back(&units[i]);
    ++i;
    while (i < units.size() && units[i].coreType == g.coreType) {
      g.versions.push_back(&units[i]);
      ++i;
    }
    groups.push_back(std::move(g));
  }
  return groups;
}

// ---------------------------------------------------------------------------
// Inject per-version sync between cross-core version groups.
// ---------------------------------------------------------------------------

static void injectVersionedSync(scf::ForOp forOp,
                                SmallVector<SyncUnit> &units,
                                OpBuilder &b) {
  auto groups = buildVersionGroups(units);
  if (groups.size() < 2)
    return;

  Location loc = forOp.getLoc();
  int nextFlag = 0;

  struct TransitionInfo {
    VersionGroup *producer;
    VersionGroup *consumer;
    int flagDataReadyBase;
    int flagDoneReadingBase;
    int numVersions;
  };

  SmallVector<TransitionInfo> transitions;

  for (size_t gi = 0; gi + 1 < groups.size(); ++gi) {
    if (groups[gi].coreType == groups[gi + 1].coreType)
      continue;

    int nv = std::max(groups[gi].versions.size(),
                      groups[gi + 1].versions.size());
    TransitionInfo t;
    t.producer = &groups[gi];
    t.consumer = &groups[gi + 1];
    t.flagDataReadyBase = nextFlag;
    nextFlag += nv;
    t.flagDoneReadingBase = nextFlag;
    nextFlag += nv;
    t.numVersions = nv;
    transitions.push_back(t);
  }

  if (transitions.empty())
    return;

  assert(nextFlag <= 16 && "flag IDs must be < 16 (hardware limit)");

  LLVM_DEBUG(llvm::dbgs()
             << "[sync] " << transitions.size()
             << " transition(s), " << nextFlag << " flags\n");

  // Insert per-version sync
  for (auto &t : transitions) {
    for (int v = 0; v < t.numVersions; ++v) {
      int flagDR = t.flagDataReadyBase + v;
      int flagDone = t.flagDoneReadingBase + v;

      bool prodIsVec = t.producer->coreType == CT::VECTOR;
      bool consIsVec = t.consumer->coreType == CT::VECTOR;

      // Producer version v: set data_ready[v].
      // If there are GM-writing ops (memref.copy to gm) between the
      // producer scope and the consumer scope, the flag must fire AFTER
      // those ops — not inside the producer scope.
      if (v < static_cast<int>(t.producer->versions.size())) {
        SyncUnit *pu = t.producer->versions[v];
        bool deferFlag = false;
        if (v < static_cast<int>(t.consumer->versions.size())) {
          auto *consOp = t.consumer->versions[v]->scope.getOperation();
          for (auto *prev = consOp->getPrevNode(); prev;
               prev = prev->getPrevNode()) {
            if (isa<CVScopeOp>(prev))
              break;
            if (auto copy = dyn_cast<memref::CopyOp>(prev)) {
              auto dstTy =
                  cast<MemRefType>(copy.getTarget().getType());
              if (dstTy.getMemorySpace()) {
                deferFlag = true;
                break;
              }
            }
          }
        }
        if (deferFlag) {
          auto *consOp = t.consumer->versions[v]->scope.getOperation();
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(consOp);
          CVScopeOp syncSc =
              createSyncScope(b, loc, t.producer->coreType);
          b.setInsertionPoint(syncSc.getBodyBlock().getTerminator());
          if (prodIsVec) emitPipeBarrier(b, loc);
          emitSyncSet(b, loc, t.producer->coreType,
                      producerPipe(t.producer->coreType), flagDR);
        } else {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(
              pu->scope.getBodyBlock().getTerminator());
          if (prodIsVec) emitPipeBarrier(b, loc);
          emitSyncSet(b, loc, t.producer->coreType,
                      producerPipe(t.producer->coreType), flagDR);
        }
      }

      // Consumer version v: at start of scope, wait data_ready[v]
      if (v < static_cast<int>(t.consumer->versions.size())) {
        SyncUnit *cu = t.consumer->versions[v];
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(&cu->scope.getBodyBlock());
        if (consIsVec) emitPipeBarrier(b, loc);
        emitSyncWait(b, loc, t.consumer->coreType, PP::PIPE_MTE2, flagDR);
      }

      // Consumer version v: at end of scope, set done_reading[v]
      if (v < static_cast<int>(t.consumer->versions.size())) {
        SyncUnit *cu = t.consumer->versions[v];
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPoint(cu->scope.getBodyBlock().getTerminator());
        if (consIsVec) emitPipeBarrier(b, loc);
        emitSyncSet(b, loc, t.consumer->coreType,
                    producerPipe(t.consumer->coreType), flagDone);
      }

      // Producer version v: at start of scope, wait done_reading[v]
      if (v < static_cast<int>(t.producer->versions.size())) {
        SyncUnit *pu = t.producer->versions[v];
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(&pu->scope.getBodyBlock());
        if (prodIsVec) emitPipeBarrier(b, loc);
        emitSyncWait(b, loc, t.producer->coreType, PP::PIPE_MTE2, flagDone);
      }
    }
  }

  // Pre-loop: initialize done_reading flags
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(forOp);
    for (auto &t : transitions) {
      for (int v = 0; v < t.numVersions; ++v) {
        CVScopeOp sc = createSyncScope(b, loc, t.consumer->coreType);
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPoint(sc.getBodyBlock().getTerminator());
        emitSyncSet(b, loc, t.consumer->coreType,
                    producerPipe(t.consumer->coreType),
                    t.flagDoneReadingBase + v);
      }
    }
  }

  // Post-loop: wait outstanding done_reading
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(forOp);
    for (auto &t : transitions) {
      bool prodIsVec = t.producer->coreType == CT::VECTOR;
      for (int v = 0; v < t.numVersions; ++v) {
        CVScopeOp sc = createSyncScope(b, loc, t.producer->coreType);
        OpBuilder::InsertionGuard g2(b);
        b.setInsertionPoint(sc.getBodyBlock().getTerminator());
        if (prodIsVec) emitPipeBarrier(b, loc);
        emitSyncWait(b, loc, t.producer->coreType, PP::PIPE_MTE2,
                     t.flagDoneReadingBase + v);
        if (prodIsVec) emitPipeBarrier(b, loc);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct TileLangIRInjectBlockSync
    : impl::TileLangIRInjectBlockSyncBase<TileLangIRInjectBlockSync> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    OpBuilder b(func.getContext());

    scf::ForOp forOp;
    for (auto &op : func.getBody().front()) {
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        forOp = f;
        break;
      }
    }
    if (!forOp)
      return;

    auto units = collectSyncUnits(*forOp.getBody());
    if (units.size() < 2)
      return;

    injectVersionedSync(forOp, units, b);
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
