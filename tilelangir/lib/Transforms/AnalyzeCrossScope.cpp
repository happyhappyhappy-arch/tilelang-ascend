// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// tilelangir-analyze-cross-scope
//
// After cv-annotate has wrapped HIVM ops into cv_scope regions, this pass:
//   1. Builds an inter-scope dependency graph (shared memref analysis).
//   2. Computes stage_id for each scope via longest-path on the DAG.
//   3. Inserts tilelangir.ws_provider inside the producer scope and
//      tilelangir.ws_consumer inside consumer scopes for cross-core
//      dependencies where the shared memref lives in core-local memory.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "tilelangir-analyze-cross-scope"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRANALYZECROSSSCOPE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool isCoreLocalMemRef(MemRefType ty) {
  auto ms = ty.getMemorySpace();
  if (!ms)
    return true;
  auto hivmAS = dyn_cast<hivm::AddressSpaceAttr>(ms);
  if (!hivmAS)
    return true;
  auto as = hivmAS.getAddressSpace();
  // All non-GM address spaces are core-private and need workspace transfers
  // for cross-core access: L1/cbuf (AIC), L0C/cc (AIC), UB (AIV), etc.
  return as != hivm::AddressSpace::GM;
}

static std::optional<hivm::TCoreType> getScopeCoreType(CVScopeOp scope) {
  if (auto a = scope->getAttrOfType<IntegerAttr>("tcore_type"))
    return hivm::symbolizeTCoreType(static_cast<uint32_t>(a.getInt()));
  return std::nullopt;
}

static int32_t getScopeId(CVScopeOp scope) {
  if (auto a = scope->getAttrOfType<IntegerAttr>("scope_id"))
    return static_cast<int32_t>(a.getInt());
  return -1;
}

static bool isWrittenInScope(BlockArgument arg) {
  for (OpOperand &use : arg.getUses()) {
    Operation *user = use.getOwner();
    if (isa<memref::CopyOp>(user)) {
      if (use.getOperandNumber() == 1)
        return true;
      continue;
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
      if (dps.isDpsInit(&use))
        return true;
      continue;
    }
  }
  return false;
}

static bool isReadInScope(BlockArgument arg) {
  for (OpOperand &use : arg.getUses()) {
    Operation *user = use.getOwner();
    if (isa<WSProviderOp, WSConsumerOp>(user))
      continue;
    if (isa<memref::CopyOp>(user)) {
      if (use.getOperandNumber() == 0)
        return true;
      continue;
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
      if (dps.isDpsInput(&use))
        return true;
      continue;
    }
  }
  return false;
}

static BlockArgument findArgForInput(CVScopeOp scope, Value outerVal) {
  for (auto [operand, arg] :
       llvm::zip(scope.getInputs(), scope.getBodyBlock().getArguments())) {
    if (operand == outerVal)
      return arg;
  }
  return nullptr;
}

static Operation *findLastWriter(BlockArgument arg) {
  Operation *lastWriter = nullptr;
  for (OpOperand &use : arg.getUses()) {
    Operation *user = use.getOwner();
    if (isa<WSProviderOp>(user))
      continue;
    if (isa<memref::CopyOp>(user)) {
      if (use.getOperandNumber() == 1)
        lastWriter = user;
      continue;
    }
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(user)) {
      if (dps.isDpsInit(&use))
        lastWriter = user;
      continue;
    }
  }
  return lastWriter;
}

struct DepEdge {
  CVScopeOp producer;
  CVScopeOp consumer;
  Value sharedMemref;
  bool crossCore;
  bool needsWorkspace;
};

// ---------------------------------------------------------------------------
// Per-block analysis
// ---------------------------------------------------------------------------

static void analyzeBlock(Block &block,
                         SmallVectorImpl<CVScopeOp> &allScopes,
                         SmallVectorImpl<DepEdge> &edges) {
  SmallVector<CVScopeOp> blockScopes;
  for (Operation &op : block) {
    if (auto sc = dyn_cast<CVScopeOp>(&op)) {
      blockScopes.push_back(sc);
      allScopes.push_back(sc);
    }
    if (!isa<CVScopeOp>(&op)) {
      for (Region &region : op.getRegions())
        for (Block &inner : region)
          analyzeBlock(inner, allScopes, edges);
    }
  }

  struct ScopeMemInfo {
    CVScopeOp scope;
    bool written;
  };
  DenseMap<Value, SmallVector<ScopeMemInfo>> memrefToScopes;

  for (CVScopeOp sc : blockScopes) {
    Block &body = sc.getBodyBlock();
    for (auto [operand, arg] :
         llvm::zip(sc.getInputs(), body.getArguments())) {
      auto mt = dyn_cast<MemRefType>(operand.getType());
      if (!mt)
        continue;
      bool written = isWrittenInScope(arg);
      memrefToScopes[operand].push_back({sc, written});
    }
  }

  bool insideForLoop = block.getParentOp() &&
                       isa<scf::ForOp>(block.getParentOp());

  for (auto &[val, infos] : memrefToScopes) {
    if (infos.size() < 2)
      continue;

    auto mt = cast<MemRefType>(val.getType());
    bool coreLocal = isCoreLocalMemRef(mt);

    for (size_t i = 0; i < infos.size(); ++i) {
      for (size_t j = i + 1; j < infos.size(); ++j) {
        // Forward edge: scope_i writes → scope_j reads.
        if (infos[i].written) {
          auto prodCore = getScopeCoreType(infos[i].scope);
          auto consCore = getScopeCoreType(infos[j].scope);
          bool crossCore =
              prodCore && consCore && *prodCore != *consCore;

          edges.push_back({infos[i].scope, infos[j].scope, val,
                           crossCore, crossCore && coreLocal});

          LLVM_DEBUG(llvm::dbgs()
                     << "[analyze] edge: scope_"
                     << getScopeId(infos[i].scope) << " -> scope_"
                     << getScopeId(infos[j].scope)
                     << (crossCore ? " [cross-core]" : " [same-core]")
                     << (crossCore && coreLocal ? " [needs-ws]" : "")
                     << "\n");
        }

        // Loop-carried reverse edge: scope_j writes → scope_i reads
        // (in the next iteration). Only for cross-core deps inside a loop
        // where scope_i actually reads the memref.
        if (insideForLoop && infos[j].written && coreLocal) {
          auto prodCore = getScopeCoreType(infos[j].scope);
          auto consCore = getScopeCoreType(infos[i].scope);
          bool crossCore =
              prodCore && consCore && *prodCore != *consCore;
          if (crossCore) {
            BlockArgument argI =
                findArgForInput(infos[i].scope, val);
            if (argI && isReadInScope(argI)) {
              edges.push_back({infos[j].scope, infos[i].scope, val,
                               true, true});

              // Mark the parent scf.for: running accumulators prevent
              // multi-buffering (the serial dependency chain would break).
              auto *forOp = block.getParentOp();
              forOp->setAttr("tilelangir.has_bidirectional_ws",
                             IntegerAttr::get(
                                 IntegerType::get(forOp->getContext(), 1),
                                 1));

              LLVM_DEBUG(llvm::dbgs()
                         << "[analyze] loop-carried edge: scope_"
                         << getScopeId(infos[j].scope) << " -> scope_"
                         << getScopeId(infos[i].scope)
                         << " [cross-core] [needs-ws] [loop-carried]\n");
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stage computation
// ---------------------------------------------------------------------------

static void computeStages(ArrayRef<CVScopeOp> scopes,
                          ArrayRef<DepEdge> edges) {
  DenseMap<CVScopeOp, int32_t> stageMap;
  DenseMap<CVScopeOp, SmallVector<CVScopeOp>> predecessors;

  for (auto sc : scopes)
    stageMap[sc] = 0;

  for (auto &e : edges)
    predecessors[e.consumer].push_back(e.producer);

  bool changed = true;
  int iters = 0;
  while (changed && iters < 1000) {
    changed = false;
    ++iters;
    for (auto sc : scopes) {
      for (auto pred : predecessors[sc]) {
        int32_t candidate = stageMap[pred] + 1;
        if (candidate > stageMap[sc]) {
          stageMap[sc] = candidate;
          changed = true;
        }
      }
    }
  }

  for (auto sc : scopes) {
    sc->setAttr("stage_id",
                IntegerAttr::get(
                    IntegerType::get(sc->getContext(), 32), stageMap[sc]));
    LLVM_DEBUG(llvm::dbgs()
               << "[analyze] scope_" << getScopeId(sc)
               << " stage_id=" << stageMap[sc] << "\n");
  }
}

// ---------------------------------------------------------------------------
// Insert ws_provider / ws_consumer inside scopes
// ---------------------------------------------------------------------------

static void insertProviderConsumer(ArrayRef<DepEdge> edges,
                                   ArrayRef<CVScopeOp> /*allScopes*/,
                                   MLIRContext *ctx) {
  struct WSGroup {
    CVScopeOp producer;
    Value memref;
    SmallVector<CVScopeOp> consumers;
    WSDirection direction;
  };

  SmallVector<WSGroup> groups;

  auto findGroup = [&](CVScopeOp prod, Value mem) -> size_t {
    for (size_t i = 0; i < groups.size(); ++i)
      if (groups[i].producer == prod && groups[i].memref == mem)
        return i;
    return groups.size();
  };

  for (auto &e : edges) {
    if (!e.needsWorkspace)
      continue;

    auto prodCore = getScopeCoreType(e.producer);
    WSDirection dir = (*prodCore == hivm::TCoreType::CUBE)
                          ? WSDirection::C2V
                          : WSDirection::V2C;

    size_t idx = findGroup(e.producer, e.sharedMemref);
    if (idx == groups.size())
      groups.push_back({e.producer, e.sharedMemref, {}, dir});

    if (!llvm::is_contained(groups[idx].consumers, e.consumer))
      groups[idx].consumers.push_back(e.consumer);
  }

  OpBuilder builder(ctx);

  for (auto [channelIdx, grp] : llvm::enumerate(groups)) {
    int32_t chId = static_cast<int32_t>(channelIdx);
    auto srcTy = cast<MemRefType>(grp.memref.getType());

    // --- Provider: insert inside producer scope body ---
    BlockArgument prodArg = findArgForInput(grp.producer, grp.memref);
    if (!prodArg)
      continue;

    {
      OpBuilder::InsertionGuard guard(builder);
      Operation *writer = findLastWriter(prodArg);
      if (writer)
        builder.setInsertionPointAfter(writer);
      else
        builder.setInsertionPoint(grp.producer.getBodyBlock().getTerminator());

      Value flag = builder.create<arith::ConstantIndexOp>(
          grp.producer.getLoc(), chId);
      builder.create<WSProviderOp>(
          grp.producer.getLoc(), prodArg, chId, flag, grp.direction);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[analyze] ws_provider in scope_"
               << getScopeId(grp.producer)
               << " channel=" << chId
               << " dir=" << stringifyWSDirection(grp.direction) << "\n");

    // --- Consumer(s): insert inside each consumer scope body ---
    auto resultTy = MemRefType::get(srcTy.getShape(), srcTy.getElementType());

    for (CVScopeOp consSc : grp.consumers) {
      BlockArgument consArg = findArgForInput(consSc, grp.memref);
      if (!consArg)
        continue;

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&consSc.getBodyBlock());

      Value flag = builder.create<arith::ConstantIndexOp>(
          consSc.getLoc(), chId);
      auto consOp = builder.create<WSConsumerOp>(
          consSc.getLoc(), resultTy, chId, flag, grp.direction);

      consArg.replaceAllUsesExcept(consOp.getResult(), consOp);

      LLVM_DEBUG(llvm::dbgs()
                 << "[analyze] ws_consumer in scope_"
                 << getScopeId(consSc)
                 << " channel=" << chId << "\n");
    }
  }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct TileLangIRAnalyzeCrossScope
    : impl::TileLangIRAnalyzeCrossScopeBase<TileLangIRAnalyzeCrossScope> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      if (!func->hasAttr("hivm.func_core_type"))
        return;

      LLVM_DEBUG(llvm::dbgs() << "[analyze] processing function: "
                               << func.getName() << "\n");

      SmallVector<CVScopeOp> allScopes;
      SmallVector<DepEdge> edges;

      for (Region &region : func->getRegions())
        for (Block &block : region)
          analyzeBlock(block, allScopes, edges);

      if (allScopes.empty())
        return;

      computeStages(allScopes, edges);
      insertProviderConsumer(edges, allScopes, &getContext());
    });
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
