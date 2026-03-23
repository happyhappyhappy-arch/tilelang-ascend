// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// tilelangir-materialize-workspace
//
// Consumes tilelangir.ws_provider / ws_consumer ops created by
// analyze-cross-scope and materializes them into HIVM DMA or GM-bypass:
//
//   c2v (nd2nz destination): bypass — clone GM source for VECTOR to load.
//   c2v (general):           fixpipe in producer CUBE scope, memref.copy in consumer.
//   v2c:                     memref.copy in producer VECTOR scope,
//                            hivm.hir.load in consumer (→ nd2nz after outline).
//
// Allocates versioned workspace GM sub-regions from the function's workspace
// argument and annotates the function with total workspace byte size.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-materialize-workspace"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRMATERIALIZEWORKSPACE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int32_t getScopeId(CVScopeOp scope) {
  if (auto a = scope->getAttrOfType<IntegerAttr>("scope_id"))
    return static_cast<int32_t>(a.getInt());
  return -1;
}

static BlockArgument addScopeInput(CVScopeOp scope, Value val) {
  unsigned n = scope.getInputs().size();
  scope->insertOperands(n, {val});
  return scope.getBody().front().addArgument(val.getType(), scope.getLoc());
}

static void emitLoad(OpBuilder &b, Location loc, Value src, Value dst) {
  OperationState s(loc, "hivm.hir.load");
  s.addOperands({src, dst});
  s.addAttribute("init_out_buffer", b.getBoolAttr(false));
  s.addAttribute("may_implicit_transpose_with_last_axis", b.getBoolAttr(false));
  s.addAttribute("operandSegmentSizes",
                  DenseI32ArrayAttr::get(b.getContext(), {1, 1, 0, 0, 0, 0}));
  b.create(s);
}

static void emitStore(OpBuilder &b, Location loc, Value src, Value dst) {
  OperationState s(loc, "hivm.hir.store");
  s.addOperands({src, dst});
  b.create(s);
}

static void emitFixpipe(OpBuilder &b, Location loc, Value src, Value dst) {
  OperationState s(loc, "hivm.hir.fixpipe");
  s.addOperands({src, dst});
  s.addAttribute("enable_nz2nd", b.getUnitAttr());
  b.create(s);
}

static int64_t getMemRefByteSize(MemRefType ty) {
  int64_t numElements = 1;
  for (int64_t dim : ty.getShape()) {
    if (ShapedType::isDynamic(dim))
      return -1;
    numElements *= dim;
  }
  unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
  return numElements * (bitWidth / 8);
}

static Value findWorkspaceArg(func::FuncOp func) {
  int dynIdx = 0;
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    auto mt = dyn_cast<MemRefType>(func.getArgument(i).getType());
    if (!mt)
      continue;
    if (mt.getElementType().isInteger(8) && mt.getRank() == 1 &&
        mt.isDynamicDim(0)) {
      if (dynIdx == 1)
        return func.getArgument(i);
      ++dynIdx;
    }
  }
  return nullptr;
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

// ---------------------------------------------------------------------------
// C→V nd2nz bypass
// ---------------------------------------------------------------------------

static bool isNd2nzDestination(CVScopeOp prodScope, Value data) {
  for (Operation &op : prodScope.getBodyBlock()) {
    if (op.getName().getStringRef() == "hivm.hir.nd2nz" &&
        op.getNumOperands() >= 2 && op.getOperand(1) == data)
      return true;
  }
  return false;
}

static void handleC2VNd2nzBypass(WSProviderOp provOp,
                                 SmallVector<WSConsumerOp> &consumers,
                                 func::FuncOp func, OpBuilder &b) {
  CVScopeOp prodScope = provOp->getParentOfType<CVScopeOp>();
  Block &prodBody = prodScope.getBodyBlock();
  Value data = provOp.getData();

  Operation *nd2nzOp = nullptr;
  for (Operation &op : prodBody) {
    if (op.getName().getStringRef() != "hivm.hir.nd2nz")
      continue;
    if (op.getNumOperands() >= 2 && op.getOperand(1) == data) {
      nd2nzOp = &op;
      break;
    }
  }
  if (!nd2nzOp)
    return;

  Value srcInside = nd2nzOp->getOperand(0);
  SmallVector<Operation *> opsToClone;
  SmallVector<Value> traceWorklist{srcInside};
  DenseSet<Operation *> visited;

  while (!traceWorklist.empty()) {
    Value v = traceWorklist.pop_back_val();
    if (auto *defOp = v.getDefiningOp()) {
      if (defOp->getParentOp() == prodScope.getOperation() &&
          visited.insert(defOp).second) {
        opsToClone.push_back(defOp);
        for (Value operand : defOp->getOperands())
          traceWorklist.push_back(operand);
      }
    }
  }

  std::reverse(opsToClone.begin(), opsToClone.end());

  IRMapping mapping;
  for (unsigned i = 0; i < prodBody.getNumArguments(); ++i)
    mapping.map(prodBody.getArgument(i), prodScope.getInputs()[i]);

  b.setInsertionPointAfter(prodScope);
  for (auto *op : opsToClone)
    b.clone(*op, mapping);

  Value gmSource = mapping.lookupOrDefault(srcInside);

  auto srcTy = cast<MemRefType>(provOp.getData().getType());
  auto localTy = MemRefType::get(srcTy.getShape(), srcTy.getElementType());

  for (auto consOp : consumers) {
    CVScopeOp consSc = consOp->getParentOfType<CVScopeOp>();

    BlockArgument gmArg = addScopeInput(consSc, gmSource);

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(consOp);

    auto localAlloc = b.create<memref::AllocOp>(consSc.getLoc(), localTy);
    b.create<memref::CopyOp>(consSc.getLoc(), gmArg, localAlloc.getResult());

    consOp.getResult().replaceAllUsesWith(localAlloc.getResult());
    consOp->erase();

    LLVM_DEBUG(llvm::dbgs()
               << "[materialize] c2v nd2nz bypass for consumer scope_"
               << getScopeId(consSc) << "\n");
  }

  provOp->erase();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct TileLangIRMaterializeWorkspace
    : impl::TileLangIRMaterializeWorkspaceBase<
          TileLangIRMaterializeWorkspace> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp f) {
      if (f->hasAttr("hivm.func_core_type"))
        funcs.push_back(f);
    });

    for (auto f : funcs)
      processFunc(f, module);
  }

  void processFunc(func::FuncOp func, ModuleOp module) {
    MLIRContext *ctx = func.getContext();
    OpBuilder b(ctx);

    // Group providers and consumers by channel_id.
    struct ChannelInfo {
      WSDirection direction;
      SmallVector<WSProviderOp> providers;
      SmallVector<WSConsumerOp> consumers;
    };
    DenseMap<int32_t, ChannelInfo> channels;

    func.walk([&](WSProviderOp op) {
      auto &info = channels[op.getChannelId()];
      info.direction = op.getDirection();
      info.providers.push_back(op);
    });
    func.walk([&](WSConsumerOp op) {
      channels[op.getChannelId()].consumers.push_back(op);
    });

    if (channels.empty())
      return;

    // Classify: nd2nz-bypass (no workspace) vs needs-workspace.
    DenseSet<int32_t> bypassChannels;
    for (auto &[chId, info] : channels) {
      if (info.direction != WSDirection::C2V || info.providers.empty())
        continue;
      CVScopeOp prodScope = info.providers[0]->getParentOfType<CVScopeOp>();
      if (prodScope && isNd2nzDestination(prodScope, info.providers[0].getData()))
        bypassChannels.insert(chId);
    }

    // Handle nd2nz bypasses first.
    // Pre-build version-matched pairs before any erasure.
    for (int32_t chId : bypassChannels) {
      auto &info = channels[chId];

      struct BypassPair {
        WSProviderOp prov;
        SmallVector<WSConsumerOp> cons;
      };
      SmallVector<BypassPair> pairs;

      for (auto provOp : info.providers) {
        CVScopeOp provScope = provOp->getParentOfType<CVScopeOp>();
        int32_t provVer = 0;
        if (auto v = provScope->getAttrOfType<IntegerAttr>("version_id"))
          provVer = v.getInt();
        BypassPair bp;
        bp.prov = provOp;
        for (auto consOp : info.consumers) {
          CVScopeOp consScope = consOp->getParentOfType<CVScopeOp>();
          int32_t consVer = 0;
          if (auto v = consScope->getAttrOfType<IntegerAttr>("version_id"))
            consVer = v.getInt();
          if (consVer == provVer)
            bp.cons.push_back(consOp);
        }
        pairs.push_back(std::move(bp));
      }

      for (auto &bp : pairs)
        handleC2VNd2nzBypass(bp.prov, bp.cons, func, b);
      channels.erase(chId);
    }

    if (channels.empty())
      return;

    // Compute workspace layout.
    Value wsArg = findWorkspaceArg(func);
    Value blockI64;
    if (wsArg) {
      func.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "hivm.hir.get_block_idx")
          blockI64 = op->getResult(0);
      });
    }

    int64_t totalWsBytes = 0;
    DenseMap<int32_t, int64_t> channelOffsets;
    DenseMap<int32_t, int64_t> perVersionSizes;

    for (auto &[chId, info] : channels) {
      if (info.providers.empty())
        continue;
      auto dataTy = cast<MemRefType>(info.providers[0].getData().getType());
      int64_t sz = getMemRefByteSize(dataTy);
      if (sz <= 0)
        continue;

      int32_t versionCount = 1;
      auto parentScope = info.providers[0]->getParentOfType<CVScopeOp>();
      if (auto vc = parentScope->getAttrOfType<IntegerAttr>("version_count"))
        versionCount = vc.getInt();

      perVersionSizes[chId] = sz;
      channelOffsets[chId] = totalWsBytes;
      totalWsBytes += sz * versionCount;

      LLVM_DEBUG(llvm::dbgs()
                 << "[materialize] channel " << chId
                 << ": " << sz << " bytes/version × " << versionCount
                 << " versions, offset=" << channelOffsets[chId] << "\n");
    }

    // Materialize each channel.
    for (auto &[chId, info] : channels) {
      if (channelOffsets.find(chId) == channelOffsets.end())
        continue;

      int64_t pvSize = perVersionSizes[chId];
      auto dataTy = cast<MemRefType>(info.providers[0].getData().getType());
      auto wsTy = MemRefType::get(dataTy.getShape(), dataTy.getElementType());

      // -- Providers: insert DMA write to workspace --
      for (auto provOp : info.providers) {
        CVScopeOp parentScope = provOp->getParentOfType<CVScopeOp>();
        int32_t versionId = 0;
        if (auto vid = parentScope->getAttrOfType<IntegerAttr>("version_id"))
          versionId = vid.getInt();

        int64_t offset = channelOffsets[chId] + versionId * pvSize;

        // Create workspace view outside the scope.
        Value wsView;
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(parentScope);
          Location loc = provOp.getLoc();

          if (wsArg && blockI64) {
            Value blkIdx =
                b.create<arith::IndexCastOp>(loc, b.getIndexType(), blockI64);
            AffineExpr s0 = getAffineSymbolExpr(0, ctx);
            Value wsOffset = b.create<affine::AffineApplyOp>(
                loc,
                AffineMap::get(0, 1,
                               s0 * totalWsBytes + offset, ctx),
                ValueRange{blkIdx});
            wsView = b.create<memref::ViewOp>(
                loc, wsTy, wsArg, wsOffset, ValueRange{});
          } else {
            auto alloc = b.create<memref::AllocOp>(loc, wsTy);
            alloc->setAttr("alignment", b.getI64IntegerAttr(64));
            wsView = alloc;
          }
        }

        BlockArgument wsBlockArg = addScopeInput(parentScope, wsView);

        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(provOp);
          if (info.direction == WSDirection::C2V)
            emitFixpipe(b, provOp.getLoc(), provOp.getData(), wsBlockArg);
          else
            b.create<memref::CopyOp>(provOp.getLoc(), provOp.getData(),
                                     wsBlockArg);
        }

        provOp->erase();

        LLVM_DEBUG(llvm::dbgs()
                   << "[materialize] provider in scope_"
                   << getScopeId(parentScope)
                   << " v" << versionId << ": "
                   << stringifyWSDirection(info.direction)
                   << " DMA, offset=" << offset << "\n");
      }

      // -- Consumers: load from workspace into local buffer --
      for (auto consOp : info.consumers) {
        CVScopeOp parentScope = consOp->getParentOfType<CVScopeOp>();
        int32_t versionId = 0;
        if (auto vid = parentScope->getAttrOfType<IntegerAttr>("version_id"))
          versionId = vid.getInt();

        int64_t offset = channelOffsets[chId] + versionId * pvSize;

        // Create workspace view outside the scope.
        Value wsView;
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(parentScope);
          Location loc = consOp.getLoc();

          if (wsArg && blockI64) {
            Value blkIdx =
                b.create<arith::IndexCastOp>(loc, b.getIndexType(), blockI64);
            AffineExpr s0 = getAffineSymbolExpr(0, ctx);
            Value wsOffset = b.create<affine::AffineApplyOp>(
                loc,
                AffineMap::get(0, 1,
                               s0 * totalWsBytes + offset, ctx),
                ValueRange{blkIdx});
            wsView = b.create<memref::ViewOp>(
                loc, wsTy, wsArg, wsOffset, ValueRange{});
          } else {
            auto alloc = b.create<memref::AllocOp>(loc, wsTy);
            alloc->setAttr("alignment", b.getI64IntegerAttr(64));
            wsView = alloc;
          }
        }

        BlockArgument wsBlockArg = addScopeInput(parentScope, wsView);

        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(consOp);

          auto localAlloc = b.create<memref::AllocOp>(consOp.getLoc(), wsTy);
          localAlloc->setAttr("alignment", b.getI64IntegerAttr(64));
          if (info.direction == WSDirection::C2V)
            b.create<memref::CopyOp>(consOp.getLoc(), wsBlockArg,
                                     localAlloc.getResult());
          else
            emitLoad(b, consOp.getLoc(), wsBlockArg, localAlloc);

          consOp.getResult().replaceAllUsesWith(localAlloc.getResult());
        }

        consOp->erase();

        LLVM_DEBUG(llvm::dbgs()
                   << "[materialize] consumer in scope_"
                   << getScopeId(parentScope)
                   << " v" << versionId
                   << ": load from workspace, offset=" << offset << "\n");
      }
    }

    if (totalWsBytes > 0)
      func->setAttr("tilelangir.workspace_bytes",
                     b.getI64IntegerAttr(totalWsBytes));

    LLVM_DEBUG(llvm::dbgs()
               << "[materialize] total workspace: " << totalWsBytes
               << " bytes\n");
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
