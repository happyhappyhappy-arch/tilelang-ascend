// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// tilelangir-simple-multibuffer
//
// Loop distribution + unrolling for multi-buffering.
//
// Transforms:
//   for(i = 0; i < N; i += step) { scope1(i); scope2(i); scope3(i); }
// into:
//   for(i = 0; i < N; i += step * factor) {
//     scope1(i+0*step); scope1(i+1*step); ...; scope1(i+(F-1)*step);
//     scope2(i+0*step); scope2(i+1*step); ...; scope2(i+(F-1)*step);
//     scope3(i+0*step); scope3(i+1*step); ...; scope3(i+(F-1)*step);
//   }
//
// Shared allocs (used by scopes in different groups) are duplicated:
// one independent alloc per version, avoiding dynamic offsets on cbuf/ub.
// Each cloned scope carries a `version_id` attribute for inject-block-sync.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-simple-multibuffer"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRSIMPLEMULTIBUFFER
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

struct TileLangIRSimpleMultiBuffer
    : impl::TileLangIRSimpleMultiBufferBase<TileLangIRSimpleMultiBuffer> {
  using TileLangIRSimpleMultiBufferBase::TileLangIRSimpleMultiBufferBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    scf::ForOp forOp;
    for (auto &op : func.getBody().front()) {
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        forOp = f;
        break;
      }
    }
    if (!forOp)
      return;

    // Read num_stages from the scf.for attribute (set by frontend codegen);
    // falls back to the pass option tile-factor.
    if (auto attr =
            forOp->getAttrOfType<IntegerAttr>("tilelangir.num_stages")) {
      tileFactor = static_cast<unsigned>(attr.getInt());
      LLVM_DEBUG(llvm::dbgs()
                 << "[multibuffer] using tilelangir.num_stages = "
                 << tileFactor << "\n");
    }
    if (tileFactor <= 1)
      return;

    // Running accumulators (flagged by analyze-cross-scope) prevent
    // multi-buffering: the serial dependency chain would break.
    if (forOp->hasAttr("tilelangir.has_bidirectional_ws")) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[multibuffer] bidirectional workspace detected; "
                    "forcing tileFactor=1\n");
      tileFactor = 1;
      return;
    }

    SmallVector<CVScopeOp> scopes;
    for (auto &op : *forOp.getBody())
      if (auto sc = dyn_cast<CVScopeOp>(&op))
        scopes.push_back(sc);

    if (scopes.size() < 2)
      return;

    // ---------------------------------------------------------------
    // Partition loop body: [g0, scope0, g1, scope1, g2, scope2, ...]
    // ---------------------------------------------------------------
    SmallVector<SmallVector<Operation *>> groups(scopes.size());
    SmallVector<Operation *> allocOps;
    {
      int gi = 0;
      for (auto &op : forOp.getBody()->without_terminator()) {
        if (isa<CVScopeOp>(&op)) {
          ++gi;
          continue;
        }
        if (isa<memref::AllocOp>(&op)) {
          allocOps.push_back(&op);
          continue;
        }
        int idx = std::min(gi, static_cast<int>(scopes.size()) - 1);
        groups[idx].push_back(&op);
      }
    }

    // All loop-body allocs need versioning — one copy per unrolled
    // iteration — to avoid HIVM NZ format conflicts when the same
    // buffer appears as nd2nz destination in multiple unrolled copies.
    LLVM_DEBUG(llvm::dbgs() << "[multibuffer] " << allocOps.size()
                            << " alloc(s) will be versioned\n");

    // ---------------------------------------------------------------
    // Build new outer loop with larger step
    // ---------------------------------------------------------------
    OpBuilder b(forOp);
    Location loc = forOp.getLoc();
    Value origLb = forOp.getLowerBound();
    Value origUb = forOp.getUpperBound();
    Value origStep = forOp.getStep();
    bool isI32 = origStep.getType().isInteger(32);

    Value factorVal;
    if (isI32)
      factorVal = b.create<arith::ConstantIntOp>(loc, tileFactor, 32);
    else
      factorVal = b.create<arith::ConstantIndexOp>(loc, tileFactor);

    Value newStep = b.create<arith::MulIOp>(loc, origStep, factorVal);
    auto outerFor = b.create<scf::ForOp>(loc, origLb, origUb, newStep);
    Value outerIV = outerFor.getInductionVar();

    b.setInsertionPointToStart(outerFor.getBody());

    // ---------------------------------------------------------------
    // Create N separate allocs per buffer per version.
    // versionAllocs[origAlloc][j] = the j-th version alloc.
    // ---------------------------------------------------------------
    IRMapping allocMapping; // empty — no shared hoisted allocs
    DenseMap<Value, SmallVector<Value>> versionAllocs;
    for (auto *op : allocOps) {
      Value allocVal = op->getResult(0);
      SmallVector<Value> versions;
      for (unsigned j = 0; j < tileFactor; ++j) {
        auto *cloned = b.clone(*op);
        versions.push_back(cloned->getResult(0));
      }
      versionAllocs[allocVal] = std::move(versions);

      LLVM_DEBUG(llvm::dbgs()
                 << "[multibuffer] created " << tileFactor
                 << " versions for alloc of type "
                 << allocVal.getType() << "\n");
    }

    // ---------------------------------------------------------------
    // Unrolled inner loops: for each scope group, emit N copies.
    // ---------------------------------------------------------------
    for (size_t i = 0; i < scopes.size(); ++i) {
      for (unsigned j = 0; j < tileFactor; ++j) {
        Value jOffset;
        if (isI32)
          jOffset = b.create<arith::ConstantIntOp>(loc, j, 32);
        else
          jOffset = b.create<arith::ConstantIndexOp>(loc, j);

        Value effectiveIV =
            b.create<arith::AddIOp>(loc, outerIV, jOffset);

        IRMapping mapping(allocMapping);
        mapping.map(forOp.getInductionVar(), effectiveIV);

        // Map all allocs to version j
        for (auto &[origAlloc, versions] : versionAllocs)
          mapping.map(origAlloc, versions[j]);

        // Clone groups [0..i] for transitive dependencies
        for (size_t g = 0; g <= i; ++g)
          for (auto *op : groups[g])
            b.clone(*op, mapping);

        // Clone scope and annotate with version_id
        auto *clonedScope = b.clone(*scopes[i], mapping);
        if (auto scopeOp = dyn_cast<CVScopeOp>(clonedScope)) {
          scopeOp->setAttr("version_id",
                           b.getI32IntegerAttr(j));
          scopeOp->setAttr("version_count",
                           b.getI32IntegerAttr(tileFactor));

          // Update ws_provider / ws_consumer flags for this version.
          // flag = channelId * tileFactor + j
          SmallVector<WSProviderOp> provOps;
          SmallVector<WSConsumerOp> consOps;
          for (Operation &op : scopeOp.getBodyBlock()) {
            if (auto p = dyn_cast<WSProviderOp>(&op))
              provOps.push_back(p);
            else if (auto c = dyn_cast<WSConsumerOp>(&op))
              consOps.push_back(c);
          }
          for (auto prov : provOps) {
            int32_t newFlag =
                prov.getChannelId() * static_cast<int32_t>(tileFactor) + j;
            OpBuilder fb(prov->getContext());
            fb.setInsertionPoint(prov);
            Value nf = fb.create<arith::ConstantIndexOp>(loc, newFlag);
            prov->setOperand(1, nf); // flag is operand 1
          }
          for (auto cons : consOps) {
            int32_t newFlag =
                cons.getChannelId() * static_cast<int32_t>(tileFactor) + j;
            OpBuilder fb(cons->getContext());
            fb.setInsertionPoint(cons);
            Value nf = fb.create<arith::ConstantIndexOp>(loc, newFlag);
            cons->setOperand(0, nf); // flag is operand 0
          }
        }
      }
    }

    forOp.erase();

    versionGMWorkspace(func, outerFor, b, loc);

    LLVM_DEBUG(llvm::dbgs()
               << "[multibuffer] unrolled " << scopes.size()
               << " scopes × " << tileFactor
               << " versions, " << versionAllocs.size()
               << " alloc(s) versioned\n");
  }

private:
  void versionGMWorkspace(func::FuncOp func, scf::ForOp outerFor,
                          OpBuilder &b, Location loc) {
    Value blockIdxI32;
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "hivm.hir.get_block_idx") {
        for (auto user : op->getResult(0).getUsers()) {
          if (auto trunc = dyn_cast<arith::TruncIOp>(user)) {
            blockIdxI32 = trunc.getResult();
          }
        }
      }
    });
    if (!blockIdxI32)
      return;

    // Find 3D GM reinterpret_casts of dynamic function args that have
    // memref.copy-to-GM users inside the loop (V→C workspace pattern).
    SmallVector<memref::ReinterpretCastOp> gmRCOps;
    for (auto &op : func.getBody().front()) {
      auto rc = dyn_cast<memref::ReinterpretCastOp>(&op);
      if (!rc)
        continue;
      auto resTy = rc.getResult().getType();
      if (resTy.getRank() != 3 || !isa<BlockArgument>(rc.getSource()))
        continue;
      auto srcTy = cast<MemRefType>(rc.getSource().getType());
      if (srcTy.getRank() != 1 || !srcTy.isDynamicDim(0))
        continue;

      bool hasCopyToGM = false;
      for (auto *user : rc.getResult().getUsers()) {
        auto sv = dyn_cast<memref::SubViewOp>(user);
        if (!sv || !outerFor->isAncestor(sv))
          continue;
        for (auto *svUser : sv.getResult().getUsers()) {
          if (auto copy = dyn_cast<memref::CopyOp>(svUser))
            if (copy.getTarget() == sv.getResult())
              hasCopyToGM = true;
        }
      }
      if (hasCopyToGM)
        gmRCOps.push_back(rc);
    }

    if (gmRCOps.empty())
      return;

    // Build version map: walk outer loop body in reverse; each op
    // inherits the version_id of the next scope encountered.
    DenseMap<Operation *, int> opVersion;
    {
      int cur = -1;
      for (auto &op :
           llvm::reverse(outerFor.getBody()->without_terminator())) {
        if (auto sc = dyn_cast<CVScopeOp>(&op))
          if (auto vid = sc->getAttrOfType<IntegerAttr>("version_id"))
            cur = vid.getInt();
        if (cur >= 0)
          opVersion[&op] = cur;
      }
    }

    for (auto rc : gmRCOps) {
      auto resTy = rc.getResult().getType();
      auto shape = resTy.getShape();
      int64_t newDim0 = shape[0] * static_cast<int64_t>(tileFactor);

      auto newTy = MemRefType::get(
          {newDim0, shape[1], shape[2]}, resTy.getElementType(),
          resTy.getLayout(), resTy.getMemorySpace());

      // Rebuild the reinterpret_cast with the enlarged first dimension.
      OpBuilder rcb(rc);
      SmallVector<int64_t> newStaticSizes(rc.getStaticSizes());
      newStaticSizes[0] = newDim0;

      auto newRC = rcb.create<memref::ReinterpretCastOp>(
          rc.getLoc(), newTy, rc.getSource(), rc.getOffsets(),
          rc.getSizes(), rc.getStrides(), rc.getStaticOffsets(),
          newStaticSizes, SmallVector<int64_t>(rc.getStaticStrides()));

      rc.getResult().replaceAllUsesWith(newRC.getResult());
      rc.erase();

      // Update scope block-arg types that receive this memref.
      for (auto *user : newRC.getResult().getUsers()) {
        auto scope = dyn_cast<CVScopeOp>(user);
        if (!scope)
          continue;
        for (unsigned k = 0; k < scope.getInputs().size(); ++k) {
          if (scope.getInputs()[k] == newRC.getResult())
            scope.getBodyBlock().getArgument(k).setType(newTy);
        }
      }

      // Find index_cast(blockIdxI32) ops inside the outer loop that
      // feed the first offset of a subview of this RC, or are passed
      // alongside this RC to a scope.
      SmallVector<arith::IndexCastOp> toVersion;
      for (auto &op : outerFor.getBody()->without_terminator()) {
        auto ic = dyn_cast<arith::IndexCastOp>(&op);
        if (!ic || ic.getIn() != blockIdxI32)
          continue;
        bool feeds = false;
        for (auto *user : ic.getResult().getUsers()) {
          if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
            if (sv.getSource() == newRC.getResult()) {
              feeds = true;
              break;
            }
          }
          if (auto scope = dyn_cast<CVScopeOp>(user)) {
            for (auto operand : scope.getInputs()) {
              if (operand == newRC.getResult()) {
                feeds = true;
                break;
              }
            }
          }
          if (feeds)
            break;
        }
        if (feeds)
          toVersion.push_back(ic);
      }

      for (auto ic : toVersion) {
        auto it = opVersion.find(ic.getOperation());
        if (it == opVersion.end())
          continue;
        int vId = it->second;

        OpBuilder fb(ic);
        Value factor = fb.create<arith::ConstantIntOp>(loc, tileFactor, 32);
        Value scaled =
            fb.create<arith::MulIOp>(loc, blockIdxI32, factor);
        Value voff = fb.create<arith::ConstantIntOp>(loc, vId, 32);
        Value newBIdx = fb.create<arith::AddIOp>(loc, scaled, voff);
        Value newIdx = fb.create<arith::IndexCastOp>(
            loc, fb.getIndexType(), newBIdx);

        ic.getResult().replaceAllUsesWith(newIdx);
        ic.erase();
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[multibuffer] versioned GM workspace "
                 << shape[0] << "→" << newDim0 << " in dim 0, "
                 << toVersion.size() << " offset(s) adjusted\n");
    }
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
