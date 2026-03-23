// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// Stage 5 – tilelangir-outline-scope (spike / 穿刺)
//
// Split a MIX-typed entry function into two clones:
//   @{name}_mix_aic  (hivm.func_core_type = AIC)
//   @{name}_mix_aiv  (hivm.func_core_type = AIV)
//
// Each clone keeps only its core-type cv_scope ops and flattens them
// (inlines the scope body into the parent block). Opposite-type scopes
// and sync_block ops are erased.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-outline-scope"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIROUTLINESCOPE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

static bool isCubeScope(CVScopeOp sc) {
  auto a = sc->getAttrOfType<IntegerAttr>("tcore_type");
  return a &&
         a.getInt() == static_cast<int64_t>(hivm::TCoreType::CUBE);
}

static bool isVectorScope(CVScopeOp sc) {
  auto a = sc->getAttrOfType<IntegerAttr>("tcore_type");
  return a &&
         a.getInt() == static_cast<int64_t>(hivm::TCoreType::VECTOR);
}

static bool isSyncBlockOp(Operation *op) {
  return isa<hivm::SyncBlockSetOp>(op) ||
         isa<hivm::SyncBlockWaitOp>(op);
}

static bool isCubeSyncBlockOp(Operation *op) {
  if (auto setOp = dyn_cast<hivm::SyncBlockSetOp>(op))
    return setOp.getTcoreType().getTcoretype() == hivm::TCoreType::CUBE;
  if (auto waitOp = dyn_cast<hivm::SyncBlockWaitOp>(op))
    return waitOp.getTcoreType().getTcoretype() == hivm::TCoreType::CUBE;
  return false;
}

/// Inline the body of a cv_scope into its parent block: remap block args
/// to the scope's input operands, move ops before the scope, then erase it.
static void flattenScope(CVScopeOp scope) {
  Block &body = scope.getBody().front();

  for (auto [arg, input] :
       llvm::zip(body.getArguments(), scope.getInputs()))
    arg.replaceAllUsesWith(input);

  Operation *term = body.getTerminator();
  while (&body.front() != term)
    body.front().moveBefore(scope);

  scope->erase();
}

/// Remove cv_scope ops that don't match the kept core type, flatten the
/// remaining ones, and erase opposite-type sync_block ops.
static void processClone(func::FuncOp fn, bool keepCube) {
  // 1. Erase opposite-type scopes (reverse walk for safe erasure).
  SmallVector<CVScopeOp> toErase;
  fn.walk([&](CVScopeOp sc) {
    bool cube = isCubeScope(sc);
    bool vector = isVectorScope(sc);
    if ((keepCube && vector) || (!keepCube && cube))
      toErase.push_back(sc);
  });
  for (auto sc : llvm::reverse(toErase))
    sc->erase();

  // 2. Flatten remaining scopes (bottom-up for safety with nesting).
  SmallVector<CVScopeOp> toFlatten;
  fn.walk([&](CVScopeOp sc) { toFlatten.push_back(sc); });
  for (auto sc : llvm::reverse(toFlatten))
    flattenScope(sc);

  // 3. Erase opposite-type sync_block ops.
  SmallVector<Operation *> syncToErase;
  fn.walk([&](Operation *op) {
    if (!isSyncBlockOp(op))
      return;
    bool cubeSyncOp = isCubeSyncBlockOp(op);
    if (cubeSyncOp != keepCube)
      syncToErase.push_back(op);
  });
  for (auto *op : llvm::reverse(syncToErase))
    op->erase();
}

/// Erase memref.alloc whose result has no uses (e.g. D_BUF duplicated into AIV).
static void eraseUnusedAllocs(func::FuncOp fn) {
  SmallVector<memref::AllocOp> dead;
  fn.walk([&](memref::AllocOp a) {
    if (a.getResult().use_empty())
      dead.push_back(a);
  });
  for (auto a : llvm::reverse(dead))
    a.erase();
}

/// Erase dead memref.subview / reinterpret_cast (no uses).
static void eraseUnusedViewLikeOps(func::FuncOp fn) {
  SmallVector<Operation *> dead;
  fn.walk([&](Operation *op) {
    if (isa<memref::SubViewOp, memref::ReinterpretCastOp>(op) &&
        op->use_empty())
      dead.push_back(op);
  });
  for (auto *op : llvm::reverse(dead))
    op->erase();
}

/// Remove value-producing ops whose results are unused inside for bodies, then
/// erase for loops that only contain a terminator (e.g. VECTOR sync_wait
/// stripped from AIC clone).
///
/// `Operation::use_empty()` is vacuously true for ops with zero results
/// (e.g. hivm.hir.nd2nz, sync_block_set). Only erase ops that define SSA
/// values and all of those values are unused.
static void eraseDegenerateForLoops(func::FuncOp fn) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<scf::ForOp> fors;
    fn.walk([&](scf::ForOp fo) { fors.push_back(fo); });
    for (auto fo : fors) {
      Block &bb = *fo.getBody();
      // Erase value-producing ops with unused results.
      for (Operation &op : llvm::make_early_inc_range(bb.without_terminator())) {
        if (op.getNumResults() == 0)
          continue;
        if (!llvm::all_of(op.getResults(),
                          [](Value v) { return v.use_empty(); }))
          continue;
        op.erase();
        changed = true;
      }
      if (bb.without_terminator().empty()) {
        fo.erase();
        changed = true;
        continue;
      }
      // A loop body with only side-effect-free 0-result ops (e.g.
      // pipe_barrier left over after stripping opposite-core ops) is
      // degenerate — erase everything and remove the loop.
      bool allTrivial = true;
      for (Operation &op : bb.without_terminator()) {
        if (op.getName().getStringRef() != "hivm.hir.pipe_barrier") {
          allTrivial = false;
          break;
        }
      }
      if (allTrivial) {
        for (Operation &op :
             llvm::make_early_inc_range(bb.without_terminator()))
          op.erase();
        fo.erase();
        changed = true;
      }
    }
  }
}

/// After MIX split, a Cube cv_scope may contain hivm.hir.nd2nz that only
/// populated a buffer consumed by Vector-only scopes (erased from the AIC
/// clone). The orphaned nd2nz + alloc stays in the AIC loop and forces a 2D
/// L1 path while mmadL1 operands use storage-aligned 4D casts, which breaks
/// HIVM lowering (func.call operand rank mismatch).  Remove nd2nz when its
/// destination memref.alloc is only used as this op's dst operand (no other
/// readers/writers).
static void eraseDeadNd2NzWithNoBufferConsumers(func::FuncOp fn) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;
    fn.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "hivm.hir.nd2nz")
        return;
      if (op->getNumOperands() < 2)
        return;
      Value dst = op->getOperand(1);
      Operation *def = dst.getDefiningOp();
      if (!def || !isa<memref::AllocOp>(def))
        return;
      if (!dst.hasOneUse() || *dst.getUsers().begin() != op)
        return;
      toErase.push_back(op);
    });
    for (auto *op : toErase) {
      op->erase();
      changed = true;
    }
    if (changed) {
      eraseUnusedAllocs(fn);
      eraseUnusedViewLikeOps(fn);
      eraseDegenerateForLoops(fn);
    }
  }
}

/// Check if an alloc result is used as the output (operand 6) of mmadL1.
static bool isUsedAsMmadOutput(memref::AllocOp alloc) {
  for (auto *user : alloc.getResult().getUsers())
    if (user->getName().getStringRef() == "hivm.hir.mmadL1" &&
        user->getNumOperands() > 6 && user->getOperand(6) == alloc.getResult())
      return true;
  return false;
}

/// Remap HIVM address spaces on memref.alloc ops in a cloned function.
///
/// AIC: zero → cc (explicit L0C); bare allocs → cc if mmadL1 output,
///      else cbuf (L1).  Keep existing cbuf unchanged.
///
/// AIV: everything non-GM, non-UB → ub (Vector uses Unified Buffer).
static void remapAllocAddressSpaces(func::FuncOp fn, bool isAIC) {
  auto *ctx = fn.getContext();
  auto ubAttr   = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
  auto ccAttr   = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::L0C);
  auto cbufAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::L1);

  SmallVector<std::pair<memref::AllocOp, Attribute>> toReplace;
  fn.walk([&](memref::AllocOp alloc) {
    auto ms = alloc.getType().getMemorySpace();

    if (isAIC) {
      if (!ms) {
        Attribute target = isUsedAsMmadOutput(alloc)
                               ? static_cast<Attribute>(ccAttr)
                               : static_cast<Attribute>(cbufAttr);
        toReplace.emplace_back(alloc, target);
        return;
      }
      auto hivmAS = dyn_cast<hivm::AddressSpaceAttr>(ms);
      if (!hivmAS) return;
      if (hivmAS.getAddressSpace() == hivm::AddressSpace::Zero)
        toReplace.emplace_back(alloc, ccAttr);
      else if (hivmAS.getAddressSpace() == hivm::AddressSpace::L1 &&
               isUsedAsMmadOutput(alloc))
        toReplace.emplace_back(alloc, ccAttr);
    } else {
      if (!ms) {
        toReplace.emplace_back(alloc, ubAttr);
        return;
      }
      auto hivmAS = dyn_cast<hivm::AddressSpaceAttr>(ms);
      if (!hivmAS) return;
      if (hivmAS.getAddressSpace() != hivm::AddressSpace::GM &&
          hivmAS.getAddressSpace() != hivm::AddressSpace::UB)
        toReplace.emplace_back(alloc, ubAttr);
    }
  });

  OpBuilder b(ctx);
  for (auto &[alloc, newSpace] : toReplace) {
    auto oldType = alloc.getType();
    auto newType = MemRefType::get(oldType.getShape(),
                                   oldType.getElementType(),
                                   oldType.getLayout(),
                                   newSpace);
    b.setInsertionPoint(alloc);
    auto newAlloc = b.create<memref::AllocOp>(
        alloc.getLoc(), newType, alloc.getDynamicSizes(),
        alloc.getAlignmentAttr());
    alloc.getResult().replaceAllUsesWith(newAlloc.getResult());
    alloc->erase();
  }
}

/// Promote workspace arg (second memref<?xi8>) to GM and rebuild all views.
/// memref.view requires source and result to share the same address space,
/// so we change the arg type AND recreate view ops with GM result types.
/// Run AFTER outline-scope splits to avoid MIX-verifier issues.
static void promoteWorkspaceArgToGM(func::FuncOp fn) {
  auto *ctx = fn.getContext();
  auto gmAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM);

  int wsArgIdx = -1;
  int dynIdx = 0;
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    auto mt = dyn_cast<MemRefType>(fn.getArgument(i).getType());
    if (!mt) continue;
    if (mt.getElementType().isInteger(8) && mt.getRank() == 1 &&
        mt.isDynamicDim(0)) {
      if (dynIdx == 1) { wsArgIdx = i; break; }
      ++dynIdx;
    }
  }
  if (wsArgIdx < 0) return;

  BlockArgument wsArg = fn.getArgument(wsArgIdx);
  auto oldArgTy = cast<MemRefType>(wsArg.getType());
  auto newArgTy = MemRefType::get(oldArgTy.getShape(),
                                  oldArgTy.getElementType(),
                                  oldArgTy.getLayout(), gmAttr);
  wsArg.setType(newArgTy);

  SmallVector<Type> argTypes;
  for (auto a : fn.getArguments())
    argTypes.push_back(a.getType());
  fn.setType(FunctionType::get(ctx, argTypes,
                               fn.getFunctionType().getResults()));

  SmallVector<memref::ViewOp> views;
  fn.walk([&](memref::ViewOp view) {
    if (view.getSource() == wsArg)
      views.push_back(view);
  });

  OpBuilder b(ctx);
  for (auto view : views) {
    auto oldTy = view.getType();
    if (oldTy.getMemorySpace()) continue;
    auto newTy = MemRefType::get(oldTy.getShape(), oldTy.getElementType(),
                                 oldTy.getLayout(), gmAttr);
    b.setInsertionPoint(view);
    auto nv = b.create<memref::ViewOp>(view.getLoc(), newTy,
                                       view.getSource(),
                                       view.getByteShift(),
                                       view.getSizes());
    view.getResult().replaceAllUsesWith(nv.getResult());
    view->erase();
  }
}

/// After remapAllocAddressSpaces on AIV, memref.alloc sources may be UB while
/// reinterpret_cast result types still carry cbuf — verifier requires matching
/// address spaces on reinterpret.
static void fixReinterpretCastResultMemSpace(func::FuncOp fn) {
  SmallVector<memref::ReinterpretCastOp> rcs;
  fn.walk([&](memref::ReinterpretCastOp op) { rcs.push_back(op); });
  for (auto op : rcs) {
    auto srcTy = cast<MemRefType>(op.getSource().getType());
    auto resTy = cast<MemRefType>(op.getType());
    if (srcTy.getMemorySpace() == resTy.getMemorySpace())
      continue;
    auto newResTy = MemRefType::get(resTy.getShape(), resTy.getElementType(),
                                    resTy.getLayout(), srcTy.getMemorySpace());
    op.getResult().setType(newResTy);
  }
}

/// Same fix for memref.subview: after address space remapping the source
/// type may differ from the result type's memory space.
static void fixSubViewResultMemSpace(func::FuncOp fn) {
  SmallVector<memref::SubViewOp> svs;
  fn.walk([&](memref::SubViewOp op) { svs.push_back(op); });
  for (auto op : svs) {
    auto srcTy = cast<MemRefType>(op.getSource().getType());
    auto resTy = cast<MemRefType>(op.getType());
    if (srcTy.getMemorySpace() == resTy.getMemorySpace())
      continue;
    auto newResTy = MemRefType::get(resTy.getShape(), resTy.getElementType(),
                                    resTy.getLayout(), srcTy.getMemorySpace());
    op.getResult().setType(newResTy);
  }
}

/// Trace through view-like ops to find the root memref (alloc or func arg).
static Value traceToRootMemRef(Value v) {
  while (auto *defOp = v.getDefiningOp()) {
    if (isa<memref::SubViewOp, memref::ReinterpretCastOp,
            memref::ViewOp, memref::CastOp>(defOp)) {
      v = defOp->getOperand(0);
    } else {
      break;
    }
  }
  return v;
}

/// Check whether any HIVM op uses this memref (or any view derived from it).
/// Used as a proxy for "this buffer is live / was written to".
static bool hasHIVMUsers(Value root) {
  SmallVector<Value> worklist{root};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    for (auto *user : v.getUsers()) {
      if (isa<memref::SubViewOp, memref::ReinterpretCastOp,
              memref::ViewOp, memref::CastOp>(user)) {
        for (auto res : user->getResults())
          worklist.push_back(res);
        continue;
      }
      if (isa<memref::CopyOp>(user))
        continue;
      if (user->getName().getStringRef().starts_with("hivm.hir."))
        return true;
    }
  }
  return false;
}

/// In AIC clone, handle memref.copy that convert-to-hivm-op cannot lower.
/// - Dead copies (source buffer never written in AIC) → erase.
/// - Live cbuf/cc → GM copies → convert to hivm.hir.fixpipe.
static void handleMemCopiesInAIC(func::FuncOp fn) {
  SmallVector<memref::CopyOp> copies;
  fn.walk([&](memref::CopyOp op) { copies.push_back(op); });

  OpBuilder b(fn.getContext());
  for (auto copy : copies) {
    auto dstTy = cast<MemRefType>(copy.getTarget().getType());
    auto srcTy = cast<MemRefType>(copy.getSource().getType());

    auto dstAS =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(dstTy.getMemorySpace());
    auto srcAS =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(srcTy.getMemorySpace());

    // Only handle non-GM source → GM dest.
    if (!dstAS || dstAS.getAddressSpace() != hivm::AddressSpace::GM)
      continue;
    if (srcAS && srcAS.getAddressSpace() == hivm::AddressSpace::GM)
      continue;

    Value root = traceToRootMemRef(copy.getSource());
    if (!hasHIVMUsers(root)) {
      copy.erase();
      continue;
    }

    b.setInsertionPoint(copy);
    OperationState s(copy.getLoc(), "hivm.hir.fixpipe");
    s.addOperands({copy.getSource(), copy.getTarget()});
    s.addAttribute("enable_nz2nd", b.getUnitAttr());
    b.create(s);
    copy.erase();
  }
}

/// In AIV clone, erase memref.copy whose source buffer was only written by
/// CUBE ops (now erased).  Prevents uninitialized UB data from being stored
/// back to GM and overwriting correct AIC output.
static void handleMemCopiesInAIV(func::FuncOp fn) {
  SmallVector<memref::CopyOp> toErase;
  fn.walk([&](memref::CopyOp copy) {
    auto dstTy = cast<MemRefType>(copy.getTarget().getType());
    auto srcTy = cast<MemRefType>(copy.getSource().getType());
    auto dstAS =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(dstTy.getMemorySpace());
    auto srcAS =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(srcTy.getMemorySpace());
    if (!dstAS || dstAS.getAddressSpace() != hivm::AddressSpace::GM)
      return;
    if (srcAS && srcAS.getAddressSpace() == hivm::AddressSpace::GM)
      return;
    Value root = traceToRootMemRef(copy.getSource());
    if (!hasHIVMUsers(root))
      toErase.push_back(copy);
  });
  for (auto copy : toErase)
    copy.erase();
}

/// In AIC functions, convert hivm.hir.load (Vector-only: GM↔UB) to
/// hivm.hir.nd2nz (Cube DMA: GM→L1).  The exp-mode reference uses nd2nz
/// for all Cube-side GM loads.
static void convertLoadsToNd2nz(func::FuncOp fn) {
  SmallVector<Operation *> loads;
  fn.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "hivm.hir.load")
      loads.push_back(op);
  });

  OpBuilder b(fn.getContext());
  for (auto *load : loads) {
    Value src = load->getOperand(0);
    Value dst = load->getOperand(1);
    b.setInsertionPoint(load);
    OperationState s(load->getLoc(), "hivm.hir.nd2nz");
    s.addOperands({src, dst});
    s.addAttribute("dst_continuous", b.getUnitAttr());
    s.addAttribute("init_out_buffer", b.getBoolAttr(false));
    s.addAttribute("operandSegmentSizes",
                   DenseI32ArrayAttr::get(b.getContext(), {1, 1, 0, 0}));
    b.create(s);
    load->erase();
  }
}

struct TileLangIROutlineScope
    : impl::TileLangIROutlineScopeBase<TileLangIROutlineScope> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> mixFuncs;
    module.walk([&](func::FuncOp f) {
      auto attr =
          f->getAttrOfType<hivm::TFuncCoreTypeAttr>("hivm.func_core_type");
      if (attr && attr.getFuncCoreType() == hivm::TFuncCoreType::MIX)
        mixFuncs.push_back(f);
    });

    for (auto fn : mixFuncs)
      splitFunction(fn, module);
  }

  void splitFunction(func::FuncOp fn, ModuleOp module) {
    MLIRContext *ctx = module.getContext();

    // AIC clone – zero→cc, bare→cbuf/cc, load→nd2nz
    IRMapping aicMap;
    auto aicFn = fn.clone(aicMap);
    aicFn.setName((fn.getName() + "_mix_aic").str());
    aicFn->setAttr("hivm.func_core_type",
                    hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIC));
    aicFn->setAttr("hivm.part_of_mix", UnitAttr::get(ctx));
    module.push_back(aicFn);
    processClone(aicFn, /*keepCube=*/true);
    eraseDegenerateForLoops(aicFn);
    eraseDeadNd2NzWithNoBufferConsumers(aicFn);
    remapAllocAddressSpaces(aicFn, /*isAIC=*/true);
    fixSubViewResultMemSpace(aicFn);
    convertLoadsToNd2nz(aicFn);
    handleMemCopiesInAIC(aicFn);
    eraseUnusedAllocs(aicFn);
    eraseUnusedViewLikeOps(aicFn);
    promoteWorkspaceArgToGM(aicFn);
    LLVM_DEBUG(llvm::dbgs() << "[outline] created AIC: "
                            << aicFn.getName() << "\n");

    // AIV clone – cbuf/zero→ub (Vector uses Unified Buffer)
    IRMapping aivMap;
    auto aivFn = fn.clone(aivMap);
    aivFn.setName((fn.getName() + "_mix_aiv").str());
    aivFn->setAttr("hivm.func_core_type",
                    hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
    aivFn->setAttr("hivm.part_of_mix", UnitAttr::get(ctx));
    // Both AIC and AIV need hacc.entry for bishengir-compile to generate
    // proper kernel entry code.  Only remove hivm.entry if present.
    module.push_back(aivFn);
    processClone(aivFn, /*keepCube=*/false);
    eraseDegenerateForLoops(aivFn);
    remapAllocAddressSpaces(aivFn, /*isAIC=*/false);
    handleMemCopiesInAIV(aivFn);
    fixSubViewResultMemSpace(aivFn);
    fixReinterpretCastResultMemSpace(aivFn);
    eraseUnusedAllocs(aivFn);
    eraseUnusedViewLikeOps(aivFn);
    promoteWorkspaceArgToGM(aivFn);

    LLVM_DEBUG(llvm::dbgs() << "[outline] created AIV: "
                            << aivFn.getName() << "\n");

    // Erase original MIX function
    fn.erase();
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
