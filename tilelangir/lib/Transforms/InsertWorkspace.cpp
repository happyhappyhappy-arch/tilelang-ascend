// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "tilelangir-insert-workspace"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINSERTWORKSPACE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

// Per-block workspace layout (bytes).
// scores:      64 * 64 * 4 = 16384
// scores_cast: 64 * 64 * 2 = 8192
// acc_o:       64 * 128* 4 = 32768
// Total:                     57344
static constexpr int64_t PER_BLOCK_WS = 57344;
static constexpr int64_t OFF_SCORES = 0;
static constexpr int64_t OFF_SCORES_CAST = 16384;
static constexpr int64_t OFF_ACC_O = 24576;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int32_t allocateNextScopeId(func::FuncOp fn) {
  int32_t maxId = -1;
  fn.walk([&](CVScopeOp sc) {
    if (auto a = sc->getAttrOfType<IntegerAttr>("scope_id"))
      maxId = std::max(maxId, static_cast<int32_t>(a.getInt()));
  });
  return maxId + 1;
}

/// Append a new input Value to a CVScopeOp and return the new block arg.
static BlockArgument addScopeInput(CVScopeOp scope, Value val) {
  unsigned n = scope.getInputs().size();
  scope->insertOperands(n, {val});
  return scope.getBody().front().addArgument(val.getType(), scope.getLoc());
}

/// Return the operand index for \p val in \p scope's inputs, or -1.
static int findInputIdx(CVScopeOp scope, Value val) {
  for (auto [i, v] : llvm::enumerate(scope.getInputs()))
    if (v == val)
      return i;
  return -1;
}

/// Find the first operation with name \p name in \p block.
static Operation *findOp(Block &block, StringRef name) {
  for (auto &op : block)
    if (op.getName().getStringRef() == name)
      return &op;
  return nullptr;
}

/// Create `hivm.hir.load ins(%src) outs(%dst) init_out_buffer = false …`
static void emitLoad(OpBuilder &b, Location loc, Value src, Value dst) {
  OperationState s(loc, "hivm.hir.load");
  s.addOperands({src, dst});
  s.addAttribute("init_out_buffer", b.getBoolAttr(false));
  s.addAttribute("may_implicit_transpose_with_last_axis", b.getBoolAttr(false));
  s.addAttribute("operandSegmentSizes",
                  DenseI32ArrayAttr::get(b.getContext(), {1, 1, 0, 0, 0, 0}));
  b.create(s);
}

/// Create `hivm.hir.store ins(%src) outs(%dst)`
static void emitStore(OpBuilder &b, Location loc, Value src, Value dst) {
  OperationState s(loc, "hivm.hir.store");
  s.addOperands({src, dst});
  b.create(s);
}

/// Create `hivm.hir.fixpipe {enable_nz2nd} ins(%src) outs(%dst)`
static void emitFixpipe(OpBuilder &b, Location loc, Value src, Value dst) {
  OperationState s(loc, "hivm.hir.fixpipe");
  s.addOperands({src, dst});
  s.addAttribute("enable_nz2nd", b.getUnitAttr());
  b.create(s);
}

/// Create `hivm.hir.vadd ins(%a, %b) outs(%c)`
static void emitVadd(OpBuilder &b, Location loc, Value a, Value c1,
                     Value out) {
  OperationState s(loc, "hivm.hir.vadd");
  s.addOperands({a, c1, out});
  s.addAttribute("operandSegmentSizes",
                  DenseI32ArrayAttr::get(b.getContext(), {2, 1, 0}));
  b.create(s);
}

/// Allocate a local memref with `alignment = 64`, stripping any address space.
static Value allocAligned(OpBuilder &b, Location loc, MemRefType ty) {
  auto localTy = MemRefType::get(ty.getShape(), ty.getElementType());
  auto alloc = b.create<memref::AllocOp>(loc, localTy);
  alloc->setAttr("alignment", b.getI64IntegerAttr(64));
  return alloc;
}

/// Find an `arith.constant true` (or false) in \p func.
static Value findBoolConst(func::FuncOp func, bool val) {
  Value result;
  func.walk([&](arith::ConstantOp op) {
    if (auto a = op.getValue().dyn_cast<BoolAttr>())
      if (a.getValue() == val)
        result = op.getResult();
  });
  return result;
}

/// Return the first shared memref input between \p s1 and \p s2
/// whose shape equals \p shape and element type is f32 (isF32) or f16.
static Value findSharedMemref(CVScopeOp s1, CVScopeOp s2,
                              ArrayRef<int64_t> shape, bool isF32) {
  for (Value v1 : s1.getInputs()) {
    auto mt = v1.getType().dyn_cast<MemRefType>();
    if (!mt || mt.getShape() != shape)
      continue;
    if (isF32 ? !mt.getElementType().isF32() : !mt.getElementType().isF16())
      continue;
    for (Value v2 : s2.getInputs())
      if (v1 == v2)
        return v1;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct TileLangIRInsertWorkspace
    : impl::TileLangIRInsertWorkspaceBase<TileLangIRInsertWorkspace> {

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

  // -----------------------------------------------------------------------
  void processFunc(func::FuncOp func, ModuleOp module) {
    MLIRContext *ctx = func.getContext();
    OpBuilder b(ctx);
    Location loc = func.getLoc();

    // 1. Workspace arg — second memref<?xi8> (WorkspaceArgIdx = 1).
    Value wsArg;
    int dynIdx = 0;
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      if (auto mt = func.getArgument(i).getType().dyn_cast<MemRefType>()) {
        if (mt.getElementType().isInteger(8) && mt.getRank() == 1 &&
            mt.isDynamicDim(0)) {
          if (dynIdx == 1) {
            wsArg = func.getArgument(i);
            break;
          }
          ++dynIdx;
        }
      }
    }
    if (!wsArg)
      return;

    // 2. block_idx (get_block_idx → i64).
    Value blockI64;
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "hivm.hir.get_block_idx")
        blockI64 = op->getResult(0);
    });
    if (!blockI64)
      return;

    // 3. scf.for loop.
    scf::ForOp forOp;
    func.walk([&](scf::ForOp f) { forOp = f; });
    if (!forOp)
      return;

    // 4. Collect the 3 loop-body CVScopeOps (Flash-Attention C0/V0/C1 pattern).
    SmallVector<CVScopeOp> scopes;
    for (auto &op : *forOp.getBody())
      if (auto sc = dyn_cast<CVScopeOp>(&op))
        scopes.push_back(sc);
    if (scopes.size() != 3) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[ws] skip: expected 3 cv_scope in loop body, got "
                 << scopes.size() << "\n");
      return;
    }

    CVScopeOp c0 = scopes[0], v0 = scopes[1], c1 = scopes[2];

    // 5. Identify cross-scope shared memrefs (fixed tile shapes).
    Value scoresMem = findSharedMemref(c0, v0, {64, 64}, true);
    Value scoresCastMem = findSharedMemref(v0, c1, {64, 64}, false);
    Value accOMem = findSharedMemref(v0, c1, {64, 128}, true);
    if (!scoresMem || !scoresCastMem || !accOMem) {
      LLVM_DEBUG(llvm::dbgs() << "[ws] shared memref identification failed\n");
      return;
    }

    // 6. Workspace views (only after this kernel matches the FA pattern).
    b.setInsertionPoint(forOp);
    loc = forOp.getLoc();

    Value blkIdx =
        b.create<arith::IndexCastOp>(loc, b.getIndexType(), blockI64);

    AffineExpr s0 = getAffineSymbolExpr(0, ctx);
    Value off0 = b.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 1, s0 * PER_BLOCK_WS + OFF_SCORES, ctx),
        ValueRange{blkIdx});
    Value off1 = b.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 1, s0 * PER_BLOCK_WS + OFF_SCORES_CAST, ctx),
        ValueRange{blkIdx});
    Value off2 = b.create<affine::AffineApplyOp>(
        loc, AffineMap::get(0, 1, s0 * PER_BLOCK_WS + OFF_ACC_O, ctx),
        ValueRange{blkIdx});

    auto f32 = b.getF32Type();
    auto f16 = b.getF16Type();
    auto wsSc = MemRefType::get({64, 64}, f32);
    auto wsScc = MemRefType::get({64, 64}, f16);
    auto wsAo = MemRefType::get({64, 128}, f32);

    Value wsScores =
        b.create<memref::ViewOp>(loc, wsSc, wsArg, off0, ValueRange{});
    Value wsScoresCast =
        b.create<memref::ViewOp>(loc, wsScc, wsArg, off1, ValueRange{});
    Value wsAccO =
        b.create<memref::ViewOp>(loc, wsAo, wsArg, off2, ValueRange{});

    Value trueVal = findBoolConst(func, true);
    if (!trueVal) {
      b.setInsertionPointToStart(&func.getBody().front());
      trueVal = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
    }

    // --- Channel 1: C0→V0 (scores via fixpipe) ---
    doC0(b, c0, wsScores);

    // --- Channel 2: V0 – load scores from ws, store scores_cast to ws ---
    doV0(b, v0, scoresMem, wsScores, scoresCastMem, wsScoresCast);

    // --- Channel 3: C1 – load scores_cast, fixpipe acc_o ---
    doC1(b, c1, scoresCastMem, wsScoresCast, accOMem, wsAccO, trueVal);

    // --- V-tail: acc_o load + vadd ---
    doVTail(b, c1, accOMem, wsAccO);

    // --- Host callback ---
    emitHostCallback(b, module, func);

    LLVM_DEBUG(llvm::dbgs() << "[ws] function processed\n");
  }

  // -----------------------------------------------------------------------
  // C0: add fixpipe after mmadL1
  // -----------------------------------------------------------------------
  void doC0(OpBuilder &b, CVScopeOp c0, Value wsScores) {
    BlockArgument wsArg = addScopeInput(c0, wsScores);
    Block &body = c0.getBody().front();
    Operation *mmad = findOp(body, "hivm.hir.mmadL1");
    if (!mmad)
      return;

    // mmadL1's 'c' operand (output buffer) is at index 6.
    Value mmadOut = mmad->getOperand(6);

    mmad->setAttr("fixpipe_already_inserted",
                  BoolAttr::get(mmad->getContext(), true));

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(mmad);
    emitFixpipe(b, mmad->getLoc(), mmadOut, wsArg);
    LLVM_DEBUG(llvm::dbgs() << "[ws] C0: fixpipe for scores\n");
  }

  // -----------------------------------------------------------------------
  // V0: load scores from ws; store scores_cast to ws
  // -----------------------------------------------------------------------
  void doV0(OpBuilder &b, CVScopeOp v0, Value scoresMem, Value wsScores,
            Value scoresCastMem, Value wsScoresCast) {
    BlockArgument wsScArg = addScopeInput(v0, wsScores);
    BlockArgument wsSccArg = addScopeInput(v0, wsScoresCast);
    Block &body = v0.getBody().front();

    // --- Scores: load from ws, replace old block arg ---
    int scIdx = findInputIdx(v0, scoresMem);
    if (scIdx >= 0) {
      BlockArgument oldArg = body.getArgument(scIdx);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&body);
      Value local = allocAligned(b, v0.getLoc(),
                                  wsScArg.getType().cast<MemRefType>());
      emitLoad(b, v0.getLoc(), wsScArg, local);
      oldArg.replaceAllUsesWith(local);
      LLVM_DEBUG(llvm::dbgs() << "[ws] V0: load scores from ws\n");
    }

    // --- Scores_cast: store to ws after vcast ---
    int sccIdx = findInputIdx(v0, scoresCastMem);
    if (sccIdx >= 0) {
      BlockArgument sccArg = body.getArgument(sccIdx);
      Operation *vcast = findOp(body, "hivm.hir.vcast");
      if (vcast) {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointAfter(vcast);
        emitStore(b, vcast->getLoc(), sccArg, wsSccArg);
        LLVM_DEBUG(llvm::dbgs() << "[ws] V0: store scores_cast to ws\n");
      }
    }
  }

  // -----------------------------------------------------------------------
  // C1: load scores_cast; change mmadL1 initC → true; fixpipe acc_o
  // -----------------------------------------------------------------------
  void doC1(OpBuilder &b, CVScopeOp c1, Value scoresCastMem,
            Value wsScoresCast, Value accOMem, Value wsAccO, Value trueVal) {
    BlockArgument wsSccArg = addScopeInput(c1, wsScoresCast);
    BlockArgument wsAoArg = addScopeInput(c1, wsAccO);
    Block &body = c1.getBody().front();

    // Change init_condition from %false to %true at scope-operand level.
    for (unsigned i = 0, e = c1.getInputs().size(); i < e; ++i) {
      Value inp = c1.getInputs()[i];
      if (auto def = inp.getDefiningOp<arith::ConstantOp>()) {
        if (auto ba = def.getValue().dyn_cast<BoolAttr>()) {
          if (!ba.getValue()) {
            c1->setOperand(i, trueVal);
            break;
          }
        }
      }
    }

    // --- Scores_cast channel: load from ws ---
    int sccIdx = findInputIdx(c1, scoresCastMem);
    if (sccIdx >= 0) {
      BlockArgument oldScc = body.getArgument(sccIdx);
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&body);
      Value local = allocAligned(b, c1.getLoc(),
                                  wsSccArg.getType().cast<MemRefType>());
      emitLoad(b, c1.getLoc(), wsSccArg, local);
      oldScc.replaceAllUsesWith(local);
    }

    // --- Acc_o channel: fresh output + fixpipe ---
    Operation *mmad = findOp(body, "hivm.hir.mmadL1");
    if (mmad) {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPoint(mmad);
      Value freshOut =
          allocAligned(b, mmad->getLoc(),
                       wsAoArg.getType().cast<MemRefType>());

      mmad->setOperand(6, freshOut);
      mmad->setAttr("fixpipe_already_inserted",
                    BoolAttr::get(mmad->getContext(), true));

      b.setInsertionPointAfter(mmad);
      emitFixpipe(b, mmad->getLoc(), freshOut, wsAoArg);
    }

    LLVM_DEBUG(llvm::dbgs() << "[ws] C1: transformed\n");
  }

  // -----------------------------------------------------------------------
  // V-tail: load acc_o delta from ws, vadd to acc_o (in-place)
  // -----------------------------------------------------------------------
  void doVTail(OpBuilder &b, CVScopeOp c1, Value accOMem, Value wsAccO) {
    Location loc = c1.getLoc();
    func::FuncOp fn = c1->getParentOfType<func::FuncOp>();
    int32_t sid = allocateNextScopeId(fn);

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(c1);

    SmallVector<Value> ins = {wsAccO, accOMem};
    auto vTail = b.create<CVScopeOp>(
        loc, ins, static_cast<int32_t>(hivm::TCoreType::VECTOR), /*stage=*/0,
        sid);

    Block &body = vTail.getBodyBlock();
    BlockArgument wsArg = body.getArgument(0);
    BlockArgument accArg = body.getArgument(1);

    b.setInsertionPoint(body.getTerminator());

    Value delta =
        allocAligned(b, loc, wsArg.getType().cast<MemRefType>());
    emitLoad(b, loc, wsArg, delta);
    emitVadd(b, loc, delta, accArg, accArg);

    LLVM_DEBUG(llvm::dbgs() << "[ws] V-tail scope created\n");
  }

  // -----------------------------------------------------------------------
  // Host callback: infer_workspace_shape_function
  // -----------------------------------------------------------------------
  void emitHostCallback(OpBuilder &b, ModuleOp module, func::FuncOp entry) {
    MLIRContext *ctx = module.getContext();
    Location loc = module.getLoc();
    std::string name =
        (entry.getName() + "_infer_workspace_shape_function").str();
    if (module.lookupSymbol(name))
      return;

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(entry);

    auto fnTy = FunctionType::get(ctx, {}, {b.getIndexType()});
    auto cb = b.create<func::FuncOp>(loc, name, fnTy);

    cb->setAttr("hacc.function_kind",
                hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::HOST));
    cb->setAttr("hacc.host_func_type",
                hacc::HostFuncTypeAttr::get(
                    ctx, hacc::HostFuncType::kInferWorkspaceShapeFunction));

    Block *body = cb.addEntryBlock();
    b.setInsertionPointToStart(body);
    Value sz = b.create<arith::ConstantIndexOp>(loc, PER_BLOCK_WS);
    b.create<func::ReturnOp>(loc, sz);
  }
};

#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
