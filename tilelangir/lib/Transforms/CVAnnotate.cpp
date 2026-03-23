// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "tilelangir/Dialect/TileLangIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-cv-annotate"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRCVANNOTATE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

enum class OpCoreType { Neutral, Cube, Vector };

/// Resolve HIVM core type using the same rules as `HIVMCoreTypeInterface`:
/// static `CoreTypeTrait` first, otherwise `InferCoreTypeInterface::inferCoreType`
/// (see `hivm::detail::queryCoreTypeHelper` in HIVMInterfaces.cpp).
static std::optional<hivm::TCoreType> getHivmCoreType(Operation *op) {
  return hivm::detail::queryCoreTypeHelper(op);
}

/// Order ops inside a cv_scope by SSA dependencies so producers appear before
/// consumers (e.g. arith.index_cast before memref.subview that uses it). The
/// group's original order may interleave Vector ops with neutrals that depend
/// only on values outside the group; those neutrals must execute first inside
/// the region. If sorting fails (cycle), preserve block order.
static SmallVector<Operation *>
topoSortGroupOps(ArrayRef<Operation *> groupOps,
                 const DenseSet<Operation *> &groupSet) {
  if (groupOps.size() <= 1)
    return SmallVector<Operation *>(groupOps.begin(), groupOps.end());

  DenseMap<Operation *, unsigned> indeg;
  for (Operation *op : groupOps)
    indeg[op] = 0;
  for (Operation *op : groupOps) {
    for (Value v : op->getOperands()) {
      Operation *def = v.getDefiningOp();
      if (def && groupSet.contains(def))
        indeg[op]++;
    }
  }

  auto sortByBlockOrder = [](SmallVector<Operation *> &w) {
    llvm::sort(w, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
  };

  SmallVector<Operation *> result;
  SmallVector<Operation *> work;
  for (Operation *op : groupOps)
    if (indeg[op] == 0)
      work.push_back(op);
  sortByBlockOrder(work);

  while (!work.empty()) {
    Operation *op = work.front();
    work.erase(work.begin());
    result.push_back(op);
    for (Operation *other : groupOps) {
      if (other == op)
        continue;
      for (Value v : other->getOperands()) {
        if (v.getDefiningOp() == op) {
          unsigned &d = indeg[other];
          assert(d > 0);
          if (--d == 0)
            work.push_back(other);
        }
      }
    }
    sortByBlockOrder(work);
  }

  if (result.size() != groupOps.size()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cv-annotate] topo sort incomplete (cycle?); using block order\n");
    result.clear();
    Block *blk = groupOps.front()->getBlock();
    for (Operation &op : *blk)
      if (groupSet.contains(&op))
        result.push_back(&op);
  }
  return result;
}

OpCoreType classifyOp(Operation *op) {
  // Already-scoped regions: do not nest or merge (inferCoreType would match).
  if (isa<CVScopeOp>(op) || isa<CVYieldOp>(op))
    return OpCoreType::Neutral;

  if (auto coreType = getHivmCoreType(op)) {
    // Only scope HIVM ops with no SSA results (typical DPS / effect-only form).
    // Ops with results (e.g. get_block_idx, or memref.subview) stay outside the
    // region so SSA remains visible in the parent block.
    if (op->getNumResults() > 0)
      return OpCoreType::Neutral;
    switch (*coreType) {
    case hivm::TCoreType::CUBE:
      return OpCoreType::Cube;
    case hivm::TCoreType::VECTOR:
      return OpCoreType::Vector;
    default:
      // CUBE_OR_VECTOR / CUBE_AND_VECTOR: control/infra ops (e.g.,
      // set_ffts_base_addr) that run on both cores stay outside scopes.
      return OpCoreType::Neutral;
    }
  }
  return OpCoreType::Neutral;
}

struct OpGroup {
  OpCoreType type;
  SmallVector<Operation *> ops;
};

/// Create a tilelangir.cv_scope op with explicit block-arg inputs.
/// External values (defined outside the group) become inputs -> block args.
/// Internal uses are remapped to the corresponding block arguments.
static void createScope(OpBuilder &builder, MLIRContext *ctx,
                        const OpGroup &group, int &nextPipelineIndex) {
  if (group.ops.empty())
    return;

  DenseSet<Operation *> groupSet(group.ops.begin(), group.ops.end());
  SmallVector<Operation *> orderedOps = topoSortGroupOps(group.ops, groupSet);

  // Insert the scope before the earliest op of this group in the parent block
  // (program order), so all SSA operands of the scope dominate the insert pt.
  Block *parent = orderedOps.front()->getBlock();
  Operation *insertBefore = nullptr;
  for (Operation &op : *parent) {
    if (groupSet.contains(&op)) {
      insertBefore = &op;
      break;
    }
  }
  assert(insertBefore && "group ops must live in one block");
  Location loc = insertBefore->getLoc();

  // Collect external values used by ops in the group (inputs).
  SmallVector<Value> inputs;
  DenseSet<Value> inputSet;
  for (auto *op : group.ops) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !groupSet.contains(defOp)) {
        if (inputSet.insert(operand).second)
          inputs.push_back(operand);
      }
    }
  }

  // Capture the enclosing scf.for IV so downstream passes
  // (analyze-cross-scope, simple-multibuffer) can build IV-dependent flags.
  if (auto forOp = dyn_cast<scf::ForOp>(parent->getParentOp())) {
    Value iv = forOp.getInductionVar();
    if (!inputSet.contains(iv)) {
      inputs.insert(inputs.begin(), iv);
      inputSet.insert(iv);
    }
  }

  hivm::TCoreType coreType = (group.type == OpCoreType::Cube)
                                 ? hivm::TCoreType::CUBE
                                 : hivm::TCoreType::VECTOR;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);

  int32_t scopeId = nextPipelineIndex++;
  constexpr int32_t kStageId = 0;
  auto scopeOp = builder.create<CVScopeOp>(
      loc, inputs, static_cast<int32_t>(coreType), kStageId, scopeId);

  // Map external values -> block arguments.
  Block &body = scopeOp.getBodyBlock();
  IRMapping mapping;
  for (auto [input, arg] : llvm::zip(inputs, body.getArguments()))
    mapping.map(input, arg);

  // Move ops in dependency order before the implicit terminator, then remap.
  Operation *terminator = body.getTerminator();
  for (auto *op : orderedOps) {
    op->moveBefore(terminator);
    for (auto &operand : op->getOpOperands()) {
      if (auto newVal = mapping.lookupOrNull(operand.get()))
        operand.set(newVal);
    }
  }

  LLVM_DEBUG(llvm::dbgs()
             << "[cv-annotate] created cv_scope: "
             << (group.type == OpCoreType::Cube ? "CUBE" : "VECTOR")
             << " scope_id=" << scopeId << " with " << group.ops.size()
             << " ops, " << inputs.size() << " inputs\n");
}

/// Scan a block, group consecutive Cube or Vector ops, and wrap each group
/// in a tilelangir.cv_scope. Neutral ops stay in place; ops with regions
/// (scf.for etc.) act as barriers that close the current group.
static void processBlock(Block &block, int &nextPipelineIndex) {
  SmallVector<Operation *> allOps;
  for (auto &op : block)
    allOps.push_back(&op);

  SmallVector<OpGroup> groups;
  size_t i = 0;

  while (i < allOps.size()) {
    OpCoreType type = classifyOp(allOps[i]);
    if (type == OpCoreType::Neutral) {
      i++;
      continue;
    }

    OpCoreType groupType = type;
    size_t lastTypedIdx = i;
    size_t j = i + 1;

    while (j < allOps.size()) {
      OpCoreType t = classifyOp(allOps[j]);
      if (t == groupType) {
        lastTypedIdx = j;
        j++;
      } else if (t == OpCoreType::Neutral) {
        if (allOps[j]->getNumRegions() > 0)
          break;
        j++;
      } else {
        break;
      }
    }

    // Pull in leading flat neutrals (e.g. index arithmetic before memref.copy)
    // when their results are only consumed within this group's op range.
    size_t startIdx = i;
    while (startIdx > 0) {
      Operation *prev = allOps[startIdx - 1];
      if (classifyOp(prev) != OpCoreType::Neutral || prev->getNumRegions() > 0)
        break;
      if (prev->getNumResults() == 0)
        break;
      bool escapes = false;
      for (auto res : prev->getResults()) {
        for (Operation *u : res.getUsers()) {
          bool inRange = false;
          for (size_t k = startIdx - 1; k <= lastTypedIdx; ++k) {
            if (allOps[k] == u) {
              inRange = true;
              break;
            }
          }
          if (!inRange) {
            escapes = true;
            break;
          }
        }
        if (escapes)
          break;
      }
      if (escapes)
        break;
      startIdx--;
    }

    // Collect candidate ops and validate sandwiched neutrals.
    SmallVector<Operation *> candidateOps;
    for (size_t k = startIdx; k <= lastTypedIdx; k++)
      candidateOps.push_back(allOps[k]);

    DenseSet<Operation *> candidateSet(candidateOps.begin(),
                                       candidateOps.end());
    SmallVector<Operation *> groupOps;
    for (auto *op : candidateOps) {
      if (classifyOp(op) == OpCoreType::Neutral && op->getNumResults() > 0) {
        bool escapes = false;
        for (auto result : op->getResults()) {
          for (auto *user : result.getUsers()) {
            if (!candidateSet.contains(user)) {
              escapes = true;
              break;
            }
          }
          if (escapes)
            break;
        }
        if (escapes) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[cv-annotate] excluding sandwiched op with "
                        "escaping result: "
                     << *op << "\n");
          continue;
        }
      }
      groupOps.push_back(op);
    }

    if (groupOps.empty()) {
      i = lastTypedIdx + 1;
      continue;
    }

    groups.push_back({groupType, std::move(groupOps)});
    i = lastTypedIdx + 1;
  }

  MLIRContext *ctx = block.getParentOp()->getContext();
  OpBuilder builder(ctx);
  for (auto &group : groups)
    createScope(builder, ctx, group, nextPipelineIndex);
}

/// Recursively process all blocks: nested regions of each op first, then the
/// block itself. `nextPipelineIndex` assigns monotonic `scope_id` per function.
static void processRegion(Region &region, int &nextPipelineIndex) {
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      for (Region &nested : op.getRegions())
        processRegion(nested, nextPipelineIndex);
    }
    processBlock(block, nextPipelineIndex);
  }
}

struct TileLangIRCVAnnotate
    : impl::TileLangIRCVAnnotateBase<TileLangIRCVAnnotate> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    bool anyChanged = false;

    module.walk([&](func::FuncOp func) {
      if (!func->hasAttr("hivm.func_core_type"))
        return;

      LLVM_DEBUG(llvm::dbgs() << "[cv-annotate] processing function: "
                               << func.getName() << "\n");

      int nextPipelineIndex = 0;
      for (Region &region : func->getRegions())
        processRegion(region, nextPipelineIndex);

      func->setAttr("hivm.func_core_type",
                     hivm::TFuncCoreTypeAttr::get(
                         ctx, hivm::TFuncCoreType::MIX));

      if (func->hasAttr("mix_mode"))
        func->setAttr("mix_mode", StringAttr::get(ctx, "mix"));

      anyChanged = true;
    });

    if (anyChanged) {
      module->setAttr("hivm.module_core_type",
                       hivm::TModuleCoreTypeAttr::get(
                           ctx, hivm::TModuleCoreType::MIX));
    }
  }
};

} // namespace

} // namespace tilelangir
} // namespace mlir
