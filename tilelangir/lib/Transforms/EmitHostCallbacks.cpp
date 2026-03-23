// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// tilelangir-emit-host-callbacks
//
// Generate all host-side callback functions required by the NPU runtime:
//
//   1. <entry>_infer_task_type_function  → i8 constant (MIX=32, AIC=20, AIV=10)
//   2. <entry>_infer_workspace_shape_function → index constant (total WS bytes)
//
// This pass consolidates callback generation that was previously spread across
// InferTaskType and MaterializeWorkspace. Modeled after AscendNPU-IR's
// hivm-insert-infer-task-type-func / hivm-insert-infer-workspace-size-func.

#include "tilelangir/Transforms/Passes.h"

#include "tilelangir/Dialect/TileLangIR.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tilelangir-emit-host-callbacks"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIREMITHOSTCALLBACKS
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

static constexpr int8_t TASK_TYPE_AIV = 10;
static constexpr int8_t TASK_TYPE_AIC = 20;
static constexpr int8_t TASK_TYPE_MIX_1_2 = 32;

struct TileLangIREmitHostCallbacks
    : impl::TileLangIREmitHostCallbacksBase<TileLangIREmitHostCallbacks> {
  using Base = impl::TileLangIREmitHostCallbacksBase<TileLangIREmitHostCallbacks>;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> deviceFuncs;
    module.walk([&](func::FuncOp f) {
      if (f->hasAttr("hacc.entry") || f->hasAttr("hivm.entry"))
        deviceFuncs.push_back(f);
    });
    if (deviceFuncs.empty())
      return;

    MLIRContext *ctx = module.getContext();
    OpBuilder b(ctx);

    for (auto f : deviceFuncs) {
      emitInferTaskType(b, module, f);
      emitInferWorkspaceShape(b, module, f);
    }
  }

  void emitInferTaskType(OpBuilder &b, ModuleOp module, func::FuncOp entry) {
    MLIRContext *ctx = module.getContext();
    std::string name =
        (entry.getName() + "_infer_task_type_function").str();
    if (module.lookupSymbol(name))
      return;

    int8_t taskType = TASK_TYPE_MIX_1_2;
    if (auto attr = entry->getAttr("hivm.func_core_type")) {
      auto ftAttr = attr.dyn_cast<hivm::TFuncCoreTypeAttr>();
      if (ftAttr) {
        switch (ftAttr.getFuncCoreType()) {
        case hivm::TFuncCoreType::AIC:
          taskType = TASK_TYPE_AIC;
          break;
        case hivm::TFuncCoreType::AIV:
          taskType = TASK_TYPE_AIV;
          break;
        default:
          taskType = TASK_TYPE_MIX_1_2;
          break;
        }
      }
    }

    Location loc = module.getLoc();
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(entry);

    auto i8Ty = b.getIntegerType(8);
    auto fnTy = FunctionType::get(ctx, {}, {i8Ty});
    auto cb = b.create<func::FuncOp>(loc, name, fnTy);

    cb->setAttr("hacc.function_kind",
                hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::HOST));
    cb->setAttr("hacc.host_func_type",
                hacc::HostFuncTypeAttr::get(
                    ctx, hacc::HostFuncType::kInferTaskTypeFunction));

    Block *body = cb.addEntryBlock();
    b.setInsertionPointToStart(body);
    Value val = b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(i8Ty, taskType));
    b.create<func::ReturnOp>(loc, val);

    LLVM_DEBUG(llvm::dbgs()
               << "[host-cb] created " << name
               << " returning " << static_cast<int>(taskType) << "\n");
  }

  void emitInferWorkspaceShape(OpBuilder &b, ModuleOp module,
                               func::FuncOp entry) {
    MLIRContext *ctx = module.getContext();

    // Read workspace size from the attribute set by materialize-workspace.
    auto wsAttr = entry->getAttrOfType<IntegerAttr>("tilelangir.workspace_bytes");
    if (!wsAttr)
      return;
    int64_t perBlockWS = wsAttr.getInt();
    if (perBlockWS <= 0)
      return;

    std::string name =
        (entry.getName() + "_infer_workspace_shape_function").str();
    if (module.lookupSymbol(name))
      return;

    Location loc = module.getLoc();
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
    Value sz = b.create<arith::ConstantIndexOp>(loc, perBlockWS);
    b.create<func::ReturnOp>(loc, sz);

    LLVM_DEBUG(llvm::dbgs()
               << "[host-cb] created " << name
               << " returning " << perBlockWS << " bytes\n");
  }
};

} // namespace
} // namespace tilelangir
} // namespace mlir
