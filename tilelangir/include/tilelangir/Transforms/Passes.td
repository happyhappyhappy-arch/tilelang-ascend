// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TILELANGIR_TRANSFORMS_PASSES_TD
#define TILELANGIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Pass ① CV Annotate
//===----------------------------------------------------------------------===//
def TileLangIRCVAnnotate : Pass<"tilelangir-cv-annotate", "::mlir::ModuleOp"> {
  let summary = "Annotate Cube/Vector operations into scope blocks";
  let description = [{
    Group consecutive Cube or Vector HIVM operations into `tilelangir.cv_scope`
    regions. Core type is resolved via HIVM's unified lookup (`CoreTypeTrait`
    and `InferCoreTypeInterface` through `queryCoreTypeHelper`). Each scope
    carries `tcore_type`, function-unique `scope_id`, and `stage_id = 0`
    (reserved for future passes).
    Updates module_core_type and func_core_type to MIX.

    `memref.copy` ops are treated as Neutral and left in place for downstream
    passes (e.g. `tilelangir-analyze-cross-scope`) to handle.
  }];
  let dependentDialects = [
    "::mlir::tilelangir::TileLangIRDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Pass ② Insert VID
//===----------------------------------------------------------------------===//
def TileLangIRInsertVID : Pass<"tilelangir-insert-vid", "::mlir::func::FuncOp"> {
  let summary = "Insert VID-based write guards for Vector sub-blocks";
  let description = [{
    Insert Sub-Block VID (get_sub_block_idx) in Vector scopes and wrap
    workspace/GM write operations with scf.if(VID==0) guards.
    Runs early (right after cv-annotate) so downstream passes can
    rely on VID guards being in place.
  }];
  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tilelangir::TileLangIRDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Pass ③ Simple Multi-Buffer
//===----------------------------------------------------------------------===//
def TileLangIRSimpleMultiBuffer : Pass<"tilelangir-simple-multibuffer", "::mlir::func::FuncOp"> {
  let summary = "Tile loop and distribute scopes for multi-buffering";
  let description = [{
    For loops containing scope.scope blocks, perform loop tiling and
    scope distribution. This is a simplified alternative to full software
    pipelining.
  }];
  let dependentDialects = [
    "::mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"tileFactor", "tile-factor", "unsigned",
           /*default=*/"1", "Tiling factor (also determines multi-buffer count)">
  ];
}

//===----------------------------------------------------------------------===//
// Pass ④ Insert Workspace (legacy — kept for compatibility)
//===----------------------------------------------------------------------===//
def TileLangIRInsertWorkspace : Pass<"tilelangir-insert-workspace", "::mlir::ModuleOp"> {
  let summary = "Insert workspace buffers for Cube-Vector data exchange";
  let description = [{
    Allocate workspace sub-regions on GM for data transfer between Cube and
    Vector scopes. Uses fixpipe for Cube->workspace, store for Vector->workspace.
    Generates a host-side infer_workspace_shape callback function.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::tilelangir::TileLangIRDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Pass ⑤a Analyze Cross-Scope Dependencies
//===----------------------------------------------------------------------===//
def TileLangIRAnalyzeCrossScope : Pass<"tilelangir-analyze-cross-scope", "::mlir::ModuleOp"> {
  let summary = "Analyze inter-scope dependencies, insert ws_provider/ws_consumer";
  let description = [{
    After cv-annotate has wrapped HIVM ops into cv_scope regions, this pass:
      1. Builds an inter-scope dependency graph from shared memref analysis.
      2. Computes `stage_id` for each scope (longest-path on the DAG).
      3. Inserts `tilelangir.ws_provider` inside the producer scope and
         `tilelangir.ws_consumer` inside each consumer scope for cross-core
         dependencies where the shared memref lives in core-local memory.

    The provider/consumer carry a `flag` SSA value (initially = channel_id)
    that simple-multibuffer updates when unrolling.
  }];
  let dependentDialects = [
    "::mlir::tilelangir::TileLangIRDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Pass ⑤b Materialize Workspace
//===----------------------------------------------------------------------===//
def TileLangIRMaterializeWorkspace : Pass<"tilelangir-materialize-workspace", "::mlir::ModuleOp"> {
  let summary = "Materialize ws_provider/ws_consumer into HIVM DMA + workspace allocation";
  let description = [{
    Consumes `tilelangir.ws_provider` and `tilelangir.ws_consumer` ops:
      - c2v (nd2nz destination): GM-bypass — clone GM source, consumer loads
        directly from GM. No workspace allocation needed.
      - c2v (general): fixpipe in producer CUBE scope, load in consumer.
      - v2c channels: store in producer VECTOR scope, load in consumer.

    Uses `version_id` / `version_count` attributes on parent CVScopeOp to
    allocate versioned workspace sub-regions for multi-buffering.
    Annotates the function with `tilelangir.workspace_bytes`.
  }];
  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::tilelangir::TileLangIRDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Pass ⑥ Inject Block Sync
//===----------------------------------------------------------------------===//
def TileLangIRInjectBlockSync : Pass<"tilelangir-inject-block-sync", "::mlir::func::FuncOp"> {
  let summary = "Insert inter-core synchronization at scope boundaries";
  let description = [{
    Inject SyncBlockSet/SyncBlockWait with bidirectional Flag protocol
    at each cross-core transition in the loop body. Supports any number
    of scopes per loop iteration.
  }];
  let dependentDialects = [
    "::mlir::tilelangir::TileLangIRDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Pass ⑦ Outline Scope (CV split)
//===----------------------------------------------------------------------===//
def TileLangIROutlineScope : Pass<"tilelangir-outline-scope", "::mlir::ModuleOp"> {
  let summary = "Split MIX function into separate AIC and AIV functions";
  let description = [{
    Clone a MIX-typed entry function into @{name}_mix_aic and @{name}_mix_aiv.
    Each clone keeps only its core-type cv_scope ops, flattens (inlines) them,
    and erases opposite-type scopes and sync_block ops.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tilelangir::TileLangIRDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Pass ⑧ Emit Host Callbacks
//===----------------------------------------------------------------------===//
def TileLangIREmitHostCallbacks : Pass<"tilelangir-emit-host-callbacks", "::mlir::ModuleOp"> {
  let summary = "Generate all host-side callback functions for NPU runtime";
  let description = [{
    Consolidated callback generation for the NPU runtime:
      - `<entry>_infer_task_type_function` → i8 constant (task type)
      - `<entry>_infer_workspace_shape_function` → index constant (WS bytes)

    Reads `hivm.func_core_type` for task type and
    `tilelangir.workspace_bytes` (set by materialize-workspace) for WS size.
    Modeled after AscendNPU-IR's hivm-insert-infer-task-type-func and
    hivm-insert-infer-workspace-size-func.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
