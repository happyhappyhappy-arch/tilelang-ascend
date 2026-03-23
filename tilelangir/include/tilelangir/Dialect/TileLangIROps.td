// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TILELANGIR_DIALECT_OPS_TD
#define TILELANGIR_DIALECT_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def TileLangIR_Dialect : Dialect {
  let name = "tilelangir";
  let summary = "TileLangIR dialect for Ascend NPU compilation";
  let cppNamespace = "::mlir::tilelangir";
}

class TileLangIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<TileLangIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CVYieldOp (defined before CVScopeOp for terminator reference)
//===----------------------------------------------------------------------===//

def TileLangIR_CVYieldOp : TileLangIR_Op<"cv_yield", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"CVScopeOp">
]> {
  let summary = "Terminator for a CV scope";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// CVScopeOp
//===----------------------------------------------------------------------===//

def TileLangIR_CVScopeOp : TileLangIR_Op<"cv_scope", [
    SingleBlockImplicitTerminator<"CVYieldOp">,
    SingleBlock,
    RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
      ["getEntrySuccessorOperands", "getSuccessorRegions"]>,
    DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>
]> {
  let summary = "CV scope with explicit input dependencies";
  let description = [{
    Groups Cube or Vector operations into a scope with explicit
    input dependencies via block arguments.

    Implements `hivm::InferCoreTypeInterface`: `inferCoreType` first walks
    body ops (including recursing into `scf.if` regions) and unifies
    CUBE/VECTOR via `queryCoreTypeHelper`; if the body has no definitive
    HIVM core ops, it falls back to the `tcore_type` attribute.
    Verify rejects (1) loop-like region ops in the body (scf.for, scf.while)
    but allows `scf.if` (e.g. VID guards), and (2) mixing CUBE and VECTOR
    HIVM ops in the same scope; it also checks that `tcore_type` matches
    body-based inference when both are available.

    Block arguments correspond 1:1 to the op's SSA operands (`inputs`).
    Any value used inside the region that is defined outside—including
    `scf.for` induction variables, `scf.while` block arguments, and
    loop-carried values—must appear as an operand so multibuffer / lowering
    passes can see the full closure (e.g. index `a` in `for %a = ...`).

    Inter-scope data dependence can use `scope_id` as a stable graph node id
    plus overlapping `inputs` and memory effects.

    Core attributes (set by tilelangir-cv-annotate or other builders):
      - `tcore_type` (i32): CUBE or VECTOR, matching `hivm::TCoreType`.
      - `stage_id` (i32): reserved for future pipeline / sync passes; builders
        and tilelangir-cv-annotate currently use `0`. Assign distinct values only
        when a pass explicitly needs stage ordering.
      - `scope_id` (i32): unique within the enclosing function (by convention);
        for dependence graphs and cross-scope analysis.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs,
               "int32_t":$tcoreType, "int32_t":$stageId, "int32_t":$scopeId)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
  }];
}

//===----------------------------------------------------------------------===//
// WSDirection enum
//===----------------------------------------------------------------------===//

def TileLangIR_WSDirC2V : I32EnumAttrCase<"C2V", 0, "c2v">;
def TileLangIR_WSDirV2C : I32EnumAttrCase<"V2C", 1, "v2c">;

def TileLangIR_WSDirection : I32EnumAttr<"WSDirection",
    "Cross-core workspace channel direction",
    [TileLangIR_WSDirC2V, TileLangIR_WSDirV2C]> {
  let cppNamespace = "::mlir::tilelangir";
}

//===----------------------------------------------------------------------===//
// WSProviderOp — placed inside the producing cv_scope
//===----------------------------------------------------------------------===//

def TileLangIR_WSProviderOp : TileLangIR_Op<"ws_provider", [
    HasParent<"CVScopeOp">
]> {
  let summary = "Provide scope-local data for cross-core workspace transfer";
  let description = [{
    Placed inside a producing `cv_scope` to mark that `data` (a scope-local
    memref) must be transferred to one or more consumer scopes on the other
    core via a GM workspace region.

    `channel_id` is a static identifier that pairs this provider with the
    matching `ws_consumer`(s).  `flag` is a dynamic Index SSA value computed
    as `channel_id * N + (iv % N)` where N is the multi-buffer factor
    (default 1).  When `simple-multibuffer` unrolls the loop, the flag
    expression is cloned and N is updated, giving each version a unique flag.
    `inject-block-sync` reads the flag to emit hardware sync primitives;
    `materialize-workspace` uses `channel_id` + parent scope `version_id`
    to allocate and index into the versioned workspace.
  }];

  let arguments = (ins
    AnyMemRef:$data,
    I32Attr:$channel_id,
    Index:$flag,
    TileLangIR_WSDirection:$direction
  );

  let assemblyFormat = [{
    $direction `(` $data `)` `channel` `=` $channel_id `,` `flag` `=` $flag
    attr-dict `:` type($data)
  }];
}

//===----------------------------------------------------------------------===//
// WSConsumerOp — placed inside the consuming cv_scope
//===----------------------------------------------------------------------===//

def TileLangIR_WSConsumerOp : TileLangIR_Op<"ws_consumer", [
    HasParent<"CVScopeOp">
]> {
  let summary = "Consume cross-core workspace data";
  let description = [{
    Placed inside a consuming `cv_scope` to mark that this scope requires
    data from a cross-core workspace transfer.  The result is a placeholder
    memref that `materialize-workspace` replaces with the actual loaded data.

    `channel_id` and `flag` must match the corresponding `ws_provider`.
  }];

  let arguments = (ins
    I32Attr:$channel_id,
    Index:$flag,
    TileLangIR_WSDirection:$direction
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $direction `channel` `=` $channel_id `,` `flag` `=` $flag
    attr-dict `:` type($result)
  }];
}

#endif // TILELANGIR_DIALECT_OPS_TD
