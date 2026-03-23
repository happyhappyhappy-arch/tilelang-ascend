// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <optional>

#include "tilelangir/Dialect/TileLangIR.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tilelangir;

#include "tilelangir/Dialect/TileLangIRDialectDefs.cpp.inc"
#include "tilelangir/Dialect/TileLangIREnumsDefs.cpp.inc"

namespace {

/// Result of scanning `cv_scope` body.
struct ScopeBodyCoreScan {
  /// First disallowed region op (scf.for, scf.while, etc.) — scf.if is OK.
  Operation *nestedRegionOp = nullptr;
  /// Both CUBE and VECTOR seen from `queryCoreTypeHelper` — grouping bug.
  bool mixedCubeVector = false;
  /// Single consistent CUBE or VECTOR if any such op appeared and no mix.
  std::optional<hivm::TCoreType> unifiedCore;
};

static void scanOpsForCoreType(Block &block,
                               std::optional<hivm::TCoreType> &seen,
                               ScopeBodyCoreScan &out);

static void scanOpsForCoreType(Block &block,
                               std::optional<hivm::TCoreType> &seen,
                               ScopeBodyCoreScan &out) {
  for (Operation &op : block.without_terminator()) {
    if (out.nestedRegionOp)
      return;

    if (op.getNumRegions() > 0) {
      // scf.if is allowed (e.g. VID guards from insert-vid); recurse into it.
      if (isa<scf::IfOp>(op)) {
        for (Region &region : op.getRegions())
          for (Block &innerBlock : region)
            scanOpsForCoreType(innerBlock, seen, out);
        continue;
      }
      out.nestedRegionOp = &op;
      return;
    }

    std::optional<hivm::TCoreType> ct =
        hivm::detail::queryCoreTypeHelper(&op);
    if (!ct)
      continue;
    switch (*ct) {
    case hivm::TCoreType::CUBE:
    case hivm::TCoreType::VECTOR:
      if (!seen)
        seen = *ct;
      else if (*seen != *ct)
        out.mixedCubeVector = true;
      break;
    default:
      break;
    }
  }
}

static ScopeBodyCoreScan scanScopeBodyCore(Block &block) {
  ScopeBodyCoreScan out;
  std::optional<hivm::TCoreType> seen;
  scanOpsForCoreType(block, seen, out);
  if (!out.mixedCubeVector)
    out.unifiedCore = seen;
  return out;
}

} // namespace

//===----------------------------------------------------------------------===//
// TileLangIRDialect
//===----------------------------------------------------------------------===//

void TileLangIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tilelangir/Dialect/TileLangIROpsDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CVScopeOp
//===----------------------------------------------------------------------===//

void CVScopeOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, int32_t tcoreType, int32_t stageId,
                       int32_t scopeId) {
  state.addOperands(inputs);
  state.addAttribute("tcore_type", builder.getI32IntegerAttr(tcoreType));
  state.addAttribute("stage_id", builder.getI32IntegerAttr(stageId));
  state.addAttribute("scope_id", builder.getI32IntegerAttr(scopeId));

  Region *body = state.addRegion();
  Block *block = new Block();
  body->push_back(block);
  for (auto inputType : inputs.getTypes())
    block->addArgument(inputType, state.location);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(block);
  builder.create<CVYieldOp>(state.location);
}

// Print:  tilelangir.cv_scope(%v1, %v2) {tcore_type = ..., stage_id = ...}
//           : (type1, type2) { ^bb0(%a1: t1, %a2: t2): ... }
void CVScopeOp::print(OpAsmPrinter &p) {
  p << "(";
  p.printOperands(getInputs());
  p << ")";

  p.printOptionalAttrDict((*this)->getAttrs());

  auto inputTypes = getInputs().getTypes();
  if (!inputTypes.empty()) {
    p << " : (";
    llvm::interleaveComma(inputTypes, p);
    p << ")";
  }

  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/false);
}

// Parse:  tilelangir.cv_scope(%v1, %v2) {tcore_type = ..., stage_id = ...}
//           : (type1, type2) { ^bb0(%a1: t1, %a2: t2): ... }
ParseResult CVScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  if (parser.parseLParen() ||
      parser.parseOperandList(inputOperands) ||
      parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type> inputTypes;
  if (!inputOperands.empty()) {
    if (parser.parseColon() || parser.parseLParen() ||
        parser.parseTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputOperands, inputTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  CVScopeOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

std::optional<hivm::TCoreType> CVScopeOp::inferCoreType() {
  ScopeBodyCoreScan scan = scanScopeBodyCore(getBodyBlock());
  // Invalid shapes are reported by verify(); infer stays conservative here.
  if (scan.nestedRegionOp || scan.mixedCubeVector)
    return std::nullopt;
  if (scan.unifiedCore)
    return scan.unifiedCore;

  // Fall back when the body has no definitive CUBE/VECTOR from helper
  // (e.g. only neutral index math, memref.copy, etc.).
  if (auto a = (*this)->getAttrOfType<IntegerAttr>("tcore_type"))
    return hivm::symbolizeTCoreType(static_cast<uint32_t>(a.getInt()));
  return std::nullopt;
}

LogicalResult CVScopeOp::verify() {
  if (!(*this)->getAttrOfType<IntegerAttr>("scope_id"))
    return emitOpError("requires 'scope_id' (i32) attribute");

  auto &block = getBodyBlock();
  if (getBodyRegion().getBlocks().size() != 1)
    return emitOpError("expected a single-block body region");
  if (!isa<CVYieldOp>(block.getTerminator()))
    return emitOpError("expected body to terminate with tilelangir.cv_yield");

  if (block.getNumArguments() != getInputs().size())
    return emitOpError("body block has ")
           << block.getNumArguments() << " arguments, expected "
           << getInputs().size() << " (matching inputs)";

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(block.getArguments(), getInputs()))) {
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError("block argument #") << idx << " type mismatch";
  }

  ScopeBodyCoreScan scan = scanScopeBodyCore(block);
  if (Operation *bad = scan.nestedRegionOp) {
    return emitOpError(
        "body must not contain loop-like region ops (scf.for, scf.while); "
        "scf.if is allowed — found '")
           << bad->getName() << "'";
  }
  if (scan.mixedCubeVector)
    return emitOpError(
        "body mixes CUBE and VECTOR HIVM ops; cv-annotate (or equivalent) "
        "should not merge them into one scope");

  if (auto a = (*this)->getAttrOfType<IntegerAttr>("tcore_type")) {
    if (auto fromAttr =
            hivm::symbolizeTCoreType(static_cast<uint32_t>(a.getInt()))) {
      if (scan.unifiedCore && *scan.unifiedCore != *fromAttr)
        return emitOpError(
            "tcore_type does not match core type inferred from body ops");
    }
  }

  return success();
}

OperandRange CVScopeOp::getEntrySuccessorOperands(RegionBranchPoint) {
  return getInputs();
}

void CVScopeOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(
        RegionSuccessor(&getBody(), getBodyBlock().getArguments()));
    return;
  }
  regions.push_back(RegionSuccessor());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tilelangir/Dialect/TileLangIROpsDefs.cpp.inc"
