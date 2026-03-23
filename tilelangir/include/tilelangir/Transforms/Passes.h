// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/include/tilelangir/Transforms/Passes.h
 * \brief TileLangIR custom passes (TableGen declarations and registration).
 *
 */

#ifndef TILELANGIR_TRANSFORMS_PASSES_H
#define TILELANGIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tilelangir/Dialect/TileLangIR.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DECL
#include "tilelangir/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tilelangir/Transforms/Passes.h.inc"

} // namespace tilelangir
} // namespace mlir

#endif // TILELANGIR_TRANSFORMS_PASSES_H
