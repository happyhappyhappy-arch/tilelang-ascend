// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TILELANGIR_DIALECT_TILELANGIR_H
#define TILELANGIR_DIALECT_TILELANGIR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "tilelangir/Dialect/TileLangIRDialect.h.inc"

#include "tilelangir/Dialect/TileLangIREnums.h.inc"

#define GET_OP_CLASSES
#include "tilelangir/Dialect/TileLangIROps.h.inc"

#endif // TILELANGIR_DIALECT_TILELANGIR_H
