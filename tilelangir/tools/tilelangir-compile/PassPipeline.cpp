// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/tools/tilelangir-compile/PassPipeline.cpp
 * \brief TileLangIR compile pipeline (canonicalizer, CSE, SCCP, cv-split, vectorize).
 *
 */

#include "tilelangir/InitAllPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace tilelangir {

void buildTileLangIRCompilePipeline(mlir::OpPassManager &pm) {
  // MLIR community passes (3)
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSCCPPass());
  // TileLangIR custom passes (2)
  pm.addPass(mlir::tilelangir::createTileLangIRCVSplit());
  pm.addPass(mlir::tilelangir::createTileLangIRVectorize());
}

} // namespace tilelangir
