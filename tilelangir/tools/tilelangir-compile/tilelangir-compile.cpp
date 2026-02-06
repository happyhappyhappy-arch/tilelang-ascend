// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/tools/tilelangir-compile/tilelangir-compile.cpp
 * \brief TileLangIR compile driver (parse MLIR/npuir, run pipeline, print IR).
 *
 */

#include "tilelangir/InitAllDialects.h"
#include "tilelangir/InitAllPasses.h"

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input .mlir or .npuir file>"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o",
    llvm::cl::desc("Output filename"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);
  ::tilelangir::registerAllDialects(registry);

  mlir::registerAllPasses();
  mlir::hfusion::registerHFusionPasses();
  ::tilelangir::registerAllPasses();

  mlir::registerAllExtensions(registry);
  bishengir::registerAllExtensions(registry);
  // TODO: Add TileLangIR extensions

  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "TileLangIR compile Tool\n");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> moduleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "Failed to parse input\n";
    return EXIT_FAILURE;
  }

  mlir::PassManager pm(&context);
  if (failed(mlir::applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Failed to apply pass manager options\n";
    return EXIT_FAILURE;
  }
  ::tilelangir::buildTileLangIRCompilePipeline(pm);

  if (failed(pm.run(*moduleRef))) {
    llvm::errs() << "Pipeline failed\n";
    return EXIT_FAILURE;
  }

  moduleRef->print(output->os(), mlir::OpPrintingFlags().useLocalScope());
  output->keep();
  return EXIT_SUCCESS;
}
