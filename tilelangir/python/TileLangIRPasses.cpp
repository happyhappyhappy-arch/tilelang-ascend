// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// Pybind11 module: TileLangIR run_pass_pipeline and pass registration.
// run_pass_pipeline(mlir_str, pipeline_str) runs the pipeline in-process with one
// MLIR context (mlir + bishengir + tilelangir dialects/passes). Use this instead
// of bishengir's PassManager for NPUIR so passes are in the same registry.

#include "tilelangir/InitAllDialects.h"
#include "tilelangir/InitAllPasses.h"

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>

#include <string>
#include <utility>

namespace tilelangir {
namespace python {

// When pipeline is "builtin.module(...)", use root "builtin.module" and parse only the
// inner part so passes run directly on the module (no "any" + OpToOpPassAdaptor wrapper).
// CSE runs in the "any" path but tilelangir passes do not (likely vtable dispatch in
// adaptor path); this path avoids the adaptor so tilelangir passes can run.
static bool tryBuiltinModuleRoot(llvm::StringRef pipelineStr, llvm::StringRef* inner) {
  pipelineStr = pipelineStr.trim();
  if (!pipelineStr.consume_front("builtin.module(") || !pipelineStr.consume_back(")"))
    return false;
  *inner = pipelineStr.trim();
  return true;
}

// Returns (true, result_mlir_str) on success, (false, error_message) on failure.
static std::pair<bool, std::string> runPassPipeline(llvm::StringRef mlirStr,
                                                    llvm::StringRef pipelineStr) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  bishengir::registerAllDialects(registry);
  bishengir::registerAllExtensions(registry);
  ::tilelangir::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();

  mlir::ParserConfig config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlirStr, config, "input.mlir");
  if (!module) {
    return {false, "Failed to parse MLIR string"};
  }

  llvm::StringRef innerPipeline;
  if (tryBuiltinModuleRoot(pipelineStr, &innerPipeline)) {
    // Root "builtin.module" + inner pipeline (no "any" wrapper) so tilelangir passes run.
    mlir::PassManager pm(&context, "builtin.module");
    if (failed(mlir::parsePassPipeline(innerPipeline, pm, llvm::errs()))) {
      return {false, "Failed to parse inner pass pipeline: " + innerPipeline.str()};
    }
    if (failed(pm.run(*module))) {
      return {false, "Pass pipeline run failed"};
    }
  } else {
    mlir::PassManager pm(&context);
    if (failed(mlir::parsePassPipeline(pipelineStr, pm, llvm::errs()))) {
      return {false, "Failed to parse pass pipeline: " + pipelineStr.str() +
                         " (pass may not be registered; check .so and registerAllPasses)"};
    }
    if (failed(pm.run(*module))) {
      return {false, "Pass pipeline run failed"};
    }
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os, mlir::OpPrintingFlags().useLocalScope());
  os.flush();
  return {true, result};
}

}  // namespace python
}  // namespace tilelangir

// Register passes when the .so is loaded so run_pass_pipeline can resolve pass names.
namespace {
struct RegisterPassesOnLoad {
  RegisterPassesOnLoad() {
    mlir::registerAllPasses();
    bishengir::registerAllPasses();
    tilelangir::registerAllPasses();
  }
} registerPassesOnLoad;
}  // namespace

PYBIND11_MODULE(tilelangir, m) {
  m.doc() = "TileLangIR: run_pass_pipeline(mlir_str, pipeline_str)";
  m.def(
      "run_pass_pipeline",
      [](const std::string& mlir_str, const std::string& pipeline_str) {
        auto [ok, out] = tilelangir::python::runPassPipeline(mlir_str, pipeline_str);
        if (ok) {
          return pybind11::make_tuple(true, out);
        }
        return pybind11::make_tuple(false, out);
      },
      pybind11::arg("mlir_str"), pybind11::arg("pipeline_str"),
      "Run pass pipeline on MLIR string. Returns (success: bool, result_or_error: str).");
}
