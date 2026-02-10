/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file inject_pipeline.h
 * \brief Shared types and NPU pipeline helpers for software pipeline injection.
 */

#ifndef TVM_TL_INJECT_PIPELINE_H_
#define TVM_TL_INJECT_PIPELINE_H_

#include <optional>

#include <tvm/ir/expr.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

using namespace tir;

struct PipelineAnnotation {
  int stage;
  int order;
  bool async;
};

struct BufferAccessInfo {
  int def = -1;
  int use = -1;
};

using PipelineInfo = std::unordered_map<Block, PipelineAnnotation,
                                        ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Options for pipeline segment emission (e.g. NPU uses different
 * prologue/body/epilogue bounds and index formula).
 */
struct PipelineSegmentOptions {
  bool epilogue_unroll_from_zero = false;  // epilogue loop 0..max_stage_
  bool body_delta_zero = false;            // body uses logical index = loop_var - stage
};

/*!
 * \brief NPU: segment options (epilogue unroll 0..2, body no +2 offset).
 */
PipelineSegmentOptions GetNpuPipelineSegmentOptions();

/*!
 * \brief NPU: expand pipeline_allocs with AllocateNode buffers and body vars.
 */
void PreparePipelineAllocsNpu(const Stmt &pipeline_body,
                               Map<Var, Buffer> *buffer_data_to_buffer,
                               Array<Buffer> &pipeline_allocs);

/*!
 * \brief NPU: merge pipeline_allocs with buffers that need multi-version from
 * pipeline_info.
 */
Array<Buffer> MergePipelineAllocsForNpu(const PipelineInfo &pipeline_info,
                                        const Array<Buffer> &pipeline_allocs);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_INJECT_PIPELINE_H_
