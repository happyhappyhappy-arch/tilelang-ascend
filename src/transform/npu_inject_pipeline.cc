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
 * \file npu_inject_pipeline.cc
 * \brief NPU-only helpers for software pipeline injection (collect allocs,
 * merge with access info). Shared pipeline rewriter is in inject_pipeline.cc.
 */

#include "transform/inject_pipeline.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {
using namespace tir;

namespace {

void CollectAllocateBuffers(const Stmt &stmt,
                            Map<Var, Buffer> *buffer_data_to_buffer,
                            Array<Buffer> &pipeline_allocs) {
  if (!stmt.defined()) return;
  if (const auto *alloc = stmt.as<AllocateNode>()) {
    if (buffer_data_to_buffer->count(alloc->buffer_var) == 0) {
      Buffer buffer(alloc->buffer_var, alloc->dtype, alloc->extents,
                    Array<PrimExpr>{}, IntImm(DataType::Int(32), 0),
                    alloc->buffer_var->name_hint, 0, 0, BufferType::kDefault);
      buffer_data_to_buffer->Set(alloc->buffer_var, buffer);
      pipeline_allocs.push_back(buffer);
    }
    CollectAllocateBuffers(alloc->body, buffer_data_to_buffer, pipeline_allocs);
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      CollectAllocateBuffers(s, buffer_data_to_buffer, pipeline_allocs);
    }
    return;
  }
  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    CollectAllocateBuffers(realize->block->body, buffer_data_to_buffer,
                           pipeline_allocs);
    return;
  }
  if (const auto *block = stmt.as<BlockNode>()) {
    CollectAllocateBuffers(block->body, buffer_data_to_buffer, pipeline_allocs);
    return;
  }
  if (const auto *for_node = stmt.as<ForNode>()) {
    CollectAllocateBuffers(for_node->body, buffer_data_to_buffer,
                           pipeline_allocs);
    return;
  }
  if (const auto *if_node = stmt.as<IfThenElseNode>()) {
    CollectAllocateBuffers(if_node->then_case, buffer_data_to_buffer,
                           pipeline_allocs);
    if (if_node->else_case.defined()) {
      CollectAllocateBuffers(if_node->else_case.value(), buffer_data_to_buffer,
                             pipeline_allocs);
    }
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    CollectAllocateBuffers(attr->body, buffer_data_to_buffer, pipeline_allocs);
    return;
  }
}

void CollectVarsInStmt(
    const Stmt &stmt,
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> *vars) {
  if (!stmt.defined() || !vars) return;
  PostOrderVisit(stmt, [vars](const ObjectRef &obj) {
    if (const auto *v = obj.as<VarNode>()) {
      vars->insert(GetRef<Var>(v));
    }
  });
}

bool MayConflict(Region region1, Region region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = tvm::arith::IntSet::FromRange(dim1);
    auto int_set2 = tvm::arith::IntSet::FromRange(dim2);
    if (tvm::arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

std::pair<std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash,
                             ObjectPtrEqual>,
          int>
GetBufferAccessInfoFromPipeline(const PipelineInfo &pipeline_info) {
  std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual>
      infos;
  int max_stage = -1;
  for (const auto &pair : pipeline_info) {
    const Block &block = pair.first;
    int stage = pair.second.stage;
    max_stage = std::max(max_stage, stage);
    for (const BufferRegion &write : block->writes) {
      if (!infos.count(write->buffer))
        infos.emplace(write->buffer, BufferAccessInfo{});
      auto &info = infos.at(write->buffer);
      info.def = (info.def == -1) ? stage : std::min(info.def, stage);
    }
    for (const BufferRegion &read : block->reads) {
      if (!infos.count(read->buffer))
        infos.emplace(read->buffer, BufferAccessInfo{});
      infos.at(read->buffer).use =
          std::max(infos.at(read->buffer).use, stage);
    }
  }
  return {infos, max_stage};
}

int ComputeBufferVersionsForBuffer(const Buffer &buffer,
                                   const BufferAccessInfo &buffer_info,
                                   const PipelineInfo &pipeline_info) {
  if (buffer_info.def == -1) return 1;
  int num_versions = buffer_info.use - buffer_info.def + 1;
  if (num_versions >= 2) {
    bool need_multi_version = false;
    for (const auto &pair1 : pipeline_info) {
      const Block &writer_block = pair1.first;
      const auto &writer_info = pair1.second;
      auto it1 = std::find_if(
          writer_block->writes.begin(), writer_block->writes.end(),
          [&](const BufferRegion &br) { return br->buffer.same_as(buffer); });
      if (it1 == writer_block->writes.end()) continue;
      for (const auto &pair2 : pipeline_info) {
        const Block &reader_block = pair2.first;
        const auto &reader_info = pair2.second;
        auto it2 = std::find_if(
            reader_block->reads.begin(), reader_block->reads.end(),
            [&](const BufferRegion &br) { return br->buffer.same_as(buffer); });
        if (it2 == reader_block->reads.end()) continue;
        if (writer_info.order < reader_info.order &&
            writer_info.stage < reader_info.stage &&
            MayConflict((*it1)->region, (*it2)->region)) {
          need_multi_version = true;
          break;
        }
      }
    }
    if (!need_multi_version) num_versions--;
  }
  return num_versions;
}

}  // namespace

void PreparePipelineAllocsNpu(const Stmt &pipeline_body,
                              Map<Var, Buffer> *buffer_data_to_buffer,
                              Array<Buffer> &pipeline_allocs) {
  CollectAllocateBuffers(pipeline_body, buffer_data_to_buffer, pipeline_allocs);
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> vars_in_body;
  CollectVarsInStmt(pipeline_body, &vars_in_body);
  for (const auto &kv : *buffer_data_to_buffer) {
    if (vars_in_body.count(kv.first) == 0) continue;
    const Buffer &b = kv.second;
    bool found = false;
    for (const Buffer &a : pipeline_allocs) {
      if (ObjectPtrEqual()(a, b)) {
        found = true;
        break;
      }
    }
    if (!found) pipeline_allocs.push_back(b);
  }
}

Array<Buffer> MergePipelineAllocsForNpu(const PipelineInfo &pipeline_info,
                                       const Array<Buffer> &pipeline_allocs) {
  auto [infos, max_stage] = GetBufferAccessInfoFromPipeline(pipeline_info);
  (void)max_stage;
  std::vector<Buffer> out(pipeline_allocs.begin(), pipeline_allocs.end());
  for (const auto &[buffer, info] : infos) {
    if (info.def == -1) continue;
    if (ComputeBufferVersionsForBuffer(buffer, info, pipeline_info) < 2)
      continue;
    bool found = false;
    for (const Buffer &a : out) {
      if (ObjectPtrEqual()(a, buffer)) {
        found = true;
        break;
      }
    }
    if (!found) out.push_back(buffer);
  }
  return Array<Buffer>(out);
}

PipelineSegmentOptions GetNpuPipelineSegmentOptions() {
  PipelineSegmentOptions opts;
  opts.epilogue_unroll_from_zero = true;
  opts.body_delta_zero = true;
  return opts;
}

}  // namespace tl
}  // namespace tvm
