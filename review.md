# TileLangIR 编译 Pipeline 代码 Review

## 1. 总体架构

TileLangIR 编译 pipeline 将一个 MIX 模式的入口函数（同时包含 CUBE 和 VECTOR 操作）逐步变换为可在昇腾 NPU 上运行的 AIC/AIV 分离内核。完整 pass 顺序：

```
vc_input.mlir (mix_mode="aic", 单函数)
  │
  ├─ ① tilelangir-cv-annotate         将 HIVM ops 分组包入 cv_scope
  ├─ ② tilelangir-analyze-cross-scope  分析跨 scope 依赖，插入 ws_channel
  ├─ ③ tilelangir-materialize-workspace 物化 ws_channel 为 DMA + workspace 分配
  ├─ ④ tilelangir-inject-block-sync    注入跨核 sync_block_set/wait
  ├─ ⑤ tilelangir-insert-vid           为 Vector sub-block 插入 VID 写守卫
  ├─ ⑥ tilelangir-outline-scope        Split 为 AIC + AIV 两个函数
  └─ ⑦ tilelangir-infer-task-type      生成 host 回调函数
         │
         ▼
  vc_pipeline_npu.mlir (module_core_type=MIX, 含 AIC + AIV 函数)
         │
         ▼
  bishengir-compile → kernel.o (NPU 二进制)
```

---

## 2. 自定义 Op 定义

### 2.1 `tilelangir.cv_scope`

**定义位置**: `tilelangir/include/tilelangir/Dialect/TileLangIROps.td`

**作用**: 将同一核类型（CUBE 或 VECTOR）的 HIVM 操作分组到一个带显式输入依赖的区域中。是整个 pipeline 的核心数据结构——所有后续 pass 都在 `cv_scope` 粒度上分析和变换。

**关键属性**:
- `tcore_type` (i32): 核类型，取值 `hivm::TCoreType::CUBE` 或 `VECTOR`
- `stage_id` (i32): 流水线阶段编号（由 analyze-cross-scope 计算）
- `scope_id` (i32): 函数内唯一标识，用于跨 scope 依赖图

**设计要点**:
- `SingleBlockImplicitTerminator<"CVYieldOp">`: 单 block，隐式 `cv_yield` 终结符
- 外部值通过 `inputs` operand 传入，body block 的参数与 `inputs` 一一对应
- 实现 `HIVMInferCoreTypeInterface`: `inferCoreType()` 递归扫描 body ops（包括 `scf.if` 内部）统一核类型；如果 body 只有中性 ops 则回退到 `tcore_type` 属性
- Verifier 拒绝 body 内的 `scf.for`/`scf.while`（循环应在 scope 外部），但允许 `scf.if`（VID guard）

**实现位置**: `tilelangir/lib/Dialect/TileLangIR.cpp`

```cpp
void CVScopeOp::build(..., ValueRange inputs, int32_t tcoreType,
                       int32_t stageId, int32_t scopeId) {
  // 1. 添加 operands 和属性
  // 2. 创建 body region + block，为每个 input 添加 block argument
  // 3. 在 block 末尾插入 CVYieldOp 终结符
}
```

### 2.2 `tilelangir.cv_yield`

**作用**: `cv_scope` body 的终结符。无操作数，无结果。标记为 `Pure`, `ReturnLike`, `Terminator`。

### 2.3 `tilelangir.ws_channel`

**定义位置**: `tilelangir/include/tilelangir/Dialect/TileLangIROps.td`

**作用**: 标记一个需要 GM workspace 中转的跨核数据依赖。由 `analyze-cross-scope` 创建，由 `materialize-workspace` 消费并擦除。

**参数**:
- `source` (AnyType): 生产者 scope 写入的 core-local memref（cbuf/ub/cc）
- `producer_scope_id` (I32Attr): 生产者 scope 的 `scope_id`
- `consumer_scope_ids` (DenseI32ArrayAttr): 消费者 scope_id 列表（支持 1:N fan-out）
- `direction` (StrAttr): `"c2v"` 或 `"v2c"`

**结果**:
- `workspace` (AnyType): GM 类型的 memref，消费者 scope 应引用此值而非原始 source

**设计考量**: 使用 `DenseI32ArrayAttr` 而非多个 op，以便一个 producer 同时对接多个不同核类型的 consumer，避免重复分配 workspace。

---

## 3. Pass 详细实现

### 3.1 Pass ① `tilelangir-cv-annotate`

**文件**: `tilelangir/lib/Transforms/CVAnnotate.cpp`  
**粒度**: `ModuleOp`  
**作用**: 扫描函数 body，将连续的同核类型 HIVM ops 分组包入 `cv_scope`。

#### 核心逻辑

1. **`classifyOp`**: 通过 `hivm::detail::queryCoreTypeHelper` 查询每个 op 的核类型。只有无 SSA 结果的 HIVM ops（如 `nd2nz`, `mmadL1`, `vexp`）被分类为 CUBE/VECTOR。有结果的 ops（如 `get_block_idx`, `memref.subview`）和 `memref.copy` 被视为 Neutral，留在 scope 外部。

2. **分组策略 (`processBlock`)**:
   - 扫描 block 中的 ops，连续的同类型 ops 合并为一个 group
   - Neutral ops 如果在同类型 ops 之间且无 region，则可被"夹带"进 group
   - 有 region 的 ops（如 `scf.for`）作为 barrier 截断当前 group
   - 前方的 Neutral ops（如 index 计算）若结果仅在 group 内使用，会被拉入 group

3. **`createScope`**: 对每个 group：
   - 收集外部值 → 成为 `cv_scope` 的 inputs
   - 拓扑排序 group 内的 ops（保证 SSA 依赖顺序）
   - 创建 `CVScopeOp`，移入排序后的 ops，重映射 operand

4. **最终更新**: `func_core_type` → MIX, `module_core_type` → MIX

#### 递归处理

`processRegion` 先递归进入嵌套 region（如 `scf.for` body），再处理当前 block。这确保循环体内部的 ops 也被正确分组。

---

### 3.2 Pass ② `tilelangir-analyze-cross-scope`

**文件**: `tilelangir/lib/Transforms/AnalyzeCrossScope.cpp`  
**粒度**: `ModuleOp`  
**作用**: 分析 scope 间共享 memref 的数据依赖，计算 stage_id，插入 `ws_channel`。

#### 三步流程

**步骤 1: 构建依赖图 (`analyzeBlock`)**

对每个 block 中的 scope 对 (i, j)（i 在 j 前），检查它们是否共享同一个 memref operand。如果 i 是 writer（通过 `isWrittenInScope` 判定）且 j 是 reader，则建立有向边。

边的分类:
- `crossCore`: 生产者和消费者的 `tcore_type` 不同
- `needsWorkspace`: crossCore 且 memref 是 core-local 地址空间（非 GM）

`isWrittenInScope` 通过检查 HIVM ops 的 `operandSegmentSizes`（第一段 ins、第二段 outs）或 `memref.copy` 的 operand 位置来判定。

**步骤 2: 计算 stage_id (`computeStages`)**

在依赖 DAG 上跑最长路径松弛（BFS-like），将 `stage_id` 赋值给每个 scope。`stage_id = max(predecessors.stage_id) + 1`。迭代至不动点。

**步骤 3: 处理跨核依赖 (`insertWSChannels`)**

- **C→V 依赖 (`handleC2VEdge`)**: 不走 workspace。在 producer CUBE scope 中找到 `nd2nz` op，追溯其 GM source 的计算链，克隆到 scope 外部，让 VECTOR consumer 直接从 GM 加载。原因：nd2nz 目标 cbuf 在 HIVM lowering 中会被 reshape 为 4D NZ 格式，fixpipe 只能处理 2D，二者冲突。

- **V→C 依赖**: 创建 `tilelangir.ws_channel` op，并将 consumer scope 的对应 input operand 替换为 ws_channel 的 workspace result。

#### `handleC2VEdge` 详细逻辑

```
1. 找到 producer scope 中写 sharedMemref 的 nd2nz op
2. 从 nd2nz 的 source operand 反向追溯计算链（subview、reinterpret_cast 等）
3. 建立 block arg → outer operand 的 IRMapping
4. 在 producer scope 之后克隆整条计算链
5. 对每个 consumer scope:
   a. 将 input operand 替换为克隆出的 GM source
   b. 在 body 开头插入 memref.alloc + memref.copy（GM→local）
   c. 替换 body 内对旧 arg 的所有引用为新 alloc
```

---

### 3.3 Pass ③ `tilelangir-materialize-workspace`

**文件**: `tilelangir/lib/Transforms/MaterializeWorkspace.cpp`  
**粒度**: `ModuleOp`  
**作用**: 消费 `ws_channel` ops，物化为具体的 DMA 操作和 workspace 内存分配。

#### 核心逻辑

1. **Workspace 布局计算**: 为每个 `ws_channel` 分配 `[offset, size]`，按顺序排列在 workspace buffer 中。

2. **创建 workspace view**: 从函数的 `%arg2`（workspace 参数，`memref<?xi8>`）通过 `memref.view` 创建类型化的 view:
   ```
   byte_offset = block_idx * totalWsBytes + channelOffset
   wsView = memref.view %arg2[byte_offset][] : memref<?xi8> to memref<ShapexElemType>
   ```

3. **Producer 端物化**:
   - V→C: 在 producer VECTOR scope 的 writer op 之后插入 `hivm.hir.store`（UB→GM workspace）
   - C→V: 在 producer CUBE scope 的 writer op 之后插入 `hivm.hir.fixpipe`（L0C→GM workspace）

4. **Consumer 端物化**: 在 consumer scope body 开头插入 `memref.alloc` + `hivm.hir.load`（GM workspace→local），替换 block arg 的所有下游使用。

5. **Host callback**: 生成 `@{name}_infer_workspace_shape_function`，返回每个 block 的 workspace 字节数（`arith.constant` → `func.return`）。

---

### 3.4 Pass ④ `tilelangir-inject-block-sync`

**文件**: `tilelangir/lib/Transforms/InjectBlockSync.cpp`  
**粒度**: `func::FuncOp`  
**作用**: 在 scope 边界注入 AIC↔AIV 跨核同步原语。

#### Emit Helpers

- `emitSyncSet(core, tpipe, flag)`: 生成 `hivm.hir.sync_block_set`，**必须带 `tsync_instr_mode = INTRA_BLOCK_SYNCHRONIZATION`**
- `emitSyncWait(core, waitPipe, flag)`: 生成 `hivm.hir.sync_block_wait`，格式 `[core, PIPE_S, waitPipe] flag=N`
- `emitPipeBarrier()`: 生成 `hivm.hir.pipe_barrier[<PIPE_ALL>]`
- `createSyncScope(coreType)`: 创建空的 `CVScopeOp`（仅包含 sync ops），outline-scope 会将其展平到对应核的函数中

#### 3-Scope 模式 (C0, V0, C1)

适用于 `vc_input.mlir` 的 vadd 用例：循环体内 3 个 scope，数据流 C0→V0→C1。

Flag 协议（单 workspace slot）：
- **flag 0** = "data ready"（V→C: AIV 写完 workspace 后 set，AIC 读之前 wait）
- **flag 1** = "done reading"（C→V: AIC 读完后 set，AIV 下次写之前 wait）

同步插入位置：

| 位置 | 核 | 操作 |
|------|-----|------|
| Pre-loop | CUBE scope | `sync_block_set flag=1`（初始化"done reading"） |
| V0 body 开头 | VECTOR | `pipe_barrier` + `sync_block_wait flag=1` |
| V0 和 C1 之间 | VECTOR scope | `pipe_barrier` + `sync_block_set flag=0` |
| C1 body 开头 | CUBE | `sync_block_wait flag=0` |
| C1 body 末尾 | CUBE | `sync_block_set flag=1` |
| Post-loop | VECTOR scope | `pipe_barrier` + `sync_block_wait flag=1` + `pipe_barrier` |

#### 4-Scope 模式 (Flash-Attention)

循环体内 4 个 scope (C0, V0, C1, V-tail)，使用 4 个 flag 的更复杂协议。详见 `inject4ScopeSync` 函数。

#### 模式匹配

Pass 在 `scf.for` body 中收集 `CVScopeOp`，根据数量和核类型模式分发到对应 handler。

---

### 3.5 Pass ⑤ `tilelangir-insert-vid`

**文件**: `tilelangir/lib/Transforms/InsertVID.cpp`  
**粒度**: `func::FuncOp`  
**作用**: 为 Vector sub-block 插入 VID 写守卫，防止两个 sub-block 重复写入同一 GM 地址。

#### 核心逻辑

1. 遍历所有 VECTOR 类型的 `CVScopeOp`
2. 在 scope body 开头插入 VID 检查:
   ```mlir
   %vid = hivm.hir.get_sub_block_idx -> i64
   %vid_idx = arith.index_cast %vid : i64 to index
   %is_vid0 = arith.cmpi eq, %vid_idx, %c0 : index
   ```
3. 找到所有"写操作"并用 `scf.if %is_vid0 { <write_op> }` 包裹

#### `isWriteOp` 判定

| Op | 判定逻辑 |
|----|---------|
| `hivm.hir.store` | 始终视为写操作 |
| `memref.copy` source=GM | **不是写操作**（GM→UB 是 load，每个 sub-block 需独立加载） |
| `memref.copy` source=UB/其他 | 是写操作（UB→GM 是 store） |

这个方向判定是关键修复之一：如果将 GM→UB 的 `memref.copy` 错误判为 write 并包入 VID guard，sub_block_idx=1 将不加载数据，导致读到未初始化的 UB。

---

### 3.6 Pass ⑥ `tilelangir-outline-scope`

**文件**: `tilelangir/lib/Transforms/OutlineScope.cpp`  
**粒度**: `ModuleOp`  
**作用**: 将 MIX 函数 split 为 AIC 和 AIV 两个独立函数。

#### Split 流程

对每个 MIX 函数，克隆两份：

**AIC 克隆 (`keepCube=true`)**:
1. `processClone`: 删除 VECTOR scope、展平 CUBE scope、删除 VECTOR sync_block ops
2. `eraseDegenerateForLoops`: 清理空循环
3. `eraseDeadNd2NzWithNoBufferConsumers`: 删除 destination 无其他 reader 的 orphan nd2nz
4. `remapAllocAddressSpaces`: zero→cc, cbuf+mmadL1输出→cc, 其余→cbuf
5. `convertLoadsToNd2nz`: `hivm.hir.load` → `hivm.hir.nd2nz`
6. `handleMemCopiesInAIC`: dead copy 删除，live cbuf/cc→gm copy → `hivm.hir.fixpipe`
7. `promoteWorkspaceArgToGM`: workspace 参数标注 GM 地址空间

**AIV 克隆 (`keepCube=false`)**:
1. `processClone`: 删除 CUBE scope、展平 VECTOR scope、删除 CUBE sync_block ops
2. `eraseDegenerateForLoops`: 清理空循环
3. `remapAllocAddressSpaces`: cbuf/zero→ub
4. **`handleMemCopiesInAIV`**: 删除 source 无 HIVM users 的 local→GM copy（防止未初始化 UB 覆盖 AIC 输出）
5. `fixReinterpretCastResultMemSpace`: 修复 reinterpret_cast 结果的地址空间与 source 匹配
6. `promoteWorkspaceArgToGM`

#### 关键辅助函数

- **`flattenScope`**: 将 `cv_scope` body 的 ops 移到父 block 中，重映射 block args → inputs，然后删除 scope op
- **`handleMemCopiesInAIV`**: 对 AIV 函数中每个 `memref.copy local→GM`，检查 source 的根 alloc 是否有 HIVM users（如 `vexp`, `load`, `store` 等）。如果没有（说明 writer 已随 CUBE scope 删除），则删除该 copy。这是防止 AIV 中残留的 CUBE 累加器 `alloc + copy` 覆盖 AIC fixpipe 输出的关键修复
- **`remapAllocAddressSpaces`**: 根据核类型将 `memref.alloc` 的地址空间映射到正确的片上存储

#### 重要约定

- **AIC 和 AIV 都保留 `hacc.entry` 属性**。`bishengir-compile` 依赖此属性为两个函数分别生成正确的内核入口代码。删除 AIV 的 `hacc.entry` 会导致 507015 CCU 指令地址检查错误。
- **AIV 保留 `memref.copy`**，不预转换为 `hivm.hir.load/store`。DMA op 的转换交给 `bishengir-compile` 内部 pipeline 处理。

---

### 3.7 Pass ⑦ `tilelangir-infer-task-type`

**文件**: `tilelangir/lib/Transforms/InferTaskType.cpp`  
**粒度**: `ModuleOp`  
**作用**: 生成 host 侧回调函数 `@{name}_infer_task_type_function`。

逻辑简单：读取入口函数的 `hivm.func_core_type`，映射为 TaskType 常量（MIX→32, AIC→20, AIV→10），生成一个返回 `i8` 常量的 host 函数。

---

## 4. Pipeline 数据流

以 `vc_input.mlir`（M=128, N=128, K=256, D = exp(A) @ C）为例：

### 4.1 cv-annotate 后

原始函数中的操作被分组为 3 个 scope（每次循环迭代）：

```
scf.for %iter = 0 to 8 {
  cv_scope (CUBE, scope_id=0) {    // C0: load A tile via nd2nz
    nd2nz GM→cbuf (A subview)
  }
  cv_scope (VECTOR, scope_id=1) {  // V0: vexp + store to workspace
    vexp cbuf→cbuf
    memref.copy cbuf→gm (workspace)
  }
  cv_scope (CUBE, scope_id=2) {    // C1: load workspace + matmul
    nd2nz gm→cbuf (C subview)
    nd2nz gm→cbuf (workspace)
    mmadL1
  }
}
memref.copy cbuf→gm (D output)     // 循环外：结果写回
```

### 4.2 analyze-cross-scope 后

- C→V 依赖（C0 的 nd2nz 输出 → V0 的 vexp 输入）: 克隆 GM source，V0 直接从 GM 加载
- V→C 依赖（V0 的 vexp 输出 → C1 的 nd2nz 输入）: 保留为 `memref.copy`，workspace 由外部 `%arg7` 提供（本用例不走 ws_channel，因为 workspace 已经是 GM）

### 4.3 inject-block-sync 后

3-scope 模式注入 2-flag 握手协议。Pre-loop CUBE scope 初始化 flag=1。

### 4.4 outline-scope 后

生成 `@minicv_mix_aic` 和 `@minicv_mix_aiv` 两个函数：

**AIC**: `nd2nz(C) → sync_wait(0) → nd2nz(ws) → mmadL1 → sync_set(1) → fixpipe(D)`

**AIV**: `sync_wait(1) → memref.copy(A GM→UB) → vexp → memref.copy(UB→ws GM) → sync_set(0)`

---

## 5. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `tilelangir/include/tilelangir/Dialect/TileLangIROps.td` | TableGen | Op 定义（cv_scope, cv_yield, ws_channel） |
| `tilelangir/lib/Dialect/TileLangIR.cpp` | C++ | Op 实现（build, verify, print, parse, inferCoreType） |
| `tilelangir/include/tilelangir/Transforms/Passes.td` | TableGen | 7 个 pass 的注册定义 |
| `tilelangir/lib/Transforms/CVAnnotate.cpp` | C++ | Pass ①: 分组包入 cv_scope |
| `tilelangir/lib/Transforms/AnalyzeCrossScope.cpp` | C++ | Pass ②: 依赖分析 + ws_channel |
| `tilelangir/lib/Transforms/MaterializeWorkspace.cpp` | C++ | Pass ③: 物化 workspace DMA |
| `tilelangir/lib/Transforms/InjectBlockSync.cpp` | C++ | Pass ④: 跨核同步注入 |
| `tilelangir/lib/Transforms/InsertVID.cpp` | C++ | Pass ⑤: VID 写守卫 |
| `tilelangir/lib/Transforms/OutlineScope.cpp` | C++ | Pass ⑥: MIX split 为 AIC + AIV |
| `tilelangir/lib/Transforms/InferTaskType.cpp` | C++ | Pass ⑦: host 回调生成 |
| `tilelangir/lib/Transforms/CMakeLists.txt` | CMake | 构建配置 |
| `scripts/build_vc_pipeline_for_npu.sh` | Shell | 本地 pipeline 构建脚本 |
| `scripts/scp_and_run_npu_vadd.sh` | Shell | 远程 NPU 部署和测试脚本 |
| `compile_vadd_pipeline.py` | Python | NPU 全精度验证脚本 |
