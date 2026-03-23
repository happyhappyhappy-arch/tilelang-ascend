# TileLang-Ascend 自定义 Pass Pipeline 重构计划

## 一、背景

### 1.1 问题

当前 TileLang-Ascend 在 expert 模式下通过 `--disable-hivm-tensor-compile=true` 跳过了
`bishengir-compile` 中 `buildOptimizeHIVMPipeline` 里的两个阶段（共 41 个 pass）：

- `hivmPreBufferizationOptimizationPipeline`（35 个 pass）
- `bufferizationPipeline`（6 个 pass）

这些 pass 负责 CV 分核、排流水、workspace 管理、多缓冲、核间同步等硬件相关优化，
但它们无法很好地覆盖 TileLang 的定制需求。

### 1.2 重构目标

编写自定义 pass pipeline，在 `bishengir-compile` 之前执行，替代被跳过的 pass。

- codegen 统一到 memref 语义（仅保留 `codegen_npuir_api.cc`），
  因此 bufferization 整条线以及 tensor 相关 pass 不再需要
- codegen 已承担同步指令生成、Fixpipe、Matmul 规范化、内存空间标注等工作，
  对应的 bishengir pass 也不需要重复
- 剩余需要自行实现的核心功能：**CV 分核、多缓冲、CV 排流水、workspace 管理、
  核间同步、VID 划分、task type 推断**

### 1.3 编译流程

```
lower.py: tladapter_passes
  ├── canonicalize
  ├── adapt_triton_kernel
  ├── ① cv_annotate           ─┐
  ├── ② simple_multibuffer     │ TileLang 自定义 pass
  ├── ③ insert_workspace       │
  ├── ④ inject_block_sync      │
  ├── ⑤ insert_vid             │
  ├── ⑥ outline_scope          │ (复用 bishengir)
  ├── ⑦ infer_task_type       ─┘
  └── canonicalize / LICM / CSE

jit_npu.py:
  └── bishengir-compile [--disable-hivm-tensor-compile=true]
        └── buildOptimizeHIVMPipeline
              ├── createInitEntryKernelPass
              ├── [被跳过] hivmPreBufferizationOptimizationPipeline  ← 由上面替代
              ├── [被跳过] bufferizationPipeline                     ← 不再需要
              └── hivmPostBufferizationOptimizationPipeline          ← 仍由 bishengir 执行
```

---

## 二、硬件背景（Ascend DaVinci 架构）

理解 pass 设计需要了解以下硬件特性：

```
AI Core (一个物理核)
├── Cube Unit (1 个) —— 矩阵计算 (mmad)，访问 L0A/L0B/L0C
├── Vector Unit (2 个 Sub-Block) —— 向量计算 (vadd/vmul/vexp...)，访问 UB
│   ├── Sub-Block 0 (VID=0)
│   └── Sub-Block 1 (VID=1)
└── DMA Engines (MTE) —— 数据搬运 (GM↔L1↔UB, GM↔L0)
```

**存储层次**：GM(全局) → L1(共享) → L0A/L0B(Cube 输入) → L0C(Cube 输出) → UB(Vector)

**同步机制**：
- **核内同步**（同一核内不同 pipeline 间）：
  - `pipe_barrier[<PIPE_ALL>]`：等待所有流水线排空，正确但性能差
  - `set_flag` / `wait_flag`：基于 Event ID（0-4）的细粒度同步，由 `InjectSync` NORMAL 模式生成
- **核间同步**（AIC↔AIV 通过 workspace 交换数据时）：
  `SyncBlockSet` / `SyncBlockWait`，16 个 Flag ID，需要 FFTS base address

**关键约束（来自 vadd 穿刺验证）**：
- bishengir 的 `InjectSync` NORMAL 模式有严格的 op 白名单（`IRTranslator`），
  **不认识** `memref.alloc`（带 strided layout）和 `memref.copy` 等 op
- 如果我们生成的 AIV IR 中包含这些 op，NORMAL 模式会报错 `"InjectSync Fail : Unrecognized type of Operation touches local or global buffer!"`
- **解决方案**：在我们的 pass 中**手动插入 `pipe_barrier[<PIPE_ALL>]`**，
  绕过 bishengir 的 InjectSync（使用 `--disable-auto-inject-sync` 或让 NORMAL 模式因无 HIVM op 而跳过）

**Cube 与 Vector 可并行执行**，软件流水是提升性能的关键。

---

## 三、Pass Pipeline 总览

```python
# lower.py
tladapter_passes = [
    transforms.mlir.canonicalize(top_down=True),
    transforms.bishengir.adapt_triton_kernel,

    # ---- TileLang 自定义 pass (tilelangir) ----
    transforms.tilelangir.cv_annotate,         # ① CV 分核标注
    transforms.tilelangir.simple_multibuffer,  # ② 多缓冲 + 循环分配
    transforms.tilelangir.insert_workspace,    # ③ 插入 Workspace
    transforms.tilelangir.inject_block_sync,   # ④ 插入核间同步
    transforms.tilelangir.insert_vid,          # ⑤ 插入 VID
    transforms.tilelangir.infer_task_type,     # ⑥ 推断 Task Type (必须在 outline 前)
    transforms.tilelangir.outline_scope,       # ⑦ CV 拆分 (MIX→AIC+AIV)

    # ---- 通用优化 ----
    transforms.mlir.canonicalize(top_down=True),
    transforms.mlir.licm,
    transforms.mlir.cse,
]
```

Pass 之间的数据流：

```
原始 IR (memref 语义，C/V 操作混合在一个函数中，通过 UB memref 共享数据)
  │
  ▼ ① cv_annotate
连续的同类型操作被聚拢到 tilelangir.cv_scope 中，标注 tcore_type 和 stage_id
  │
  ▼ ② simple_multibuffer
循环按 tile_factor 切块，每个 scope 分配独立的内层循环 + 多缓冲
  │
  ▼ ③ insert_workspace
scope 间的共享 UB 引用被替换为 workspace (GM) 读写：
  - Cube→workspace 用 fixpipe (L0C→GM 硬件直通)
  - Vector→workspace 用 store (UB→GM)
  - acc_o 的 mmadL1 累加被拆分: Cube 写 fresh 结果, Vector 做 vadd 累加
  - workspace 按 per-block 偏移 (block_idx × size)
  - 生成 host 端 infer_workspace_shape_function 回调
  │
  ▼ ④ inject_block_sync
双向同步协议: 每条 workspace 通道分配 "就绪" + "已读完" 两个 Flag 信号
时序不重叠的 "就绪" 信号可复用同一 Flag ID; 循环前插入初始 "已读完" 信号
  │
  ▼ ⑤ insert_vid
Vector scope 内插入 VID (get_sub_block_idx) 写入守卫:
两个 Sub-Block 冗余计算, 仅 VID=0 执行 workspace 写入和 GM 写出
  │
  ▼ ⑥ infer_task_type
根据入口函数的 core type 生成 host 端 task type 回调函数（在 outline 前执行以获取原始函数名）
  │
  ▼ ⑦ outline_scope
MIX 函数被 clone + filter + flatten 为 @xxx_mix_aic 和 @xxx_mix_aiv 两个独立函数
  │
  ▼ bishengir-compile (hivmPostBufferizationOptimizationPipeline)
```

---

## 四、各 Pass 详细设计

> **穿刺（Spike）说明**：当前 Stage 1–6 的实现均为穿刺方案，专门针对
> Flash Attention (C0→V0→C1 模式) 做了 case-specific 的硬编码处理。
> 泛化处理将在穿刺验证通过后逐步推进。

### Pass ① CV 分核标注 (`tilelangir-cv-annotate`)

**目标**：识别函数内的 Cube 和 Vector 操作，将连续的同类型操作聚拢到
`tilelangir.cv_scope` 块中，标注 `tcore_type` 和 `stage_id` 属性。

#### 1.1 `tilelangir` 方言设计

**为什么不用 `scope.scope`**：bishengir 的 `scope.scope` 没有 block arguments，
无法显式表达 scope 的输入依赖。后续 pass（workspace 插入、同步分析）需要知道
每个 scope 读写了哪些外部 memref。通过 `cv_scope` 的显式 inputs → block args 映射，
scope 间的共享 memref 一交集就能得到。

**Op 定义**：

```
tilelangir.cv_scope — 带显式 inputs 的 CV 作用域
├── inputs: Variadic<AnyType>   — 外部值（memref、scalar）作为 operands
├── body: SizedRegion<1>        — block args 与 inputs 一一对应
├── tcore_type: i32 attr        — 1=CUBE, 2=VECTOR（discardable attr）
├── stage_id: i32 attr          — 同类型 scope 的序号（discardable attr）
└── terminator: tilelangir.cv_yield（无参数，纯终结器）
```

**不需要 yield 值**：在 memref 语义下，写操作是 in-place 修改内存，不产生新的
SSA 值。scope 间的数据依赖通过 **inputs 集合交集 + 读写效应分析** 检测：

```python
# 后续 inject_block_sync pass 的依赖分析逻辑
for (A, B) in adjacent_scope_pairs:
    shared = set(A.inputs) ∩ set(B.inputs)   # 共享 memref
    for memref in shared:
        a_writes = any(op writes memref for op in A.body)  # HIVM outs()
        b_reads  = any(op reads  memref for op in B.body)  # HIVM ins()
        if a_writes and b_reads:
            insert_sync_between(A, B)  # RAW dependency
```

**IR 格式**：

```mlir
tilelangir.cv_scope(%buf1, %buf2, %scalar) {tcore_type = 1 : i32, stage_id = 0 : i32}
    : (memref<64x128xf16>, memref<64x64xf32>, i1) {
^bb0(%arg0: memref<64x128xf16>, %arg1: memref<64x64xf32>, %arg2: i1):
  hivm.hir.nd2nz ins(%arg0 : ...) outs(%arg1 : ...)
}
```

#### 1.2 Op 分类算法 (`classifyOp`)

每个 op 被分为三类：`Cube`、`Vector`、`Neutral`。

```
classifyOp(op):
  1. 调用 hivm::detail::queryCoreTypeHelper(op)
     — 该函数通过 MLIR traits/interfaces 查询 op 的 TCoreType
  2. 如果有 core type:
     a. op 有 SSA results (op.numResults > 0) → Neutral
        原因: 如 get_block_idx 产出值在 scope 外使用，
        移入 scope 会导致 SSA dominance 违规
     b. TCoreType::CUBE → Cube
     c. TCoreType::VECTOR → Vector
     d. CUBE_OR_VECTOR / CUBE_AND_VECTOR → Neutral
        原因: 如 set_ffts_base_addr 是控制/基础设施 op，
        在两种核上都需运行，不应被限制在单一核 scope 中
  3. 如果无 core type:
     a. op name == "memref.copy" → Vector
        原因: DMA 写回（UB/cbuf → GM）属于 Vector 侧操作
     b. 其他 → Neutral
```

**分类覆盖表**：

| Op | queryCoreTypeHelper | numResults | 最终分类 | 说明 |
|----|---------------------|------------|---------|------|
| `hivm.hir.nd2nz` | CUBE | 0 | **Cube** | DMA: GM→L1 |
| `hivm.hir.mmadL1` | CUBE | 0 | **Cube** | 矩阵乘 |
| `hivm.hir.vmul/vadd/vexp/...` | VECTOR | 0 | **Vector** | 向量计算 |
| `hivm.hir.vbrc` | VECTOR | 0 | **Vector** | 标量广播 |
| `hivm.hir.vcast` | VECTOR | 0 | **Vector** | 类型转换 |
| `hivm.hir.vreduce` | VECTOR | 0 | **Vector** | 归约 |
| `memref.copy` | 无 | 0 | **Vector** | DMA: UB→GM 写回 |
| `hivm.hir.get_block_idx` | CUBE_OR_VECTOR | 1 | **Neutral** | 有 result |
| `hivm.hir.set_ffts_base_addr` | CUBE_AND_VECTOR | 0 | **Neutral** | 双核控制 |
| `arith.constant/muli/...` | 无 | 1 | **Neutral** | 标量计算 |
| `memref.alloc/subview/...` | 无 | 1 | **Neutral** | 地址计算 |
| `scf.for` | 无 | 0+ | **Neutral** | 循环（且作为 barrier） |

#### 1.3 分组算法 (`processBlock`)

对一个 block 中的 op 序列进行贪心分组：

```
processBlock(block):
  allOps = snapshot(block.ops)
  groups = []
  cubeStageId = 0, vectorStageId = 0
  i = 0

  while i < len(allOps):
    type = classifyOp(allOps[i])
    if type == Neutral:
      i++; continue

    # ---- 贪心扩展：尽可能多地收纳同类型 op ----
    groupType = type
    lastTypedIdx = i    # 最后一个确定类型的 op 位置
    j = i + 1

    while j < len(allOps):
      t = classifyOp(allOps[j])
      if t == groupType:
        lastTypedIdx = j; j++         # 同类型，继续扩展
      elif t == Neutral:
        if allOps[j].numRegions > 0:
          break                        # 有 region 的 op 是 barrier (scf.for 等)
        j++                            # 普通 neutral，跳过继续看后面
      else:
        break                          # 遇到对立类型，停止

    # ---- 收集 candidate ops: [i .. lastTypedIdx] ----
    candidateOps = allOps[i : lastTypedIdx+1]
    candidateSet = Set(candidateOps)

    # ---- 验证夹心 neutral ops 的 SSA 安全性 ----
    groupOps = []
    for op in candidateOps:
      if classifyOp(op) == Neutral and op.numResults > 0:
        escapes = false
        for result in op.results:
          for user in result.users:
            if user not in candidateSet:
              escapes = true; break
        if escapes:
          continue    # 排除：该 neutral op 的结果被 scope 外使用
      groupOps.append(op)

    if groupOps is empty:
      i = lastTypedIdx + 1; continue

    stageId = cubeStageId++ if groupType==Cube else vectorStageId++
    groups.append(OpGroup{groupType, groupOps, stageId})
    i = lastTypedIdx + 1

  # ---- 为每个 group 创建 cv_scope ----
  for group in groups:
    createScope(group)
```

**关键设计决策**：

1. **Barrier 语义**：带 region 的 op（`scf.for`、`scf.if`）强制截断当前 group。
   这确保循环体内外的 scope 边界清晰，不会把循环体内的 op 和循环外的混在一起。

2. **夹心 Neutral 排除**：如果一个 Neutral op（如 `memref.subview`）夹在两个
   同类型 typed op 之间，其结果被 scope 外的 op 使用，则必须排除该 op。
   否则移入 scope 后会导致 SSA dominance 违规（region 内定义的值在 region 外不可见）。

3. **Stage ID 独立编号**：Cube 和 Vector 各自维护独立的 stage_id 计数器。
   循环体内典型模式 `C0, V0, C1` 对应 `cube_stage=0,1; vector_stage=0`。

#### 1.4 Scope 创建算法 (`createScope`)

```
createScope(group):
  groupSet = Set(group.ops)

  # ---- 收集外部值（inputs）----
  inputs = []
  for op in group.ops:
    for operand in op.operands:
      defOp = operand.definingOp
      if defOp is None or defOp not in groupSet:
        if operand not in inputs:
          inputs.append(operand)

  # ---- 创建 cv_scope op ----
  scopeOp = builder.create<CVScopeOp>(
      loc, inputs, tcore_type, stage_id)
  # CVScopeOp builder 自动创建：
  #   - body region + block
  #   - block args (与 inputs 类型一一对应)
  #   - cv_yield terminator

  # ---- 建立 external value → block arg 映射 ----
  mapping = {}
  for (input, blockArg) in zip(inputs, scopeOp.body.args):
    mapping[input] = blockArg

  # ---- 将 ops 移入 scope body, 并重映射 operands ----
  for op in group.ops:
    op.moveBefore(terminator)
    for operand in op.operands:
      if operand in mapping:
        operand.set(mapping[operand])
```

**为什么 inputs 收集能正确工作**：

- 组内 op 互相引用的值（如 `nd2nz` 的 `outs` 被 `mmadL1` 的 `ins` 引用）
  不会成为 input，因为 `defOp ∈ groupSet`
- 组外定义的值（如 `memref.alloc`、`arith.constant`、循环变量）会成为 input
- Block args 是 block 参数而非 op 结果，`defOp` 为 null，也会被收集为 input

#### 1.5 递归处理 (`processRegion`)

```
processRegion(region):
  for block in region:
    # 先递归处理内层 region（bottom-up）
    for op in block:
      for nested in op.regions:
        processRegion(nested)
    # 再处理当前 block
    processBlock(block)
```

Bottom-up 顺序确保内层循环的 scope 先创建完毕，外层再处理。
这样 `scf.for` 体内的 ops 已被分组到 scope 中，外层看到的是 scope op 而非原始 op。

#### 1.6 顶层入口

```
runOnOperation():
  module = getOperation()
  for func in module.walk<FuncOp>():
    if not func.hasAttr("hivm.func_core_type"):
      continue
    for region in func.regions:
      processRegion(region)
    func.setAttr("hivm.func_core_type", MIX)
    func.setAttr("mix_mode", "mix")
  module.setAttr("hivm.module_core_type", MIX)
```

#### 1.7 Flash Attention 示例输出

输入 `C0 → V0 → C1` 模式，输出 6 个 scope（循环前 2 + 循环内 3 + 循环后 1）：

```mlir
// 循环前
tilelangir.cv_scope(%subview, %alloc) {tcore_type = 1, stage_id = 0}
    : (memref<64x128xf16, ..., gm>, memref<64x128xf16, ..., cbuf>) {
^bb0(%arg13: ..., %arg14: ...):
  hivm.hir.nd2nz ins(%arg13) outs(%arg14) ...
}

tilelangir.cv_scope(%cst, %alloc_7, ...) {tcore_type = 2, stage_id = 0}
    : (f32, memref<64x128xf32, ..., zero>, ...) {
^bb0(%arg13: f32, %arg14: ..., ...):
  hivm.hir.vbrc ins(%arg13) outs(%arg14)
  ...
}

// 主循环
scf.for %i = ... {
  // C0: Load K + QK gemm (inputs: K_gm_subview, K_l1, Q_l1, ...)
  tilelangir.cv_scope(%subview_20, %alloc_10, %alloc, %true, %c64, %c128, %alloc_12)
      {tcore_type = 1, stage_id = 0} : (...) {
  ^bb0(...):
    hivm.hir.nd2nz ins(%arg14) outs(%arg15) ...
    hivm.hir.mmadL1 ins(%arg16, %arg15, %arg17, ...) outs(%arg20)
  }

  // V0: Softmax (inputs: scores, scales, acc_m, acc_l, acc_o, ...)
  tilelangir.cv_scope(%alloc_12, %alloc_8, ...)
      {tcore_type = 2, stage_id = 0} : (...) {
  ^bb0(%arg14: memref<64x64xf32, ..., zero>, %arg15: ..., ...):
    hivm.hir.vmul ins(%arg14, %arg15) outs(%arg14) ...
    ...14 个 softmax ops...
  }

  // C1: Load V + SV gemm (inputs: V_gm_subview, V_l1, scores_cast, ...)
  tilelangir.cv_scope(%subview_21, %alloc_11, %alloc_13, ...)
      {tcore_type = 1, stage_id = 1} : (...) {
  ^bb0(...):
    hivm.hir.nd2nz ins(%arg14) outs(%arg15) ...
    hivm.hir.mmadL1 ins(%arg16, %arg15, ...) outs(%arg20)
  }
}

// 循环后
tilelangir.cv_scope(%alloc_7, %alloc_6, %alloc_9, %reinterpret_cast_4, %3)
    {tcore_type = 2, stage_id = 1} : (...) {
^bb0(...):
  hivm.hir.vdiv ins(%arg13, %arg14) outs(%arg13) ...
  hivm.hir.vcast ins(%arg13) outs(%arg15)
  %subview_10 = memref.subview %arg16[%arg17, 0] ...
  memref.copy %arg15, %subview_10
}
```

**跨 scope 的共享 memref（后续 pass 依赖分析的关键）**：

| 共享 memref | Scope A (producer) | Scope B (consumer) | 依赖类型 |
|-------------|-------------------|--------------------|---------|
| `%alloc_12` (scores) | C0: mmadL1 writes | V0: vmul reads | RAW |
| `%alloc_13` (scores_cast) | V0: vcast writes | C1: mmadL1 reads | RAW |
| `%alloc_7` (acc_o) | C1: mmadL1 writes | V0(next iter): vmul reads | RAW (cross-iter) |

**替代的 bishengir pass**：`InferFuncCoreType` + `InsertLoadStoreForMixCV`

---

### Pass ② 多缓冲 + 循环分配 (`tilelangir-simple-multibuffer`)

**目标**：将包含多个 `cv_scope` 的循环做 loop tiling + distribution，使每个 scope
获得独立的内层循环，天然实现多缓冲和 C/V 并行的前置条件。

**硬件动机**：Cube 和 Vector 是独立的计算单元，可并行执行。
通过将循环切块并让每个 scope 批量执行一组迭代，
同一类型的连续迭代可以利用多缓冲隐藏 DMA 延迟，
不同类型的 scope batch 之间可以通过 sync 实现跨 batch 的并行。

**核心变换：loop tiling + scope distribution**

```
输入 (① 之后):                   输出 (tile_factor=4):
for i in 0..N:                    for i_outer in 0..(N/4):
  scope C0(i)                       for j in 0..4: scope C0(i_outer*4+j)
  scope V0(i)                       for j in 0..4: scope V0(i_outer*4+j)
  scope C1(i)                       for j in 0..4: scope C1(i_outer*4+j)
```

**变换示意**（Flash Attention，N=8，tile_factor=4）：

```mlir
// 输入
scf.for %i = %c0 to %c8 step %c1 : i32 {
  scope.scope { /* C0[i]: nd2nz K + mmadL1 Q×K */ } {tcore_type = CUBE, stage_id = 0}
  scope.scope { /* V0[i]: softmax 14 ops */        } {tcore_type = VECTOR, stage_id = 0}
  scope.scope { /* C1[i]: nd2nz V + mmadL1 S×V */  } {tcore_type = CUBE, stage_id = 1}
}

// 输出
scf.for %i_outer = %c0 to %c2 step %c1 : i32 {
  %base = arith.muli %i_outer, %c4 : i32

  // ---- C0 batch: Cube 连续跑 4 次 QK gemm ----
  scf.for %j = %c0 to %c4 step %c1 : i32 {
    %i = arith.addi %base, %j : i32
    scope.scope {
      // nd2nz K[i] → alloc_11[j % 4]  (4 个 buffer 实例)
      // mmadL1 Q × K[i] → alloc_13[j % 4]
      scope.return
    } {tcore_type = #hivm.tcore_type<CUBE>, stage_id = 0 : i32}
  }

  // ---- V0 batch: Vector 连续跑 4 次 softmax ----
  scf.for %j = %c0 to %c4 step %c1 : i32 {
    %i = arith.addi %base, %j : i32
    scope.scope {
      // softmax(alloc_13[j % 4]) → alloc_14[j % 4]
      scope.return
    } {tcore_type = #hivm.tcore_type<VECTOR>, stage_id = 0 : i32}
  }

  // ---- C1 batch: Cube 连续跑 4 次 SV gemm ----
  scf.for %j = %c0 to %c4 step %c1 : i32 {
    %i = arith.addi %base, %j : i32
    scope.scope {
      // nd2nz V[i] → alloc_12[j % 4]
      // mmadL1 alloc_14[j % 4] × V[i] → acc_o
      scope.return
    } {tcore_type = #hivm.tcore_type<CUBE>, stage_id = 1 : i32}
  }
}
```

**多缓冲自然产生**：
- `tile_factor = 4` 意味着每个 scope batch 需要 4 份独立 buffer
- C0 batch 产出 4 份 scores（`alloc_13[0..3]`），V0 batch 消费这 4 份
- buffer 数量 = tile_factor，不需要单独的 mark_multi_buffer pass

**并行机会**：
```
时间线 (外层迭代 0):
  Cube:   [C0×4]            [C1×4]
  Vector:        [V0×4]

时间线 (跨外层迭代，sync 正确时):
  Cube:   [C0×4 #0][C1×4 #0][C0×4 #1][C1×4 #1]
  Vector:           [V0×4 #0]         [V0×4 #1]
                    ↑                  ↑
              与 C1 batch #0 并行   与 C1 batch #1 并行
```

关键：当 ④ inject_block_sync 在 scope batch 边界处正确插入 sync 后，
C1 batch[i] 和 V0 batch[i+1] 就可以并行执行——Cube 在做第 i 组的 S×V 时，
Vector 可以同时做第 i+1 组的 softmax。

**实现要点**：
- 识别包含 `cv_scope` 的 `scf.for` 循环
- 将循环 tiling：外层步长 = tile_factor，内层 trip count = tile_factor
- 为每个 scope 生成独立的内层循环（loop distribution）
- 循环内的 alloc 扩展为 tile_factor 份（多缓冲），用 `%j % tile_factor` 索引
- tile_factor 作为 pass 参数可配（默认 2）
- 需要处理 trip count 不被 tile_factor 整除的边界情况

**替代的 bishengir pass**：`MarkMultiBuffer` + `CVPipelining` + `TileCubeVectorLoop`

---

### Pass ③ 插入 Workspace (`tilelangir-insert-workspace`)

**目标**：为 Cube scope 和 Vector scope 之间的数据传递分配 workspace 缓冲区，
规划内存布局，并生成 host 端 workspace size 回调。

**硬件动机**：Cube 和 Vector 各自有独立的 on-chip 存储（L0A/B/C vs UB），
不能直接互访。当 Cube 的计算结果需要传给 Vector（或反向）时，
必须通过 Global Memory 上的 workspace 中转。

**关键设计点（来自参考输出验证）**：

1. **Cube→workspace 使用 fixpipe**：Cube 的 mmadL1 结果在 L0C 中，
   通过 `hivm.hir.fixpipe` 直接写到 workspace（GM），比 memref.copy 更高效。
   fixpipe 是硬件支持的 L0C→GM 通路，附带格式转换（NZ→ND）。
2. **acc_o 累加拆分**：原 IR 中 `mmadL1(initC=false)` 直接累加到 acc_o，
   但拆分后 Cube 无法访问 Vector 持有的 acc_o（在 Vector 的 UB 中）。
   需要拆分为：Cube 只算乘法结果写到 workspace，Vector 加载后做 `vadd` 累加。
3. **Workspace 是 per-block 的**：每个 AI Core (block) 有独立的 workspace 区域，
   偏移量 = `block_idx × per_block_size`。host 回调返回 **per-block** 大小，
   runtime 负责乘以 block 数。
4. **Vector→workspace 使用 store**：Vector 结果在 UB 中，
   通过 `hivm.hir.store` 写到 workspace（GM）。

#### 3.1 实现算法概览

整个 pass 在 `ModuleOp` 层面运行，遍历每个带 `hivm.func_core_type` 属性的函数，
执行以下步骤：

```
processFunction(func, module):
  1. 定位 workspace 参数（第 2 个 memref<?xi8>，由 WorkspaceArgIdx=1 指示）
  2. 获取 block_idx（hivm.hir.get_block_idx → i64）
  3. 找到 scf.for 主循环
  4. 在循环前创建 3 个 workspace view
  5. 在循环体内找到 3 个 CVScopeOp（C0, V0, C1）
  6. 识别跨 scope 共享 memref（通过 inputs 交集 + 类型匹配）
  7. 变换各通道：C0→V0(scores), V0→C1(scores_cast), C1→V0(acc_o)
  8. 新增 V-tail scope 完成 acc_o 累加
  9. 生成 host 回调函数
```

#### 3.2 跨 scope 数据依赖分析

利用 `cv_scope` 的显式 inputs 设计，跨 scope 数据流检测极其简洁：

```
findSharedMemref(scope_A, scope_B, shape, elemType):
  for v1 in scope_A.inputs:
    if v1.type != MemRefType or v1.shape != shape or v1.elemType != elemType:
      continue
    for v2 in scope_B.inputs:
      if v1 == v2:            // 同一个 Value 出现在两个 scope 的 inputs 中
        return v1             // → 该 memref 是跨 scope 的共享数据
  return null
```

**Flash Attention 中识别到的 3 条跨 scope 通道**：

| 共享 memref | 类型 | Scope A | Scope B | 通道方向 |
|-------------|------|---------|---------|---------|
| `%alloc_12` | `64×64×f32` | C0 (mmadL1 writes) | V0 (vmul reads) | C→V (fixpipe) |
| `%alloc_13` | `64×64×f16` | V0 (vcast writes) | C1 (mmadL1 reads) | V→C (store/load) |
| `%alloc_7` | `64×128×f32` | V0 (vmul writes) | C1 (mmadL1 writes) | C→V (累加拆分) |

#### 3.3 Workspace 视图创建

在 `scf.for` 循环之前，基于 `block_idx` 计算 per-block workspace 地址并创建 typed view：

```
createWorkspaceViews(builder, blockIdxI64, wsArg):
  blkIdx = arith.index_cast(blockIdxI64) : i64 → index

  // 3 个 affine_map，对应 3 个 sub-region
  off0 = affine.apply <()[s0] -> (s0 * 57344)>[blkIdx]           // scores
  off1 = affine.apply <()[s0] -> (s0 * 57344 + 16384)>[blkIdx]  // scores_cast
  off2 = affine.apply <()[s0] -> (s0 * 57344 + 24576)>[blkIdx]  // acc_o

  wsScores     = memref.view wsArg[off0][] → memref<64×64×f32>
  wsScoresCast = memref.view wsArg[off1][] → memref<64×64×f16>
  wsAccO       = memref.view wsArg[off2][] → memref<64×128×f32>
```

**workspace view 的地址空间处理（来自 vadd 穿刺验证）**：

workspace 参数 `%arg2` **必须预标注** `#hivm.address_space<gm>`，否则 bishengir 的
`InferHIVMMemScope` pass 会尝试传播地址空间，遇到不认识的 user op 就会失败。

```mlir
// 正确：预标注地址空间，InferHIVMMemScope 跳过传播
%arg2: memref<?xi8, #hivm.address_space<gm>>
%ws_view = memref.view %arg2[%off][] : memref<?xi8, #hivm.address_space<gm>>
    to memref<64x64xf32, #hivm.address_space<gm>>

// 错误：无地址空间，InferHIVMMemScope 传播失败
%arg2: memref<?xi8>
```

**不能使用 `memref_ext.alloc_workspace`**：该 op 未实现 `ViewLikeOpInterface`，
不被 `InferHIVMMemScope` 识别，会导致编译失败。

**不能使用 `memref.reinterpret_cast`**：不支持元素类型变更（`i8` → `f16`），
必须使用 `memref.view` 来完成字节级 reinterpret。

#### 3.4 通道变换：辅助操作 `addScopeInput`

向 `CVScopeOp` 动态添加新 input 的核心辅助函数：

```
addScopeInput(scope, newValue):
  n = scope.inputs.size()
  scope.insertOperands(n, [newValue])      // 在 operand 末尾追加
  return scope.body.front().addArgument(    // 添加对应的 block argument
      newValue.type, scope.loc)
```

这使得 workspace view 能被注入到已有的 scope 中，而不需要重建 scope。

#### 3.5 通道 1: C0→V0 (scores via fixpipe)

```
doC0(c0Scope, wsScores):
  // 1. 添加 workspace view 作为新 input → 获得 block arg
  wsArg = addScopeInput(c0Scope, wsScores)

  // 2. 在 C0 body 内找到 mmadL1
  mmad = findOp(c0Scope.body, "hivm.hir.mmadL1")
  mmadOut = mmad.operand[6]        // 'c' 操作数 = scores buffer

  // 3. 标记 fixpipe 已手动插入（避免后续 bishengir 重复插入）
  mmad.setAttr("fixpipe_already_inserted", true)

  // 4. 在 mmadL1 之后插入 fixpipe
  builder.insertAfter(mmad):
    hivm.hir.fixpipe {enable_nz2nd}
      ins(mmadOut)        // L0C 中的 scores
      outs(wsArg)         // workspace GM view
```

**效果**：mmadL1 结果通过硬件 fixpipe 直通路径写入 workspace。

#### 3.6 通道 2: V0 — load scores, store scores_cast

```
doV0(v0Scope, scoresMem, wsScores, scoresCastMem, wsScoresCast):
  // 1. 添加两个 workspace view 作为新 inputs
  wsScArg  = addScopeInput(v0Scope, wsScores)
  wsSccArg = addScopeInput(v0Scope, wsScoresCast)

  // ---- scores 通道: 从 workspace 加载替换原始 block arg ----
  scIdx = findInputIndex(v0Scope, scoresMem)
  oldScoresArg = body.arg[scIdx]

  builder.insertAtStart(body):
    localScores = memref.alloc() {alignment = 64} : memref<64×64×f32>
    hivm.hir.load ins(wsScArg) outs(localScores)
      init_out_buffer = false, may_implicit_transpose = false

  oldScoresArg.replaceAllUsesWith(localScores)
  // → softmax 中所有使用 scores 的 op 自动转为使用 loaded buffer

  // ---- scores_cast 通道: vcast 之后写入 workspace ----
  sccIdx = findInputIndex(v0Scope, scoresCastMem)
  scoresCastArg = body.arg[sccIdx]
  vcast = findOp(body, "hivm.hir.vcast")

  builder.insertAfter(vcast):
    hivm.hir.store ins(scoresCastArg) outs(wsSccArg)
```

**效果**：V0 从 workspace 读 scores，向 workspace 写 scores_cast。

#### 3.7 通道 3: C1 — load scores_cast, fixpipe acc_o, 累加拆分

这是最复杂的变换，涉及三个子步骤：

```
doC1(c1Scope, scoresCastMem, wsScoresCast, accOMem, wsAccO, trueVal):
  wsSccArg = addScopeInput(c1Scope, wsScoresCast)
  wsAoArg  = addScopeInput(c1Scope, wsAccO)

  // ---- 子步骤 A: 将 init_condition 从 %false 改为 %true ----
  // 在 scope 的 operand 层级替换（block arg 的值由 operand 决定）
  for i in range(c1Scope.inputs.size()):
    if c1Scope.input[i] is arith.constant(false):
      c1Scope.setOperand(i, trueVal)
      break
  // → mmadL1 内部通过 block arg 看到的值变为 true
  // → 语义从 "acc_o += A×B" 变为 "fresh = A×B"（不再累加）

  // ---- 子步骤 B: 从 workspace 加载 scores_cast ----
  oldSccArg = body.arg[findInputIndex(c1Scope, scoresCastMem)]
  builder.insertAtStart(body):
    localScc = memref.alloc() {alignment = 64} : memref<64×64×f16>
    hivm.hir.load ins(wsSccArg) outs(localScc) ...
  oldSccArg.replaceAllUsesWith(localScc)

  // ---- 子步骤 C: mmadL1 输出到 fresh buffer + fixpipe ----
  mmad = findOp(body, "hivm.hir.mmadL1")
  builder.insertBefore(mmad):
    freshOut = memref.alloc() {alignment = 64} : memref<64×128×f32>
  mmad.setOperand(6, freshOut)          // 替换 'c' 操作数
  mmad.setAttr("fixpipe_already_inserted", true)
  builder.insertAfter(mmad):
    hivm.hir.fixpipe {enable_nz2nd}
      ins(freshOut) outs(wsAoArg)       // fresh → workspace acc_o
```

**acc_o 累加语义拆分的数学等价性**：

```
原始: acc_o = acc_o × correction + scores_cast × V
  (由 vmul in-place 做 correction, mmadL1 initC=false 做累加)

拆分后:
  Cube:   delta = scores_cast × V          (mmadL1 initC=true)
  Cube:   fixpipe(delta) → workspace
  V-tail: acc_o = acc_o × correction + delta  (vmul + load + vadd)
  等价于原始语义 ✓
```

#### 3.8 V-tail scope: acc_o 累加

在 C1 scope 之后、循环终结器之前，插入一个新的 VECTOR scope：

```
doVTail(c1Scope, accOMem, wsAccO):
  builder.insertAfter(c1Scope):
    vTail = CVScopeOp(
      inputs = [wsAccO, accOMem],
      tcore_type = VECTOR,
      stage_id = 1
    )

  // 在 vTail body 内构建:
  builder.insertBeforeTerminator(vTail.body):
    delta = memref.alloc() {alignment = 64} : memref<64×128×f32>
    hivm.hir.load ins(wsArg) outs(delta)
      init_out_buffer = false, may_implicit_transpose = false
    hivm.hir.vadd ins(delta, accArg) outs(accArg)
    // accArg 就是 accOMem 的 block arg，in-place 累加
```

**为什么不需要 iter_args**：acc_o 是 memref 语义，`vadd` 直接 in-place 修改
`accOMem` 指向的内存。跨迭代的 V0 vmul correction 也是 in-place 修改同一
buffer。因此不需要 `scf.for` 的 iter_args 来传递 SSA 值。

#### 3.9 Host 回调函数

在 module 级别、入口函数之前插入 host 端回调：

```
emitHostCallback(module, entryFunc):
  funcName = entryFunc.name + "_infer_workspace_shape_function"
  cbFunc = func.func @{funcName}() -> index {
    %c57344 = arith.constant 57344 : index
    return %c57344 : index
  }
  cbFunc.setAttr("hacc.function_kind", #hacc.function_kind<HOST>)
  cbFunc.setAttr("hacc.host_func_type",
    #hacc.host_func_type<infer_workspace_shape_function>)
```

#### 3.10 实现要点总结

1. **分析数据依赖**：通过 scope inputs 交集 + 类型匹配识别跨 scope 数据流
2. **分配 workspace sub-region**：在函数参数 `%workspace: memref<?xi8>` 上
   用 `affine.apply` 计算 per-block 偏移，然后用 `memref.view` 创建 typed view
3. **处理 Cube→workspace 的写入方式**：
   - mmadL1 的输出改为写到新的 **local buffer**（而非原来的共享 buffer）
   - 在 local buffer 上调用 `hivm.hir.fixpipe {enable_nz2nd}` 写到 workspace view
4. **拆分 acc_o 的累加语义**：
   - 将 C1 的 `mmadL1(initC=false, outs=acc_o)` 改为 `mmadL1(initC=true, outs=fresh_buf)`
   - fixpipe fresh_buf → workspace
   - V-tail scope：load workspace → `hivm.hir.vadd` 与已有 acc_o 累加（in-place）
5. **动态修改 CVScopeOp**：通过 `addScopeInput` + `replaceAllUsesWith` 注入
   workspace view 并替换原始共享 memref 引用，无需重建 scope
6. **生成 host 回调**：创建 `@{kernel}_infer_workspace_shape_function` 函数，
   返回 per-block workspace 大小，供 host 端 alloc

**变换示意**：

```mlir
// 输入：scope 间通过 UB memref 直接共享（CV 拆分后不可行）
scope.scope {
  hivm.hir.mmadL1 ins(...) outs(%shared_ub : memref<..., #hivm.address_space<zero>>)
  scope.return
} {tcore_type = CUBE}
scope.scope {
  hivm.hir.vmul ins(%shared_ub, ...) outs(...)  // 直接引用 Cube 的输出
  scope.return
} {tcore_type = VECTOR}

// 输出：通过 workspace (GM) 中转
// 先计算 per-block workspace 基地址
%block_idx = hivm.hir.get_block_idx -> i64
%idx = arith.index_cast %block_idx : i64 to index
%base = affine.apply affine_map<()[s0] -> (s0 * 57344)>()[%idx]
%ws_view = memref.view %workspace[%base][] : memref<?xi8> to memref<64x64xf32>

scope.scope {
  %local = memref.alloc() : memref<64x64xf32>           // 新建 local buffer
  hivm.hir.mmadL1 ins(...) outs(%local : memref<64x64xf32>)
  hivm.hir.fixpipe {enable_nz2nd}                        // L0C → workspace (GM)
    ins(%local : memref<64x64xf32>)
    outs(%ws_view : memref<64x64xf32>)
  scope.return
} {tcore_type = CUBE}

scope.scope {
  %ub_buf = memref.alloc() : memref<64x64xf32>
  hivm.hir.load ins(%ws_view) outs(%ub_buf)              // workspace → UB
  hivm.hir.vmul ins(%ub_buf, ...) outs(...)
  scope.return
} {tcore_type = VECTOR}
```

**替代的 bishengir pass**：
`InsertWorkSpaceForMixCV` + `BindWorkSpaceArg` + `PlanMemory(GLOBAL_WORKSPACE_PLAN)` + `InsertInferWorkSpaceSizeFunc`

---

### Pass ④ 插入同步 (`tilelangir-inject-block-sync`)

**目标**：
1. 在 Cube scope 和 Vector scope 的边界处插入**核间同步** `SyncBlockSet` / `SyncBlockWait`
2. 在 AIV 函数中插入**核内同步** `pipe_barrier[<PIPE_ALL>]`

**硬件动机**：
- **核间**：Cube Unit 将数据写入 workspace (GM) 后，Vector Unit 才能读取。
  硬件不会自动保证一致性，需要显式的 block sync 指令，由 Flag ID 标识。
- **核内**：AIV 的 MTE2（DMA GM→UB）、VECTOR（计算）、MTE3（DMA UB→GM）
  三条流水线完全独立、异步执行。bishengir 的 `InjectSync` NORMAL 模式因 op 白名单
  限制无法处理我们生成的 IR（含 `memref.alloc`/`memref.copy`），
  因此必须在本 pass 中手动插入 `pipe_barrier[<PIPE_ALL>]`。

**关键设计点（来自参考输出验证）：双向同步协议**

每条 workspace 数据通道需要 **两个方向** 的信号：
- **"数据就绪"**（producer→consumer）：producer 写完后通知 consumer 可以读
- **"已读完可覆写"**（consumer→producer）：consumer 读完后通知 producer 可以覆写

单向信号不够——如果 producer 在 consumer 还没读完上一轮数据时就覆写 workspace，
会导致数据损坏。因此每条通道需要 2 个 Flag ID，或者通过时序不重叠来复用 Flag ID。

**Flash Attention 的 Flag ID 分配（4 个）**：

| Flag ID | 语义 | 方向 |
|---------|------|------|
| 0 | "Vector 已读完 scores" | AIV → AIC（Cube 可覆写 scores workspace） |
| 1 | "数据已就绪"（复用） | 复用于 scores/scores_cast/acc_o 的就绪信号（时序不重叠） |
| 2 | "Cube 已读完 scores_cast" | AIC → AIV（Vector 可覆写 scores_cast workspace） |
| 3 | "Vector 已读完 acc_o" | AIV → AIC（Cube 可覆写 acc_o workspace） |

**Flag 1 复用的安全性**：scores 就绪、scores_cast 就绪、acc_o 就绪三个信号
在时序上严格顺序发生（C0→V0→C1），不会同时 pending，因此可以共享同一个 Flag ID。

**AIC 循环体内的时序**：
```
sync_block_wait  flag=0          // 等 Vector 读完上一轮 scores
fixpipe scores → workspace       // 写 scores
sync_block_set  flag=1           // 通知 Vector: scores 就绪
sync_block_wait  flag=1          // 等 Vector 写好 scores_cast
load scores_cast ← workspace    // 读 scores_cast
sync_block_set  flag=2           // 通知 Vector: 已读完 scores_cast
load V, mmadL1                   // 计算
sync_block_wait  flag=3          // 等 Vector 读完上一轮 acc_o
fixpipe acc_o → workspace        // 写 acc_o
sync_block_set  flag=1           // 通知 Vector: acc_o 就绪
```

**AIV 循环体内的时序**：
```
sync_block_wait  flag=1          // 等 Cube: scores 就绪
load scores ← workspace         // 读 scores
sync_block_set  flag=0           // 通知 Cube: 已读完 scores
softmax → scores_cast            // 计算
sync_block_wait  flag=2          // 等 Cube: 已读完上一轮 scores_cast
store scores_cast → workspace   // 写 scores_cast（仅 VID=0）
sync_block_set  flag=1           // 通知 Cube: scores_cast 就绪
sync_block_wait  flag=1          // 等 Cube: acc_o 就绪
load acc_o ← workspace          // 读 acc_o
sync_block_set  flag=3           // 通知 Cube: 已读完 acc_o
vadd 累加 acc_o                  // 计算
```

**初始信号**：循环开始前需要发送初始 "已读完" 信号，让第一轮 producer 知道可以写入：
- AIC 循环前：`sync_block_set flag=2`（"Cube 已读完 scores_cast"，初始空）
- AIV 循环前：`sync_block_set flag=0`（"Vector 已读完 scores"，初始空）
- AIV 循环前：`sync_block_set flag=3`（"Vector 已读完 acc_o"，初始空）

**实现要点**：

1. **分析 workspace 数据流图**：为每条 workspace 通道建立 producer/consumer 关系
2. **分配 Flag ID**：
   - 每条通道至少需要 1 个 "就绪" flag + 1 个 "已读完" flag
   - 时序不重叠的 "就绪" 信号可以复用同一个 Flag ID
   - 硬件限制 16 个 Flag ID，需要分配算法避免冲突
3. **插入核间 sync 指令**：
   - producer scope 尾部：先写 workspace，再 `SyncBlockSet`（就绪）
   - consumer scope 头部：先 `SyncBlockWait`（就绪），再读 workspace
   - consumer scope 读完后：`SyncBlockSet`（已读完）
   - producer scope 写入前：`SyncBlockWait`（已读完）
4. **插入初始信号**：在循环开始前为每个 consumer 插入初始 "已读完" SyncBlockSet
5. **指令格式**：`hivm.hir.sync_block_set[<CUBE/VECTOR>, <PIPE_xxx>, <PIPE_S>] flag = N`
   - pipe 参数取决于前一个 op 的 pipe 类型（MTE2/MTE3/FIX/V/M 等）
6. **插入 AIV 核内 `pipe_barrier[<PIPE_ALL>]`**（vadd 穿刺验证的关键发现）：
   - `sync_block_wait` 之前：确保前一次迭代的 MTE3 写回已完成
   - `memref.copy` (GM→UB) 之后、VECTOR 计算之前：确保 MTE2 搬运完成
   - `memref.copy` (UB→GM) 之后、`sync_block_set` 之前：确保 MTE3 搬运完成
   - 尾部循环的 `sync_block_wait` 之前：排空残留操作
   - `return` 之前：确保所有操作完成
   - **AIC 不需要 pipe_barrier**：`nd2nz → mmadL1 → fixpipe` 由硬件隐式缓冲区依赖保证

#### 4.1 穿刺实现算法

穿刺方案针对 Flash Attention 的 C0→V0→C1→V-tail 模式硬编码同步插入：

```
runOnOperation():
  func = getOperation()
  forOp = findFirst(scf.for, func)

  // 收集 pre-loop / loop-body / post-loop 的 cv_scope
  preScopes  = [sc for sc in func.body before forOp]
  loopScopes = [sc for sc in forOp.body]   // 期望 4 个: C0, V0, C1, V-tail
  postScopes = [sc for sc in func.body after forOp]

  // 1. 初始信号
  for sc in preScopes:
    if isCube(sc):
      insertBefore(terminator): set[CUBE, PIPE_MTE2, PIPE_S] flag=2
    else:
      insertBefore(terminator): set[VECTOR, PIPE_MTE2, PIPE_S] flag=0
      insertBefore(terminator): set[VECTOR, PIPE_MTE2, PIPE_S] flag=3

  // 2. C0 scope: fixpipe scores
  fixpipe = findOp(C0.body, "hivm.hir.fixpipe")
  insertBefore(fixpipe):  wait[CUBE, PIPE_MTE2, PIPE_S] flag=0
  insertAfter(fixpipe):   set[CUBE, PIPE_FIX, PIPE_S]   flag=1

  // 3. V0 scope: load scores, store scores_cast
  load = findOp(V0.body, "hivm.hir.load")      // first load = scores
  insertBefore(load):  wait[VECTOR, PIPE_FIX, PIPE_S]   flag=1
  insertAfter(load):   set[VECTOR, PIPE_MTE2, PIPE_S]   flag=0

  store = findOp(V0.body, "hivm.hir.store")
  insertBefore(store): wait[VECTOR, PIPE_MTE2, PIPE_S]  flag=2
  insertAfter(store):  set[VECTOR, PIPE_MTE3, PIPE_S]   flag=1

  // 4. C1 scope: load scores_cast, fixpipe acc_o
  load = findOp(C1.body, "hivm.hir.load")
  insertBefore(load):  wait[CUBE, PIPE_MTE3, PIPE_S]    flag=1
  insertAfter(load):   set[CUBE, PIPE_MTE2, PIPE_S]     flag=2

  fixpipe = findOp(C1.body, "hivm.hir.fixpipe")
  insertBefore(fixpipe): wait[CUBE, PIPE_MTE2, PIPE_S]  flag=3
  insertAfter(fixpipe):  set[CUBE, PIPE_FIX, PIPE_S]    flag=1

  // 5. V-tail scope: load acc_o
  load = findOp(VTail.body, "hivm.hir.load")
  insertBefore(load):  wait[VECTOR, PIPE_FIX, PIPE_S]   flag=1
  insertAfter(load):   set[VECTOR, PIPE_MTE2, PIPE_S]   flag=3

  // 6. Post-loop cleanup
  insertAfter(forOp):  wait[CUBE, PIPE_MTE2, PIPE_S]    flag=0
  insertAfter(forOp):  wait[CUBE, PIPE_MTE2, PIPE_S]    flag=3
  insertAfter(lastPostScope): wait[VECTOR, PIPE_MTE2, PIPE_S] flag=2
```

#### 4.2 Sync 指令生成

使用 `OperationState` 动态创建 `hivm.hir.sync_block_set/wait`：

```
emitSyncSet(builder, loc, coreType, tpipe, flagId):
  state = OperationState(loc, "hivm.hir.sync_block_set")
  state.addAttribute("tcore_type", TCoreTypeAttr(coreType))
  state.addAttribute("tpipe", PipeAttr(tpipe))
  state.addAttribute("pipe", PipeAttr(PIPE_S))           // 始终是 PIPE_S
  state.addAttribute("static_flag_id", I64IntegerAttr(flagId))
  state.addAttribute("operandSegmentSizes", [0, 0])      // 无 dynamic_flag_id/ffts
  builder.create(state)
```

`tpipe` 参数的选择规则（决定 sync 的 pipe 亲和性）：
- fixpipe 后面: `PIPE_FIX`
- `hivm.hir.load` (MTE2 DMA) 后面: `PIPE_MTE2`
- `hivm.hir.store` (MTE3 DMA) 后面: `PIPE_MTE3`
- producer 的 wait 用 consumer 之前那个 set 的 pipe

**替代的 bishengir pass**：`MarkRealCoreType` + `InjectBlockSync`

**注意**：核内同步（`InjectSync` / `GraphSyncSolver`，处理 pipe 间的 `set_flag`/`wait_flag`）
仍由 `hivmPostBufferizationOptimizationPipeline` 执行，不在此 pass 范围内。

---

### Pass ⑤ 插入 VID (`tilelangir-insert-vid`)

**目标**：处理 1 Cube + 2 Vector Sub-Block 架构中的 VID（Virtual ID / Sub-Block ID）。

**硬件动机**：每个 AI Core 有 1 个 Cube Unit 和 2 个 Vector Sub-Block。
两个 Sub-Block 执行相同的 kernel 代码，通过 VID（0 或 1）区分身份。

**关键设计点（来自参考输出验证）：写入守卫模式**

参考输出表明 VID 的实际用法是 **写入守卫（write guard）**，而非数据划分：
- **两个 Vector Sub-Block 冗余计算相同的 softmax**（完全相同的输入、输出）
- **仅 VID=0 执行写回**：写 workspace 和写最终输出时用 `scf.if (VID==0)` 守卫

这种设计的合理性：
- Flash Attention 的 softmax 包含 reduce_max、reduce_sum 等行级归约操作
- 如果做数据划分（每个 Sub-Block 算一半行），reduce 结果需要跨 Sub-Block merge
- 跨 Sub-Block 的 merge 需要额外的同步和通信，复杂度高
- 冗余计算虽然浪费一倍 Vector 算力，但避免了 merge 同步的复杂度和延迟
- 对于 softmax 这种计算密集但数据量小的操作，冗余计算的代价可接受

**两种模式**（根据算法特征选择）：

| 模式 | 适用场景 | 实现 |
|------|---------|------|
| **写入守卫** | 含 reduce 操作、难以按维度切分 | 冗余计算 + VID=0 写入 |
| **数据划分** | 纯 element-wise 操作、可按行均分 | VID 切分输入输出范围 |

穿刺阶段先实现写入守卫模式（更简单，与参考输出一致）。

**实现要点**：

1. 在 Vector scope 头部插入 VID 获取：
   `%vid = hivm.hir.get_sub_block_idx -> i64`
2. 计算守卫条件：`%is_vid0 = arith.cmpi eq, %vid, %c0 : index`
3. 用 `scf.if %is_vid0` 包裹所有 **写入 workspace** 和 **写回 GM** 的操作
4. Vector 的计算操作不加守卫——两个 Sub-Block 都执行
5. 写 GM 的 store 加 `{limit_sub_block_id0}` 属性标记

**变换示意**：

```mlir
scope.scope {
  %vid = hivm.hir.get_sub_block_idx -> i64
  %vid_idx = arith.index_cast %vid : i64 to index
  %is_vid0 = arith.cmpi eq, %vid_idx, %c0 : index

  // ---- 读 workspace：所有 Sub-Block 都读 ----
  hivm.hir.load ins(%ws_scores) outs(%ub_scores)

  // ---- 计算：所有 Sub-Block 都算（冗余计算）----
  hivm.hir.vmul ins(%ub_scores, %scales) outs(%ub_scores)
  hivm.hir.vreduce <max> ...
  // ... 完整的 softmax ...
  hivm.hir.vcast ins(%scores_f32) outs(%scores_f16)

  // ---- 写 workspace：仅 VID=0 ----
  scf.if %is_vid0 {
    hivm.hir.store ins(%scores_f16) outs(%ws_scores_cast)
  }

  scope.return
} {tcore_type = VECTOR, stage_id = 0}

// 循环后写最终输出
scope.scope {
  // ... vdiv, vcast ...
  scf.if %is_vid0 {
    hivm.hir.store ins(%output_f16) outs(%gm_output)
  } {limit_sub_block_id0}
  scope.return
} {tcore_type = VECTOR, stage_id = 1}
```

#### 5.1 穿刺实现算法

穿刺方案对所有 Vector scope 执行写入守卫包裹：

```
runOnOperation():
  func = getOperation()
  vectorScopes = [sc for sc in func.walk<CVScopeOp>()
                  if sc.tcore_type == VECTOR]

  for sc in vectorScopes:
    processVectorScope(sc)

processVectorScope(sc):
  body = sc.body.front()

  // 1. 收集所有写操作
  writes = [op for op in body
            if op.name in {"hivm.hir.store", "memref.copy"}]

  if writes is empty: return

  // 2. 在 scope body 开头插入 VID 检查
  insertAtStart(body):
    %vid     = hivm.hir.get_sub_block_idx -> i64
    %vid_idx = arith.index_cast %vid : i64 to index
    %c0      = arith.constant 0 : index
    %is_vid0 = arith.cmpi eq, %vid_idx, %c0 : index

  // 3. 用 scf.if 包裹每个写操作
  for writeOp in writes:
    isPostLoop = (writeOp.parent not in scf.for)
    isGMWrite  = (writeOp.name == "memref.copy")

    ifOp = scf.if(%is_vid0):
      <move writeOp into ifOp.then block>

    if isPostLoop and isGMWrite:
      ifOp.setAttr("limit_sub_block_id0", UnitAttr)
```

**关键设计决策**：

1. **VID 在每个 scope 内独立计算**：穿刺方案中 `get_sub_block_idx` 在每个
   Vector scope 开头重复调用。冗余但简单，outline 后每个函数恰好获得一份。
2. **只包裹写操作**：读操作（`hivm.hir.load`）和计算操作不加守卫，
   两个 Sub-Block 冗余执行完整的 softmax。
3. **`limit_sub_block_id0` 属性**：用于标记 GM 写出的 VID 守卫，
   bishengir 后续 pass 可能依赖此标记。

**替代的 bishengir pass**：`TileAndBindSubBlock` + `AutoBlockifyParallelLoop`

---

### Pass ⑥ CV 拆分 (`tilelangir-outline-scope`)

**目标**：将 MIX 函数中的 `tilelangir.cv_scope` 块拆分为独立的 AIC 和 AIV 函数，
完成真正的 CV 分核。

**为什么单独一步**：
①~⑤ 都在 `cv_scope` 层面操作，需要看到完整的函数上下文（循环结构、scope 间依赖等）。
如果在 ① 就直接拆分为独立函数，后续 pass 将无法获取这些信息。
因此标注（①）和拆分（⑥）分开执行。

**变换**：

```mlir
// 输入：一个 MIX 函数，包含多个 cv_scope
func.func @kernel(...) attributes {hivm.func_core_type = MIX} {
  tilelangir.cv_scope(...) {tcore_type = 1}  { /* Cube ops */ }
  tilelangir.cv_scope(...) {tcore_type = 2}  { /* Vector ops */ }
  scf.for ... {
    tilelangir.cv_scope(...) {tcore_type = 1}  { /* Cube ops */ }
    tilelangir.cv_scope(...) {tcore_type = 2}  { /* Vector ops */ }
  }
}

// 输出：两个独立函数，scope 被消除
func.func @kernel_mix_aic(...) attributes {
    hivm.func_core_type = AIC, hivm.part_of_mix} {
  /* Cube ops (展平后的) */
  scf.for ... { /* 只含 Cube ops */ }
}
func.func @kernel_mix_aiv(...) attributes {
    hivm.func_core_type = AIV, hivm.part_of_mix} {
  /* Vector ops (展平后的) */
  scf.for ... { /* 只含 Vector ops */ }
}
```

**替代的 bishengir pass**：`SplitMixKernel` + `InlineScopePass`

#### 6.1 穿刺实现算法

> ⚠️ **穿刺（Spike）方案**：与 Stage 1~5 一样，采用 spike 实现。

**核心策略**：Clone + Filter + Flatten（克隆 + 过滤 + 展平）

1. **识别 MIX 函数**：遍历 ModuleOp 中的 `func::FuncOp`，找到 `hivm.func_core_type = MIX` 的函数。

2. **克隆为 AIC 和 AIV**：
   - 使用 `FuncOp::clone(IRMapping &)` 分别克隆出 AIC 和 AIV 副本
   - AIC 副本命名为 `@{name}_mix_aic`，设置 `hivm.func_core_type = AIC`
   - AIV 副本命名为 `@{name}_mix_aiv`，设置 `hivm.func_core_type = AIV`
   - 两个副本均添加 `hivm.part_of_mix` 属性（UnitAttr）

3. **过滤（processClone）**，对每个副本执行：

   **Step 3a: 删除对立类型的 cv_scope**
   - AIC 副本：遍历所有 `CVScopeOp`，删除 `tcore_type = VECTOR(2)` 的 scope
   - AIV 副本：遍历所有 `CVScopeOp`，删除 `tcore_type = CUBE(1)` 的 scope
   - 安全性：cv_scope 无返回值（memref 语义），删除不破坏 SSA

   **Step 3b: 展平保留的 cv_scope（flattenScope）**
   - 将 scope body 中的 block arguments 替换为对应的 scope inputs
   - 将 body 中的所有 ops（除 terminator）移动到 scope 所在的 parent block 中
   - 删除 scope op（包括其 cv_yield terminator）
   ```
   flattenScope(CVScopeOp):
     body = scope.getBody().front()
     for (arg, input) in zip(body.args, scope.inputs):
       arg.replaceAllUsesWith(input)
     while body.front != terminator:
       body.front.moveBefore(scope)
     scope.erase()
   ```

   **Step 3c: 删除对立类型的 sync_block ops**
   - 使用 `dyn_cast<hivm::SyncBlockSetOp>` / `SyncBlockWaitOp` 识别 sync 指令
   - 通过 `getTcoreType().getTcoretype()` 读取核心类型
   - AIC 副本：删除 `tcore_type = VECTOR` 的 sync ops
   - AIV 副本：删除 `tcore_type = CUBE` 的 sync ops

4. **删除原始 MIX 函数**：原函数被两个副本替代。

**效果**：
- AIC 函数只包含 Cube 运算（load, mmadL1, fixpipe）和 Cube sync 指令
- AIV 函数只包含 Vector 运算（softmax 链, vadd）、VID 守卫和 Vector sync 指令
- 中性 ops（arith.constant, memref.alloc, memref.subview, get_block_idx 等）在两个副本中均保留，后续 DCE 可清理未使用部分

**Pass 执行顺序注意**：`tilelangir-infer-task-type` 必须在 `tilelangir-outline-scope` **之前**运行，
因为 task_type 回调的函数名需要基于原始 MIX 函数名生成。

---

### Pass ⑦ 推断 Task Type (`tilelangir-infer-task-type`)

**目标**：根据入口函数的核心类型生成 host 端回调函数，
告知 runtime 以何种模式 launch kernel。

**当前设计的缺陷**：TileLang 目前通过正则表达式从 MLIR 文本中解析
`#hivm.module_core_type<MIX>`，然后在 Python 层决定 launch 参数。
这种方式脆弱且不规范。bishengir 的 `InsertInferTaskTypeFunc` 设计更合理——
通过编译器 pass 生成标准的 host 回调函数，runtime 直接调用。

**Task Type 枚举**：

| 核心类型 | TaskType 值 | 含义 |
|----------|------------|------|
| AIV      | 10         | 纯 Vector kernel |
| AIC      | 20         | 纯 Cube kernel |
| MIX 1:1  | 31         | Cube+Vector 混合，1 Cube : 1 Vector |
| MIX 1:2  | 32         | Cube+Vector 混合，1 Cube : 2 Vector（默认） |

**变换**：

```mlir
// 输入：入口函数带有 func_core_type 属性
func.func @kernel(...) attributes {
  hivm.entry,
  hivm.func_core_type = #hivm.func_core_type<MIX>
} { ... }

// 输出：新增一个 host 端回调函数
func.func @kernel_infer_task_type_function() -> i8
  attributes {hacc.host, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
  %c = arith.constant 32 : i8
  return %c : i8
}
```

#### 7.1 穿刺实现算法

```
runOnOperation():
  module = getOperation()
  entry = findFirst(func.func with "hacc.entry" or "hivm.entry", module)
  if not entry: return

  // 读取 func_core_type → 映射为 task type 常量
  taskType = 32  // 默认 MIX 1:2
  if attr = entry.getAttr("hivm.func_core_type"):
    ftAttr = attr.dyn_cast<TFuncCoreTypeAttr>()
    switch ftAttr.getFuncCoreType():
      case AIC: taskType = 20
      case AIV: taskType = 10
      case MIX: taskType = 32

  // 生成 host 回调函数
  name = entry.name + "_infer_task_type_function"
  if module.lookupSymbol(name): return  // 已存在

  insertBefore(entry):
    func.func @{name}() -> i8
      attributes {
        hacc.function_kind = HOST,
        hacc.host_func_type = infer_task_type_function
      } {
      %c = arith.constant {taskType} : i8
      return %c : i8
    }
```

**实现要点**：
- 遍历 module 找到 `hivm.entry` / `hacc.entry` 入口函数
- 读取 `hivm.func_core_type` 属性映射为 TaskType 常量
- 生成 host 回调函数
- 后续需要重构 `jit_npu.py` 中的 runtime launch 逻辑，
  从正则解析切换到调用此回调

**替代的 bishengir pass**：`InsertInferTaskTypeFunc`

---

## 五、Pass 与 bishengir 原始 Pass 对应关系

| # | TileLang Pass | 替代的 bishengir Pass | 说明 |
|---|---------------|----------------------|------|
| ① | `cv_annotate` | `InferFuncCoreType` + `InsertLoadStoreForMixCV` | cv_scope 分组（带 block args），支持多 C/V stage |
| ② | `simple_multibuffer` | `MarkMultiBuffer` + `CVPipelining` + `TileCubeVectorLoop` | loop tiling + distribution，多缓冲数 = tile_factor |
| ③ | `insert_workspace` | `InsertWorkSpaceForMixCV` + `BindWorkSpaceArg` + `PlanMemory` + `InsertInferWorkSpaceSizeFunc` | 合并为一个 pass |
| ④ | `inject_block_sync` | `MarkRealCoreType` + `InjectBlockSync` | 基于 scope batch 边界做 Flag ID 分配 |
| ⑤ | `insert_vid` | `TileAndBindSubBlock` + `AutoBlockifyParallelLoop` | 1C+2V Sub-Block 划分 |
| ⑥ | `outline_scope` | `SplitMixKernel` | 复用 bishengir |
| ⑦ | `infer_task_type` | `InsertInferTaskTypeFunc` | host 端 task type 回调 |

**不再需要的 bishengir Pass**（~30 个）：

| 分类 | Pass | 不需要的原因 |
|------|------|-------------|
| Bufferization | `OneShotBufferize` 等整条线 | codegen 统一 memref，无需 tensor→memref |
| Tensor 优化 | `PropagateReshape`, `FoldTensorEmpty`, `CloneTensorEmpty` | memref IR 中不存在 tensor ops |
| Cube 规范化 | `NormalizeMatmul`, `TileBatchMMIntoLoop` | codegen 已生成规范化的 mmadL1 |
| Fixpipe | `InlineFixpipe` | codegen 已直接生成 fixpipe ops |
| OTF | `InlineOTFBroadcast`, `InlineOTFLoadStore` | 基于 tensor SSA，memref 下不适用 |
| 内存空间 | `InferHIVMMemScope` | codegen 已通过 GetHIVMAddressSpace 标注 |
| 通用优化 | `LICM`, `CSE`, `RemoveRedundantLoopInit` | 作为独立 pass 在 pipeline 尾部追加 |

---

## 六、Passes.td 定义

Pass 定义位于 `tilelangir/include/tilelangir/Transforms/Passes.td`：

```tablegen
def TileLangIRCVAnnotate : Pass<"tilelangir-cv-annotate", "::mlir::ModuleOp"> {
  let summary = "Annotate Cube/Vector operations into scope blocks";
  let description = [{
    Group consecutive Cube or Vector HIVM operations into tilelangir.cv_scope
    blocks with explicit block-arg inputs. Annotated with tcore_type and
    stage_id attributes. Handles multiple C/V alternations within a single
    loop iteration. Inter-scope dependencies are later analyzed via shared
    inputs + read/write effect analysis.
  }];
  let dependentDialects = ["::mlir::tilelangir::TileLangIRDialect"];
}

def TileLangIRSimpleMultiBuffer : Pass<"tilelangir-simple-multibuffer", "::mlir::func::FuncOp"> {
  let summary = "Tile loop and distribute scopes for multi-buffering";
  let description = [{
    For loops containing tilelangir.cv_scope blocks, perform loop tiling and
    scope distribution: the outer loop iterates over tile groups, and
    each scope gets its own inner loop over the tile. Buffer allocations
    within the loop are replicated by tile_factor for multi-buffering.
    This is a simplified alternative to full software pipelining.
  }];
  let dependentDialects = [
    "scope::ScopeDialect", "hivm::HIVMDialect", "scf::SCFDialect"
  ];
  let options = [
    Option<"tileFactor", "tile-factor", "unsigned",
           /*default=*/"2", "Tiling factor (also determines multi-buffer count)">
  ];
}

def TileLangIRInsertWorkspace : Pass<"tilelangir-insert-workspace", "::mlir::ModuleOp"> {
  let summary = "Insert workspace buffers for Cube-Vector data exchange";
  let description = [{
    Allocate workspace sub-regions on GM for data transfer between Cube and
    Vector scopes. Plan memory layout with reuse and alignment, and generate
    a host-side infer_workspace_size callback function.
  }];
  let dependentDialects = [
    "hivm::HIVMDialect", "memref::MemRefDialect"
  ];
}

def TileLangIRInjectBlockSync : Pass<"tilelangir-inject-block-sync", "::mlir::func::FuncOp"> {
  let summary = "Insert inter-core synchronization at scope boundaries";
  let description = [{
    Inject SyncBlockSet/SyncBlockWait operations where Cube scopes write to
    workspace and Vector scopes read from it (and vice versa).
    Allocates Flag IDs and handles FFTS base addresses.
  }];
  let dependentDialects = ["hivm::HIVMDialect", "hacc::HACCDialect"];
}

def TileLangIRInsertVID : Pass<"tilelangir-insert-vid", "::mlir::func::FuncOp"> {
  let summary = "Insert VID-based write guards for Vector sub-blocks";
  let description = [{
    Insert Sub-Block VID (get_sub_block_idx) in Vector scopes and wrap
    workspace/GM write operations with scf.if(VID==0) guards. Both
    Sub-Blocks perform redundant computation, but only VID=0 writes back.
    This avoids complex cross-sub-block synchronization for reduce operations.
  }];
  let dependentDialects = ["hivm::HIVMDialect", "scf::SCFDialect"];
}

def TileLangIRInferTaskType : Pass<"tilelangir-infer-task-type", "::mlir::ModuleOp"> {
  let summary = "Generate host-side task type inference callback";
  let description = [{
    Read the entry function's core type attribute and generate a host-side
    callback function that returns the TaskType constant for runtime kernel
    launch configuration.
  }];
  let dependentDialects = [
    "func::FuncDialect", "arith::ArithDialect", "hacc::HACCDialect"
  ];
}
```

---

## 七、风险与注意事项

1. **Pass 顺序不可调换**：① 的 scope 标注是后续所有 pass 的前提；
   ② simple_multibuffer 会改变循环结构，必须在 ③④ 之前；
   ⑥ outline-scope 必须在 ①~⑤ 全部完成后。

2. **核内同步需自行处理**：bishengir 的 `InjectSync` NORMAL 模式要求 IR 中的 op
   全部在其 `IRTranslator` 白名单内（`PointerCastOp`、`DestinationStyleOpInterface` 等）。
   我们生成的 AIV IR 中包含 `memref.alloc`（带 strided layout）、`memref.copy` 等 op，
   会导致 NORMAL 模式分析失败。**解决方案**：
   - **AIV 函数**：在 `inject_block_sync` pass 中同时插入核内 `pipe_barrier[<PIPE_ALL>]`。
     具体规则见 `experience.md` 第 5.2 节。
   - **AIC 函数**：AIC 的 `nd2nz → mmadL1 → fixpipe` 链路由硬件隐式缓冲区依赖保证顺序，
     不需要核内 barrier。`sync_block_set` 绑定的 `<PIPE_FIX>` 已保证信号在 fixpipe 完成后才发出。
   - 后续可优化为只插必要的 `pipe_barrier[<PIPE_V>]` / `pipe_barrier[<PIPE_MTE2>]` 等细粒度 barrier，
     或者在 outline 前将 `memref.alloc`/`memref.copy` lower 为 `hivm.hir.pointer_cast`/`hivm.hir.load`/`hivm.hir.store`，使 NORMAL 模式能工作。

3. **Flag ID 资源有限 + 双向协议**：硬件 16 个 Flag ID（核间），
   双向协议下每条 workspace 通道需要 2 个 Flag 方向（就绪 + 已读完）。
   Flash Attention 的 3 条通道需要 4 个 Flag ID（就绪信号复用 flag 1）。
   更复杂的算法可能需要更多 Flag ID，需要生命周期分析和复用算法。

4. **simple_multibuffer 的 tile_factor 需整除 trip count**：
   原循环迭代数必须能被 tile_factor 整除，否则需要处理 remainder loop。
   穿刺阶段可先假设整除，后续完善边界处理。

5. **InjectSync 白名单约束**：bishengir 的 `InjectSync` NORMAL 模式的 `IRTranslator`
   只认识 `PointerCastOp`、`DestinationStyleOpInterface`、`memref::LoadOp/StoreOp` 等 op。
   我们 IR 中的 `memref.alloc`（带 strided layout）和 `memref.copy` 不在白名单中。
   有两条解决路径：
   - **路径 A（当前选择）**：手动在 AIV 中插入 `pipe_barrier[<PIPE_ALL>]`，
     禁用或绕过 InjectSync。正确但性能粗粒度。
   - **路径 B（后续优化）**：在 outline 前将 `memref.alloc` lower 为 `hivm.hir.pointer_cast`、
     将 `memref.copy` lower 为 `hivm.hir.load`/`hivm.hir.store`，使 IRTranslator 能识别，
     让 NORMAL 模式自动生成精细的 `set_flag`/`wait_flag`。

6. **jit_npu.py 的 runtime 重构**：⑦ 生成 task type 回调后，
   需要配套重构 `_parse_npuir_metadata()` 中的正则解析逻辑，
   改为调用编译器生成的 host 函数。

7. **acc_o 累加语义拆分**：原 IR 中 mmadL1(initC=false) 直接累加到 acc_o，
   CV 拆分后必须改为 Cube 写 fresh 结果 + Vector vadd 累加。
   ③ insert_workspace 需要识别 `initC=false` 的 mmadL1 并做这一语义变换。

8. **fixpipe 的 pipe 属性**：Cube→workspace 使用 fixpipe 而非 memref.copy，
   后续 ④ inject_block_sync 的 sync 指令 pipe 参数需要用 `PIPE_FIX`（而非 `PIPE_MTE3`）。

9. **VID 写入守卫模式下的冗余计算**：两个 Vector Sub-Block 执行相同的 softmax，
   对计算密集型场景是合理的（避免 reduce merge 的复杂度），
   但对 element-wise 密集型算法可能需要演进到数据划分模式。

10. **workspace 是 per-block 的**：host 回调返回 per-block 大小，
    runtime 负责 `per_block_size × num_blocks` 的总分配。
    需确保 jit_npu.py 正确处理此乘法关系。

---

## 附录：Flash Attention 实例的逐 Pass 期望输出

> **注**：以下附录中的示例 MLIR 使用 `scope.scope` / `scope.return` 伪代码表示 scope 结构。
> 实际实现中使用 `tilelangir.cv_scope`（带显式 block args 输入）和 `tilelangir.cv_yield`
> 作为终结器。Pass ① A.1 节的示例已在主文档中更新为实际输出格式。

### A.0 输入 IR 分析

输入文件：`example_input.mlir`，由 `testing/compile/flash_attn_npuir_dev.py` 生成。

**Op 分类（Cube / Vector / DMA / Scalar）：**

| 行 | Op | 类型 | 说明 |
|----|----|------|------|
| 31 | `hivm.hir.nd2nz` GM→cbuf | **DMA(Cube侧)** | 加载 Q 到 L1 |
| 32-35 | `hivm.hir.vbrc` ×4 | **Vector** | 初始化 acc_o/acc_l/acc_m/scales |
| 50 | `hivm.hir.nd2nz` GM→cbuf | **DMA(Cube侧)** | 加载 K tile 到 L1 |
| 51 | `hivm.hir.mmadL1` | **Cube** | Q × K^T → scores |
| 52 | `hivm.hir.vmul` | **Vector** | scores × scale |
| 53 | `hivm.hir.vreduce <max>` | **Vector** | row max |
| 54 | `hivm.hir.vmax` | **Vector** | running max |
| 55 | `hivm.hir.vsub` | **Vector** | old_max - new_max |
| 56 | `hivm.hir.vexp` | **Vector** | correction factor |
| 57 | `hivm.hir.vsub` (broadcast) | **Vector** | scores - new_max |
| 58 | `hivm.hir.vexp` | **Vector** | softmax exp |
| 59 | `hivm.hir.vreduce <sum>` | **Vector** | row sum |
| 60 | `hivm.hir.vmul` | **Vector** | correct acc_l |
| 61 | `hivm.hir.vadd` | **Vector** | update acc_l |
| 62 | `hivm.hir.vmul` (broadcast) | **Vector** | correct acc_o |
| 63 | `hivm.hir.vcast` | **Vector** | f32→f16 scores_cast |
| 64 | `hivm.hir.vbrc` | **Vector** | clear tmp |
| 65 | `hivm.hir.vadd` | **Vector** | update acc_m |
| 66-67 | `hivm.hir.nd2nz` GM→cbuf | **DMA(Cube侧)** | 加载 V tile 到 L1 |
| 68 | `hivm.hir.mmadL1` | **Cube** | scores_cast × V → acc_o |
| 70 | `hivm.hir.vdiv` | **Vector** | normalize |
| 71 | `hivm.hir.vcast` | **Vector** | f32→f16 |
| 73 | `memref.copy` cbuf→GM | **DMA(Vector侧)** | 写回 Output |

**循环体内的 C/V 模式：`C0 → V0 → C1`**

- **C0**（行 49-51）：nd2nz(K) + mmadL1(Q×K^T)
- **V0**（行 52-65）：softmax 全流程（14 个 Vector op）
- **C1**（行 66-68）：nd2nz(V) + mmadL1(scores×V)

**跨 scope 的关键数据流：**

| 数据 | 产出 scope | 消费 scope | 输入 IR 中的 buffer | 拆分后的传递方式 |
|------|-----------|-----------|-------------------|----------------|
| scores (QK结果) | C0: mmadL1 | V0: vmul | `%alloc_13` (UB) | Cube fixpipe→workspace, Vector load←workspace |
| scores_cast (fp16) | V0: vcast | C1: mmadL1 | `%alloc_14` (UB) | Vector store→workspace (VID=0), Cube load←workspace |
| acc_o (累加结果) | C1: mmadL1 | V0(下一迭代): vmul | `%alloc_7` (UB) | Cube fixpipe→workspace (fresh buf), Vector load+vadd←workspace |

> **注意**：
> - 当前 IR 中 Cube 和 Vector 通过 UB（zero）空间的 memref 直接共享数据，
>   CV 拆分后需由 ③ insert_workspace 替换为 workspace (GM) 中转。
> - acc_o 的传递涉及 **累加语义拆分**：原 IR 中 C1 的 mmadL1(initC=false) 直接累加到 acc_o，
>   拆分后 Cube 无法访问 Vector 持有的 acc_o，需改为 Cube 写 fresh 结果到 workspace，
>   Vector 加载后 vadd 完成累加。
> - scores_cast 的 workspace 写入需要 VID 守卫：仅 VID=0 执行 store。

---

### A.1 Pass ① cv_annotate 期望输出

**变换要点**：
- 循环体内按 op 类型切分为 3 个 scope：C0, V0, C1
- 循环前的 DMA（nd2nz Q）和 Vector 初始化（vbrc）也分别归入 scope
- 循环后的 Vector 收尾（vdiv/vcast/copy）归入 scope
- alloc、arith、subview 等标量/地址计算操作留在 scope 外
- 函数属性更新为 MIX

```mlir
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8>, %arg2: memref<?xi8>,
      %arg3: memref<?xf16, #hivm.address_space<gm>>,
      %arg4: memref<?xf16, #hivm.address_space<gm>>,
      %arg5: memref<?xf16, #hivm.address_space<gm>>,
      %arg6: memref<?xf16, #hivm.address_space<gm>>,
      %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>, ...} {

    // ---- 常量、reinterpret_cast、block_idx、alloc 等保持不变 ----
    %c64 = arith.constant 64 : index
    // ... (所有常量定义不变) ...
    hivm.hir.set_ffts_base_addr %arg0
    // ... (reinterpret_cast 不变) ...
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    // ... (alloc 不变) ...
    %2 = arith.muli %1, %c64_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %subview = memref.subview %reinterpret_cast[%3, 0] [64, 128] [1, 1] : ...

    // ---- 循环前：Cube scope（加载 Q）----
    scope.scope {
      hivm.hir.nd2nz {dst_continuous}
        ins(%subview : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%alloc : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>)
        init_out_buffer = false
      scope.return
    } {tcore_type = #hivm.tcore_type<CUBE>, stage_id = 0 : i32}

    // ---- 循环前：Vector scope（初始化累加器）----
    scope.scope {
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_7 : ...)    // acc_o = 0
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_6 : ...)    // acc_l = 0
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%alloc_5 : ...)  // acc_m = -inf
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%alloc_8 : ...)  // scales = 1/sqrt(d)
      scope.return
    } {tcore_type = #hivm.tcore_type<VECTOR>, stage_id = 0 : i32}

    // ---- 主循环 ----
    scf.for %arg13 = %c0_i32 to %c8_i32 step %c1_i32 : i32 {
      // alloc 保持在 scope 外
      %alloc_11 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_12 = memref.alloc() : memref<64x128xf16, strided<[128, 1]>, #hivm.address_space<cbuf>>
      %alloc_13 = memref.alloc() : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<zero>>
      %alloc_14 = memref.alloc() : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<zero>>
      // ... (其他 alloc 不变) ...
      %4 = arith.muli %arg13, %c64_i32 : i32
      %5 = arith.index_cast %4 : i32 to index

      // ---- C0: Load K + QK gemm ----
      scope.scope {
        %subview_21 = memref.subview %reinterpret_cast_3[%5, 0] [64, 128] [1, 1] : ...
        hivm.hir.nd2nz {dst_continuous}
          ins(%subview_21 : ...) outs(%alloc_11 : ...) init_out_buffer = false
        hivm.hir.mmadL1 {b_transpose}
          ins(%alloc, %alloc_11, %true, %c64, %c64, %c128 : ...)
          outs(%alloc_13 : memref<64x64xf32, strided<[64, 1]>, #hivm.address_space<zero>>)
        scope.return
      } {tcore_type = #hivm.tcore_type<CUBE>, stage_id = 0 : i32}

      // ---- V0: Softmax 全流程 ----
      scope.scope {
        hivm.hir.vmul ins(%alloc_13, %alloc_8 : ...) outs(%alloc_13 : ...)
        hivm.hir.vreduce <max> ins(%alloc_13 : ...) outs(%alloc_16 : ...) reduce_dims = [1]
        hivm.hir.vmax ins(%alloc_5, %alloc_16 : ...) outs(%alloc_20 : ...)
        hivm.hir.vsub ins(%alloc_5, %alloc_20 : ...) outs(%alloc_19 : ...)
        hivm.hir.vexp ins(%alloc_19 : ...) outs(%alloc_15 : ...)
        hivm.hir.vsub ins(%alloc_13, %alloc_20 : ...) outs(%alloc_18 : ...) broadcast = [1]
        hivm.hir.vexp ins(%alloc_18 : ...) outs(%alloc_13 : ...)
        hivm.hir.vreduce <sum> ins(%alloc_13 : ...) outs(%alloc_17 : ...) reduce_dims = [1]
        hivm.hir.vmul ins(%alloc_6, %alloc_15 : ...) outs(%alloc_6 : ...)
        hivm.hir.vadd ins(%alloc_6, %alloc_17 : ...) outs(%alloc_6 : ...)
        hivm.hir.vmul ins(%alloc_7, %alloc_15 : ...) outs(%alloc_7 : ...) broadcast = [1]
        hivm.hir.vcast ins(%alloc_13 : ...) outs(%alloc_14 : ...)
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_19 : ...)
        hivm.hir.vadd ins(%alloc_19, %alloc_20 : ...) outs(%alloc_5 : ...)
        scope.return
      } {tcore_type = #hivm.tcore_type<VECTOR>, stage_id = 0 : i32}

      // ---- C1: Load V + SV gemm ----
      scope.scope {
        %subview_22 = memref.subview %reinterpret_cast_2[%5, 0] [64, 128] [1, 1] : ...
        hivm.hir.nd2nz {dst_continuous}
          ins(%subview_22 : ...) outs(%alloc_12 : ...) init_out_buffer = false
        hivm.hir.mmadL1
          ins(%alloc_14, %alloc_12, %false, %c64, %c64, %c128 : ...)
          outs(%alloc_7 : memref<64x128xf32, strided<[128, 1]>, #hivm.address_space<zero>>)
        scope.return
      } {tcore_type = #hivm.tcore_type<CUBE>, stage_id = 1 : i32}
    }

    // ---- 循环后：Vector scope（归一化 + 写回）----
    scope.scope {
      hivm.hir.vdiv ins(%alloc_7, %alloc_6 : ...) outs(%alloc_7 : ...) broadcast = [1]
      hivm.hir.vcast ins(%alloc_7 : ...) outs(%alloc_9 : ...)
      %subview_10 = memref.subview %reinterpret_cast_4[%3, 0] [64, 128] [1, 1] : ...
      memref.copy %alloc_9, %subview_10 : ...
      scope.return
    } {tcore_type = #hivm.tcore_type<VECTOR>, stage_id = 1 : i32}

    return
  }
}
```

---

### A.2 Pass ② simple_multibuffer 期望输出

**变换要点**：
- 原循环 `scf.for 0..8` 被 tiling 为外层 `0..4`（tile_factor=2）+ 内层 `0..2`
- 每个 scope 获得独立的内层循环（loop distribution）
- 循环内的 alloc 扩展为 tile_factor 份（用 `%j` 索引）

```mlir
    // ---- 循环前 scope 不变 ----
    scope.scope { /* nd2nz Q → L1 */ }   {tcore_type = CUBE, stage_id = 0}
    scope.scope { /* vbrc init ×4 */ }    {tcore_type = VECTOR, stage_id = 0}

    // ---- 主循环：外层按 tile_factor=2 切块 ----
    scf.for %i_outer = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
      %base = arith.muli %i_outer, %c2_i32 : i32  // tile_factor=2

      // 多缓冲 alloc: 2 份 L1 buffer (K), 2 份 L1 buffer (V)
      %K_l1_0 = memref.alloc() : memref<64x128xf16, ..., #hivm.address_space<cbuf>>
      %K_l1_1 = memref.alloc() : memref<64x128xf16, ..., #hivm.address_space<cbuf>>
      %V_l1_0 = memref.alloc() : memref<64x128xf16, ..., #hivm.address_space<cbuf>>
      %V_l1_1 = memref.alloc() : memref<64x128xf16, ..., #hivm.address_space<cbuf>>
      // 多缓冲 alloc: 2 份 scores, 2 份 scores_cast
      %scores_0 = memref.alloc() : memref<64x64xf32, ..., #hivm.address_space<zero>>
      %scores_1 = memref.alloc() : memref<64x64xf32, ..., #hivm.address_space<zero>>
      %scores_cast_0 = memref.alloc() : memref<64x64xf16, ..., #hivm.address_space<zero>>
      %scores_cast_1 = memref.alloc() : memref<64x64xf16, ..., #hivm.address_space<zero>>
      // ... 其他临时 buffer 也各 2 份 ...

      // ==== C0 batch: Cube 连续跑 2 次 QK gemm ====
      scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
        %i = arith.addi %base, %j : i32
        %idx = arith.index_cast %i : i32 to index
        // 根据 j 选择 buffer 实例
        %K_l1 = scf.if (%j == 0) -> %K_l1_0 else -> %K_l1_1  // 简化表示
        %scores = scf.if (%j == 0) -> %scores_0 else -> %scores_1
        scope.scope {
          %K_sub = memref.subview %K_gm[%idx * 64, 0] [64, 128] [1, 1] : ...
          hivm.hir.nd2nz ins(%K_sub) outs(%K_l1) ...
          hivm.hir.mmadL1 {b_transpose}
            ins(%Q_l1, %K_l1, ...) outs(%scores : ...)
          scope.return
        } {tcore_type = #hivm.tcore_type<CUBE>, stage_id = 0 : i32}
      }

      // ==== V0 batch: Vector 连续跑 2 次 softmax ====
      scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
        %scores = scf.if (%j == 0) -> %scores_0 else -> %scores_1
        %scores_cast = scf.if (%j == 0) -> %scores_cast_0 else -> %scores_cast_1
        scope.scope {
          hivm.hir.vmul ins(%scores, %scales : ...) outs(%scores : ...)
          hivm.hir.vreduce <max> ins(%scores : ...) outs(%local_max : ...)
          // ... 其余 softmax ops ...
          hivm.hir.vcast ins(%scores : ...) outs(%scores_cast : ...)
          hivm.hir.vbrc ins(%cst : ...) outs(%tmp1 : ...)
          hivm.hir.vadd ins(%tmp1, %new_max : ...) outs(%acc_m : ...)
          scope.return
        } {tcore_type = #hivm.tcore_type<VECTOR>, stage_id = 0 : i32}
      }

      // ==== C1 batch: Cube 连续跑 2 次 SV gemm ====
      scf.for %j = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
        %i = arith.addi %base, %j : i32
        %idx = arith.index_cast %i : i32 to index
        %V_l1 = scf.if (%j == 0) -> %V_l1_0 else -> %V_l1_1
        %scores_cast = scf.if (%j == 0) -> %scores_cast_0 else -> %scores_cast_1
        scope.scope {
          %V_sub = memref.subview %V_gm[%idx * 64, 0] [64, 128] [1, 1] : ...
          hivm.hir.nd2nz ins(%V_sub) outs(%V_l1) ...
          hivm.hir.mmadL1
            ins(%scores_cast, %V_l1, ...) outs(%acc_o : ...)
          scope.return
        } {tcore_type = #hivm.tcore_type<CUBE>, stage_id = 1 : i32}
      }
    }

    // ---- 循环后 Vector scope 不变 ----
    scope.scope { /* vdiv, vcast, memref.copy */ } {tcore_type = VECTOR, stage_id = 1}
```

**关键特点**：
- 原循环 8 次迭代 → 外层 4 次 × 内层 2 次
- C0 batch 产出 scores[0]、scores[1]，V0 batch 按序消费
- tile_factor=2 → 每类 buffer 2 份，恰好对应双缓冲
- 跨 scope batch 的数据依赖由后续 ④ inject_block_sync 在 batch 边界插入 sync 保证

---

### A.3 Pass ③ insert_workspace 期望输出

**变换要点**：
- 当前 Cube 和 Vector scope 通过 UB（`address_space<zero>`）上的 memref 直接共享数据
- CV 拆分后 Cube 和 Vector 成为独立任务，不能假设 UB 共享
- 需要在 scope 边界处插入 workspace (GM) 中转：
  - C0 → V0：scores 通过 workspace 传递（Cube 侧用 **fixpipe** 写入）
  - V0 → C1：scores_cast 通过 workspace 传递（Vector 侧用 **store** 写入）
  - C1 → V0(next)：acc_o 通过 workspace 传递（Cube 侧用 **fixpipe** 写入）
- **Cube→workspace 使用 fixpipe**（L0C→GM 硬件直通路径），不用 memref.copy
- **acc_o 累加拆分**：C1 的 mmadL1 改为写到 fresh buffer + fixpipe 到 workspace，
  Vector 侧 load workspace + vadd 完成累加
- Workspace 是 **per-block** 的，偏移 = `block_idx × 57344`
- 规划 workspace 内存布局并生成 host 端 size 回调

**workspace 布局规划（per-block）**：

```
每个 AI Core (block) 的 workspace：
├── [0, 16384)       scores:      64×64×f32 = 16384 bytes
├── [16384, 24576)   scores_cast: 64×64×f16 = 8192 bytes
└── [24576, 57344)   acc_o:       64×128×f32 = 32768 bytes
Per-block total: 57344 bytes

对应的 affine_map:
  #map  = affine_map<()[s0] -> (s0 * 57344)>          // scores 基地址
  #map1 = affine_map<()[s0] -> (s0 * 57344 + 16384)>  // scores_cast 基地址
  #map2 = affine_map<()[s0] -> (s0 * 57344 + 24576)>  // acc_o 基地址
```

**变化部分**：

```mlir
    // ======== 新增：per-block workspace view 分配 ========
    %block_idx = hivm.hir.get_block_idx -> i64
    %idx = arith.index_cast %block_idx : i64 to index
    %ws_scores_off = affine.apply #map()[%idx]       // block_idx * 57344
    %ws_scores_cast_off = affine.apply #map1()[%idx]  // block_idx * 57344 + 16384
    %ws_acc_o_off = affine.apply #map2()[%idx]        // block_idx * 57344 + 24576
    %ws_scores_view = memref.view %arg2[%ws_scores_off][]
      : memref<?xi8> to memref<64x64xf32>
    %ws_scores_cast_view = memref.view %arg2[%ws_scores_cast_off][]
      : memref<?xi8> to memref<64x64xf16>
    %ws_acc_o_view = memref.view %arg2[%ws_acc_o_off][]
      : memref<?xi8> to memref<64x128xf32>

    scf.for ... {
      // ---- C0: mmadL1 结果通过 fixpipe 写入 workspace ----
      scope.scope {
        %K_sub = memref.subview %K_gm[...] : ...
        hivm.hir.nd2nz ins(%K_sub) outs(%K_l1) ...
        // mmadL1 写到 local buffer（而非共享的 %alloc_13）
        %scores_local = memref.alloc() : memref<64x64xf32>
        hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true}
          ins(%Q_l1, %K_l1, %true, ...) outs(%scores_local : memref<64x64xf32>)
        // fixpipe: L0C → workspace (GM)
        hivm.hir.fixpipe {enable_nz2nd}
          ins(%scores_local : memref<64x64xf32>)
          outs(%ws_scores_view : memref<64x64xf32>)
        scope.return
      } {tcore_type = CUBE, stage_id = 0}

      // ---- V0: 从 workspace 读取 scores，softmax 后写 scores_cast 到 workspace ----
      scope.scope {
        // workspace → UB
        %ub_scores = memref.alloc() : memref<64x64xf32>
        hivm.hir.load ins(%ws_scores_view) outs(%ub_scores) ...
        // softmax（操作对象改为 ub_scores）
        hivm.hir.vmul ins(%ub_scores, %scales : ...) outs(%ub_scores : ...)
        hivm.hir.vreduce <max> ins(%ub_scores : ...) outs(%local_max : ...)
        // ... 完整的 softmax ops ...
        hivm.hir.vcast ins(%scores_f32 : ...) outs(%scores_f16 : ...)
        hivm.hir.vbrc ins(%cst : ...) outs(%tmp1 : ...)
        hivm.hir.vadd ins(%tmp1, %new_max : ...) outs(%acc_m : ...)
        // scores_cast: UB → workspace
        hivm.hir.store ins(%scores_f16 : memref<64x64xf16>)
          outs(%ws_scores_cast_view : memref<64x64xf16>)
        scope.return
      } {tcore_type = VECTOR, stage_id = 0}

      // ---- C1: 从 workspace 读 scores_cast，mmadL1 结果 fixpipe 到 workspace ----
      scope.scope {
        // workspace → L1
        %ub_scores_cast = memref.alloc() : memref<64x64xf16>
        hivm.hir.load ins(%ws_scores_cast_view) outs(%ub_scores_cast) ...
        %V_sub = memref.subview %V_gm[...] : ...
        hivm.hir.nd2nz ins(%V_sub) outs(%V_l1) ...
        // mmadL1 写到 fresh buffer（不再累加到 acc_o）
        %matmul_result = memref.alloc() : memref<64x128xf32>
        hivm.hir.mmadL1 {fixpipe_already_inserted = true}
          ins(%ub_scores_cast, %V_l1, %true, ...)       // initC=true! 不再累加
          outs(%matmul_result : memref<64x128xf32>)
        // fixpipe: L0C → workspace (GM)
        hivm.hir.fixpipe {enable_nz2nd}
          ins(%matmul_result : memref<64x128xf32>)
          outs(%ws_acc_o_view : memref<64x128xf32>)
        scope.return
      } {tcore_type = CUBE, stage_id = 1}

      // ---- V0(隐式，下一轮迭代): 从 workspace 读 acc_o，vadd 累加 ----
      // （在下一轮 V0 scope 的开头）：
      //   %ws_acc_o = hivm.hir.load ins(%ws_acc_o_view) ...
      //   %acc_o_new = hivm.hir.vadd ins(%ws_acc_o, %corrected_acc_o) ...
    }

    // ======== 新增：host 端 workspace size 回调 ========
    // (在 module 级别生成)
    func.func @flash_attention_infer_workspace_shape_function() -> index
      attributes {
        hacc.function_kind = #hacc.function_kind<HOST>,
        hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>
      } {
      %c57344 = arith.constant 57344 : index  // per-block size
      return %c57344 : index
    }
```

---

### A.4 Pass ④ inject_block_sync 期望输出

**变换要点**：
- 采用 **双向同步协议**：每条通道需要 "数据就绪" + "已读完可覆写" 两个信号
- 使用 4 个 Flag ID（flag 1 复用，因三个"就绪"信号时序不重叠）
- 循环前插入初始 "已读完" 信号

**Flag ID 分配**：

| Flag ID | 语义 | 方向 | 对应 pipe |
|---------|------|------|-----------|
| 0 | Vector 已读完 scores | AIV → AIC | PIPE_MTE2 |
| 1 | 数据就绪（复用） | 双向（时序不重叠） | PIPE_FIX / PIPE_MTE3 |
| 2 | Cube 已读完 scores_cast | AIC → AIV | PIPE_MTE2 |
| 3 | Vector 已读完 acc_o | AIV → AIC | PIPE_MTE2 |

```mlir
    // ======== 初始信号（循环前）========
    // AIC 侧：初始通知 "scores_cast 空间可写"
    scope.scope {
      hivm.hir.nd2nz ...  // 加载 Q（不变）
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2  // ← 初始
      scope.return
    } {tcore_type = CUBE, stage_id = 0}

    // AIV 侧：初始通知 "scores 空间可写" + "acc_o 空间可写"
    scope.scope {
      hivm.hir.vbrc ...   // 初始化（不变）
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0  // ← 初始
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3  // ← 初始
      scope.return
    } {tcore_type = VECTOR, stage_id = 0}

    // ======== 主循环 ========
    scf.for ... {
      // ---- C0: fixpipe scores → workspace ----
      scope.scope {
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0  // ← 等 V 读完 scores
        // ... nd2nz K + mmadL1 ...
        hivm.hir.fixpipe {enable_nz2nd}
          ins(%scores_local) outs(%ws_scores_view)          // scores → workspace
        // 使用 annotation.mark 标注 workspace view（参考输出中出现）
        annotation.mark %ws_scores_view : memref<64x64xf32>
        annotation.mark %ws_scores_view : memref<64x64xf32>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1   // ← 通知 V: scores 就绪
        scope.return
      } {tcore_type = CUBE, stage_id = 0}

      // ---- V0: softmax ----
      scope.scope {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1 // ← 等 C: scores 就绪
        %ub_scores = memref.alloc() : memref<64x64xf32>
        hivm.hir.load ins(%ws_scores_view) outs(%ub_scores) ...          // workspace → UB
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0 // ← 通知 C: 已读完 scores
        // ... softmax 计算 ...
        hivm.hir.vcast ins(%scores_f32) outs(%scores_f16)
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2 // ← 等 C 读完 scores_cast
        hivm.hir.store ins(%scores_f16) outs(%ws_scores_cast_view)        // VID=0 写 workspace
        annotation.mark %ws_scores_cast_view : memref<64x64xf16>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1 // ← 通知 C: scores_cast 就绪
        // ... 更新 acc_m ...
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1 // ← 等 C: acc_o 就绪
        %ub_acc_o = memref.alloc() : memref<64x128xf32>
        hivm.hir.load ins(%ws_acc_o_view) outs(%ub_acc_o) ...            // workspace → UB
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3 // ← 通知 C: 已读完 acc_o
        hivm.hir.vadd ins(%ub_acc_o, %corrected_acc_o) outs(%acc_o_new)  // 累加
        scope.return
      } {tcore_type = VECTOR, stage_id = 0}

      // ---- C1: fixpipe acc_o → workspace ----
      scope.scope {
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1  // ← 等 V 写好 scores_cast
        %ub_scores_cast = memref.alloc() : memref<64x64xf16>
        hivm.hir.load ins(%ws_scores_cast_view) outs(%ub_scores_cast) ...
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2   // ← 通知 V: 已读完 scores_cast
        // ... nd2nz V + mmadL1 → matmul_result ...
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3  // ← 等 V 读完上一轮 acc_o
        hivm.hir.fixpipe {enable_nz2nd}
          ins(%matmul_result) outs(%ws_acc_o_view)         // acc_o → workspace
        annotation.mark %ws_acc_o_view : memref<64x128xf32>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1   // ← 通知 V: acc_o 就绪
        scope.return
      } {tcore_type = CUBE, stage_id = 1}
    }

    // ======== 循环后同步（确保最后一轮完成）========
    // AIC 侧
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0  // 等 V 读完最后一轮 scores
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3  // 等 V 读完最后一轮 acc_o
```

---

### A.5 Pass ⑤ insert_vid 期望输出

**变换要点**：
- 1 Cube + 2 Vector Sub-Block 架构，两个 Vector 跑相同程序
- 采用 **写入守卫模式**：两个 Sub-Block 冗余执行所有 softmax 计算，
  仅 VID=0 执行 workspace 写入和最终 GM 写出
- 在 Vector scope 头部插入 `get_sub_block_idx` 获取 VID
- 用 `scf.if (VID==0)` 包裹所有写操作

```mlir
      // ---- V0: 写入守卫模式 ----
      scope.scope {
        // ======== 获取 VID ========
        %vid = hivm.hir.get_sub_block_idx -> i64       // ← 获取 Sub-Block ID
        %vid_idx = arith.index_cast %vid : i64 to index
        %c0 = arith.constant 0 : index
        %is_vid0 = arith.cmpi eq, %vid_idx, %c0 : index

        // ======== 读 workspace：所有 Sub-Block 都读 ========
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %ub_scores = memref.alloc() : memref<64x64xf32>
        hivm.hir.load ins(%ws_scores_view) outs(%ub_scores) ...
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0

        // ======== 计算：所有 Sub-Block 都执行（冗余计算）========
        hivm.hir.vmul ins(%ub_scores, %scales : ...) outs(%ub_scores : ...)
        hivm.hir.vreduce <max> ins(%ub_scores : ...) outs(%local_max : ...)
        hivm.hir.vmax ins(%acc_m, %local_max : ...) outs(%new_max : ...)
        hivm.hir.vsub ins(%acc_m, %new_max : ...) outs(%tmp1 : ...)
        hivm.hir.vexp ins(%tmp1 : ...) outs(%correction : ...)
        hivm.hir.vsub ins(%ub_scores, %new_max : ...) outs(%tmp2 : ...) broadcast = [1]
        hivm.hir.vexp ins(%tmp2 : ...) outs(%ub_scores : ...)
        hivm.hir.vreduce <sum> ins(%ub_scores : ...) outs(%local_sum : ...)
        hivm.hir.vmul ins(%acc_l, %correction : ...) outs(%acc_l : ...)
        hivm.hir.vadd ins(%acc_l, %local_sum : ...) outs(%acc_l : ...)
        hivm.hir.vmul ins(%acc_o, %correction : ...) outs(%acc_o : ...) broadcast = [1]
        hivm.hir.vcast ins(%ub_scores : ...) outs(%scores_f16 : ...)

        // ======== 写 workspace：仅 VID=0 ========
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
        scf.if %is_vid0 {
          hivm.hir.store ins(%scores_f16 : memref<64x64xf16>)
            outs(%ws_scores_cast_view : memref<64x64xf16>)
        }
        annotation.mark %ws_scores_cast_view : memref<64x64xf16>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1

        // 更新 acc_m
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%tmp1 : ...)
        hivm.hir.vadd ins(%tmp1, %new_max : ...) outs(%acc_m : ...)

        // 读 acc_o workspace + 累加
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
        %ws_acc_o = memref.alloc() : memref<64x128xf32>
        hivm.hir.load ins(%ws_acc_o_view) outs(%ws_acc_o) ...
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
        hivm.hir.vadd ins(%ws_acc_o, %acc_o : ...) outs(%acc_o_new : ...)
        scope.return
      } {tcore_type = VECTOR, stage_id = 0}

    // ======== 循环后：Vector 写最终输出（仅 VID=0）========
    scope.scope {
      hivm.hir.vdiv ins(%acc_o, %acc_l : ...) outs(%acc_o : ...) broadcast = [1]
      %output_f16 = memref.alloc() : memref<64x128xf16>
      hivm.hir.vcast ins(%acc_o : ...) outs(%output_f16 : ...)
      %gm_output = memref.subview %output_gm[%block_offset, 0] [64, 128] [1, 1] : ...
      scf.if %is_vid0 {
        hivm.hir.store ins(%output_f16 : memref<64x128xf16>)
          outs(%gm_output : memref<64x128xf16, strided<[128, 1], offset: ?>>)
      } {limit_sub_block_id0}
      scope.return
    } {tcore_type = VECTOR, stage_id = 1}
```

> **注**：写入守卫模式下两个 Sub-Block 冗余计算，浪费一倍 Vector 算力，
> 但避免了 reduce 操作的跨 Sub-Block merge 同步。对 softmax 这种计算密集但
> 数据量小的场景，冗余计算代价可接受。后续可演进到数据划分模式以优化吞吐。

---

### A.6 Pass ⑥ outline_scope 期望输出

**变换要点**：
- 每个 `tilelangir.cv_scope` 被提取为独立的 `func.func`
- 原位替换为 `func.call`
- CUBE scope → `@xxx_aic` 函数，VECTOR scope → `@xxx_aiv` 函数
- 所有被 scope 引用的外部 value 成为函数参数

```mlir
module {
  // ======== 提取出的 Cube 函数 ========
  func.func @flash_attention_c0_aic(
      %Q_l1: memref<64x128xf16, ..., #hivm.address_space<cbuf>>,
      %K_gm: memref<512x128xf16, ..., #hivm.address_space<gm>>,
      %ws_scores: memref<64x64xf32, #hivm.address_space<gm>>,
      %k_idx: index, ...)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // nd2nz K + mmadL1 + copy to workspace + sync_block_set
    // ...
    return
  }

  func.func @flash_attention_c1_aic(
      %ws_scores_cast: memref<64x64xf16, #hivm.address_space<gm>>,
      %V_gm: memref<512x128xf16, ..., #hivm.address_space<gm>>,
      %ws_acc_o: memref<64x128xf32, #hivm.address_space<gm>>,
      %k_idx: index, ...)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // sync_block_wait + copy from workspace + nd2nz V + mmadL1 + copy to workspace + sync_block_set
    // ...
    return
  }

  // ======== 提取出的 Vector 函数 ========
  func.func @flash_attention_v0_aiv(
      %ws_scores: memref<64x64xf32, #hivm.address_space<gm>>,
      %ws_scores_cast: memref<64x64xf16, #hivm.address_space<gm>>,
      %acc_m: memref<64x1xf32, ..., #hivm.address_space<zero>>,
      %acc_l: memref<64x1xf32, ..., #hivm.address_space<zero>>,
      %acc_o: memref<64x128xf32, ..., #hivm.address_space<zero>>, ...)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    // sync_block_wait + copy from workspace + softmax + copy to workspace + sync_block_set
    // (含 VID 划分逻辑)
    // ...
    return
  }

  // ======== 入口函数变为调度器 ========
  func.func @flash_attention(%arg0: i64, %arg1: memref<?xi8>, ...)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // ... 常量、alloc、初始化 ...
    func.call @flash_attention_init_aiv(...)  // 循环前 Vector 初始化
    func.call @flash_attention_load_q_aic(...)  // 循环前 Cube 加载 Q

    scf.for %i = ... {
      func.call @flash_attention_c0_aic(...)   // C0
      func.call @flash_attention_v0_aiv(...)   // V0
      func.call @flash_attention_c1_aic(...)   // C1
    }

    func.call @flash_attention_epilogue_aiv(...)  // 循环后 Vector 收尾
    return
  }

  // ... (如果做了 pipeline 展开，调度结构会更复杂，包含 prologue/epilogue) ...
}
```

---

### A.7 Pass ⑦ infer_task_type 期望输出 (已由参考输出验证)

**变换要点**：
- 入口函数 `@flash_attention` 的 `hivm.func_core_type = MIX`
- 生成 host 端回调，返回 `TaskType::CubeVectorMix_1_2 = 32`（1 Cube + 2 Vector）

```mlir
module {
  // ======== 新增：host 端 task type 回调 ========
  func.func @flash_attention_infer_task_type_function() -> i8
    attributes {
      hacc.host,
      hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>
    } {
    %c = arith.constant 32 : i8  // CubeVectorMix_1_2
    return %c : i8
  }

  // ======== 之前 Pass ④ 中已生成的 workspace size 回调 ========
  func.func @flash_attention_infer_workspace_size_function() -> i64
    attributes {hacc.host, ...} {
    %size = arith.constant 57344 : i64
    return %size : i64
  }

  // ======== Cube / Vector / 入口函数不变 ========
  func.func @flash_attention_c0_aic(...) { ... }
  func.func @flash_attention_c1_aic(...) { ... }
  func.func @flash_attention_v0_aiv(...) { ... }
  func.func @flash_attention(...) attributes {hivm.func_core_type = MIX} { ... }
}
```

---

## 附录 B：与 bishengir 参考输出的对照验证

参考输出文件：`example-reference.mlir`，由 bishengir 被跳过的 pass pipeline 产出。

### B.1 验证结论总览

| 维度 | 我们的设计 | 参考输出 | 结论 |
|------|-----------|---------|------|
| Module core type | AIC → **MIX** | `#hivm.module_core_type<MIX>` | ✅ 一致 |
| 函数拆分 | scope → outline → AIC/AIV 函数 | `@flash_attention_mix_aic` + `@flash_attention_mix_aiv` | ✅ 一致 |
| Workspace 布局 | scores(16384) + scores_cast(8192) + acc_o(32768) | 完全一致，total=57344 | ✅ 一致 |
| Workspace host 回调 | `infer_workspace_size_function` | `infer_workspace_shape_function`，返回 57344 | ✅ 一致（函数名略不同） |
| Task type 回调 | 返回 32 (MIX 1:2) | `arith.constant 32 : i8` | ✅ 一致 |
| Block sync | 3 个 Flag ID | **4 个 Flag ID**（双向协议） | ⚠️ **需要修正** |
| VID 用途 | 按行切分数据，各 Sub-Block 算不同行 | **写入守卫**：所有 Sub-Block 算相同数据，仅 VID=0 写 workspace/output | ⚠️ **需要修正** |
| Cube→workspace 方式 | `memref.copy` UB→GM | **`fixpipe`** L0C→workspace(GM) | ⚠️ **需要修正** |
| acc_o 累加方式 | mmadL1 直接累加到 acc_o | Cube 写到 fresh buffer→workspace，**Vector 做 `vadd`** | ⚠️ **需要修正** |
| 多缓冲 / 流水 | loop tiling (simple_multibuffer) | 无多缓冲，靠 **细粒度 sync** 实现重叠 | ℹ️ 不同策略，均可行 |

### B.2 需修正的设计细节

#### 修正 1：双向同步协议（4 个 Flag ID）

我们原计划为每个 workspace 通道分配 1 个 Flag ID（producer → consumer），
但参考输出表明需要 **双向协议**：每个通道需要 2 个信号——
"数据已就绪"（producer→consumer）和"已读完可覆写"（consumer→producer）。

参考输出的 Flag 分配：

```
flag 0: "Vector 已读完 scores"    — AIV → AIC（Cube 可覆写 scores workspace）
flag 1: "数据已就绪"              — 复用于 scores/scores_cast/acc_o 的就绪信号
                                     （三者时序不重叠，可共享同一 flag）
flag 2: "Cube 已读完 scores_cast" — AIC → AIV（Vector 可覆写 scores_cast workspace）
flag 3: "Vector 已读完 acc_o"     — AIV → AIC（Cube 可覆写 acc_o workspace）
```

AIC 循环体内的时序：
```
sync_block_wait flag=0              // 等 Vector 读完上一轮 scores
fixpipe scores → workspace          // 写 scores
sync_block_set flag=1               // 通知 Vector: scores 就绪
sync_block_wait flag=1              // 等 Vector 写好 scores_cast
load scores_cast ← workspace       // 读 scores_cast
sync_block_set flag=2               // 通知 Vector: 已读完 scores_cast
load V, mmadL1                      // 计算 scores_cast × V
sync_block_wait flag=3              // 等 Vector 读完上一轮 acc_o
fixpipe acc_o → workspace           // 写 acc_o
sync_block_set flag=1               // 通知 Vector: acc_o 就绪
```

AIV 循环体内的时序：
```
sync_block_wait flag=1              // 等 Cube: scores 就绪
load scores ← workspace            // 读 scores
sync_block_set flag=0               // 通知 Cube: 已读完 scores
softmax(scores) → scores_cast       // 计算
sync_block_wait flag=2              // 等 Cube: 已读完上一轮 scores_cast
if VID==0: store scores_cast → ws   // 仅 VID=0 写 workspace
sync_block_set flag=1               // 通知 Cube: scores_cast 就绪
sync_block_wait flag=1              // 等 Cube: acc_o 就绪
load acc_o ← workspace             // 读 acc_o
sync_block_set flag=3               // 通知 Cube: 已读完 acc_o
acc_o += correction * old_acc_o     // 累加
```

#### 修正 2：VID 用作写入守卫（非数据划分）

参考输出的 VID 用法：

```mlir
// AIV 函数中
%vid = hivm.hir.get_sub_block_idx -> i64
%is_vid0 = arith.cmpi eq, %vid, %c0 : index

// 写 workspace 时仅 VID=0 执行
scf.if %is_vid0 {
  hivm.hir.store ins(%scores_cast) outs(%ws_view)
}

// 写最终输出时仅 VID=0 执行
scf.if %is_vid0 {
  hivm.hir.store ins(%output) outs(%gm_subview)
} {limit_sub_block_id0}
```

**两个 Vector Sub-Block 冗余计算相同的 softmax**，但只有 VID=0 执行写回。
这避免了两个 Sub-Block 同时写同一地址的冲突，且不需要 partial reduce + merge。

对于 Flash Attention 这种 reduce 操作密集的场景，冗余计算比数据划分更简单——
行维度的 reduce (max/sum) 如果切分需要额外的跨 Sub-Block 同步和 merge。

#### 修正 3：Cube→workspace 使用 fixpipe（非 memref.copy）

参考输出中 Cube 侧写 workspace 使用 `hivm.hir.fixpipe`：

```mlir
// AIC 函数中
%alloc_6 = memref.alloc() : memref<64x64xf32>       // L0C 局部 buffer
hivm.hir.mmadL1 ... outs(%alloc_6 : memref<64x64xf32>)  // mmad → L0C
hivm.hir.fixpipe {enable_nz2nd}
  ins(%alloc_6 : memref<64x64xf32>)
  outs(%view : memref<64x64xf32>)                    // fixpipe: L0C → workspace(GM)
```

**fixpipe 直接从 L0C 写到 workspace**（而不是先到 UB 再 DMA 到 GM），
这是硬件支持的路径，比 `memref.copy` 更高效。

#### 修正 4：acc_o 累加被拆分到两个核

输入 IR 中 mmadL1 直接累加到 acc_o：
```mlir
// 输入：mmadL1 累加模式（initC=false → acc_o += scores_cast × V）
hivm.hir.mmadL1 ins(%alloc_14, %alloc_12, %false, ...) outs(%alloc_7 : ...)
```

参考输出中被拆分为：
```mlir
// AIC: mmadL1 写到 fresh buffer，fixpipe 到 workspace
hivm.hir.mmadL1 {fixpipe_already_inserted = true}
  ins(...) outs(%alloc_10 : memref<64x128xf32>)      // 不累加，写到新 buffer
hivm.hir.fixpipe ins(%alloc_10) outs(%view_3)         // → workspace

// AIV: 从 workspace 加载，与 correction * old_acc_o 相加
hivm.hir.load ins(%view_6) outs(%alloc_16)            // workspace → UB
hivm.hir.vadd ins(%alloc_16, %arg11) outs(%alloc_17)  // acc_o_new = ws_result + corrected_old
```

原因：Cube 的 mmadL1 无法读取 Vector 持有的 acc_o（在 Vector 的 UB 中），
所以累加被拆为：Cube 只算乘法结果，Vector 负责累加。

### B.3 Workspace 的 per-block 布局

参考输出使用 `block_idx * 57344` 作为每个 block 的 workspace 基地址偏移：

```mlir
#map  = affine_map<()[s0] -> (s0 * 57344)>         // scores 基地址
#map1 = affine_map<()[s0] -> (s0 * 57344 + 16384)> // scores_cast 基地址
#map2 = affine_map<()[s0] -> (s0 * 57344 + 24576)> // acc_o 基地址

%block_idx = hivm.hir.get_block_idx -> i64
%idx = arith.index_cast %block_idx : i64 to index
%base = affine.apply #map()[%idx]
%view = memref.view %workspace[%base][] : memref<?xi8> to memref<64x64xf32>
```

**每个 AI Core (block) 都有独立的 57344 字节 workspace**，互不干扰。
总 workspace 大小 = `57344 × num_blocks`。
host 回调函数返回的 57344 是 **per-block** 大小，runtime 负责乘以 block 数。

### B.4 simple_multibuffer 与参考方案的对比

参考输出 **没有** 使用多缓冲或循环 tiling，而是在每次迭代内通过细粒度 sync 实现 C/V 重叠。

| 维度 | 参考方案（细粒度 sync） | 我们的方案（simple_multibuffer） |
|------|------------------------|-------------------------------|
| 循环结构 | 原循环不变，每次迭代内 C→sync→V→sync→C | 外层 tiling + 每个 scope 独立内层循环 |
| 同步频率 | 每迭代 ~8 次 sync | 每 tile 组 ~6 次 sync |
| 并行粒度 | 迭代内 C/V 细粒度重叠 | 批量 C 与批量 V 重叠 |
| 缓冲数量 | 单缓冲（workspace 每次覆写） | tile_factor 份缓冲 |
| 实现复杂度 | 高（精细的 sync 时序） | **低（loop tiling + distribution）** |

两种方案都是正确的。作为穿刺方案，simple_multibuffer 更容易实现，
后续可以演进到参考方案的细粒度 sync 模式以获得更好的性能。
