# TileLang-Ascend 同步与 Workspace 经验总结

## 1. 昇腾 NPU 硬件架构概要

### 1.1 核类型

| 核类型 | 缩写 | 职责 | MLIR 标记 |
|--------|------|------|-----------|
| Cube Core | AIC | 矩阵运算 (GEMM) | `hivm.func_core_type = #hivm.func_core_type<AIC>` |
| Vector Core | AIV | 向量运算、DMA 搬运 | `hivm.func_core_type = #hivm.func_core_type<AIV>` |
| 混合模式 | MIX | AIC + AIV 协同 | `mix_mode = "mix"`, `hivm.part_of_mix` |

MIX 模式下，AIC 和 AIV 各有独立的函数，通过 `sync_block_set`/`sync_block_wait` 进行跨核同步。

### 1.2 存储层次

| 存储 | 地址空间 | MLIR 标记 | 所属核 |
|------|---------|-----------|--------|
| Global Memory (GM) | `gm` | `#hivm.address_space<gm>` | 共享 |
| L1 Buffer (cbuf) | `cbuf` | `#hivm.address_space<cbuf>` | AIC |
| L0A | `ca` | `#hivm.address_space<ca>` | AIC |
| L0B | `cb` | `#hivm.address_space<cb>` | AIC |
| L0C | `cc` | `#hivm.address_space<cc>` | AIC |
| Unified Buffer (UB) | `ub` | `#hivm.address_space<ub>` | AIV |

### 1.3 硬件流水线 (Pipeline)

**AIC 核的流水线：**
- `PIPE_MTE2`: DMA 搬运 GM → L1 (`nd2nz`)
- `PIPE_M` / `CUBE`: 矩阵计算 (`mmadL1`)
- `PIPE_FIX`: 数据搬出 L0C → GM (`fixpipe`)

**AIV 核的流水线：**
- `PIPE_MTE2`: DMA 搬运 GM → UB (`memref.copy` GM→UB)
- `VECTOR`: 向量计算 (`vexp`, `vadd`, `vmul` 等)
- `PIPE_MTE3`: DMA 搬运 UB → GM (`memref.copy` UB→GM)

关键事实：**同一核内的不同流水线是独立、异步执行的**。硬件不会自动追踪跨流水线的数据依赖。

## 2. 同步机制

### 2.1 核内同步 (Intra-core)

#### pipe_barrier

```mlir
hivm.hir.pipe_barrier[<PIPE_ALL>]
```

等待当前核**所有流水线**排空后再继续。这是最粗粒度的同步方式，正确但性能差。

#### set_flag / wait_flag（精细同步）

```mlir
hivm.hir.set_flag[<src_pipe>] event_id = N
hivm.hir.wait_flag[<dst_pipe>] event_id = N
```

基于事件 ID (0-4) 的细粒度同步。`set_flag` 在源流水线操作完成后发出信号，`wait_flag` 在目标流水线操作开始前等待信号。这是 `InjectSync` NORMAL 模式生成的同步方式。

### 2.2 跨核同步 (Cross-core, AIC ↔ AIV)

#### sync_block_set / sync_block_wait

```mlir
// AIC 侧：CUBE 完成 fixpipe 后通知 AIV
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %val
    syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>

// AIV 侧：等待 AIC 的通知后开始 MTE2 搬运
hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %val
```

- `sync_block_set` 的 pipe 列表 `[<CUBE>, <PIPE_FIX>, <PIPE_S>]` 表示信号绑定在 `PIPE_FIX` 阶段完成后才发出
- `sync_block_wait` 的 pipe 列表 `[<VECTOR>, <PIPE_S>, <PIPE_MTE2>]` 表示等待信号后才允许 `PIPE_MTE2` 执行
- `flag` 值用于区分不同迭代的同步，通常等于循环迭代变量的某个变换

## 3. InjectSync Pass 分析

### 3.1 源码位置

```
3rdparty/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/InjectSync/InjectSync.cpp
```

### 3.2 两种模式

Pass 由 `--enable-hivm-inject-barrier-all-sync` 标志控制，对应 `SyncMode` 枚举：

```cpp
enum class SyncMode {
  NORMAL,       // 精细依赖分析
  BARRIERALL,   // 暴力全插 (only for debug)
};
```

#### BARRIERALL 模式

```cpp
void InjectSyncAnalysis::InjectSyncAll() {
  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getDialect()->getNamespace() ==
            HIVMDialect::getDialectNamespace() ||
        mlir::isa<func::ReturnOp>(op)) {
      rewriter.setInsertionPoint(op);
      rewriter.create<hivm::PipeBarrierOp>(loc, PipeAttr::get(ctx, PIPE::PIPE_ALL));
    }
  });
}
```

逻辑极简：遍历函数中所有 operation，只要属于 `hivm` dialect 或是 `func.return`，就在其**前方**无条件插入 `pipe_barrier[<PIPE_ALL>]`。不做任何依赖分析。

#### NORMAL 模式

6 阶段精细分析流水线：

1. **IRTranslator** — 遍历 IR，构建 SyncIR 表示，分类操作的 pipe 类型并追踪 buffer 读写
2. **SyncAnalyzer** — 基于 buffer 的 WAR/WAW/RAW 依赖，确定需要同步的操作对
3. **MoveSyncState** — 优化同步点位置
4. **RemoveRedundantSync** — 删除冗余同步
5. **SyncEventIdAllocation** — 分配硬件 event ID (0-4)，用 set_flag/wait_flag 替代 pipe_barrier
6. **SyncCodegen** — 生成最终同步 IR

### 3.3 NORMAL 模式的限制

`IRTranslator::RecursionIR` 有严格的 op 白名单：

```cpp
// 识别的 op 类型:
// - ViewLikeOpInterface (alias info)
// - PointerCastOp (alloc-like)
// - scf::ForOp, WhileOp, IfOp (控制流)
// - DestinationStyleOpInterface (hivm.hir.* 运算)
// - memref::LoadOp, StoreOp
// - affine::AffineLoadOp, AffineStoreOp

// 不认识的 op 如果触及 local/global buffer 就会报错:
if (isOpTouchLocalBuffer(op) || isOpTouchGlobalBuffer(op)) {
  op->emitError("InjectSync Fail : Unrecognized type of Operation "
                "touches local or global buffer! ");
  return failure();
}
```

**`memref.alloc` (带 strided layout) 和 `memref.copy` 不在白名单中**，这导致 NORMAL 模式在我们手写的 MLIR 上失败。原生流程中这些 op 在 InjectSync 之前已被 lower 为 `hivm.hir.load`/`hivm.hir.store` 和 `hivm.hir.pointer_cast`。

### 3.4 Pipeline 中的位置

```
hivmPostBufferizationOptimizationPipeline:
  ...
  → createHIVMLowerToLoopsPass()      // Lower hivm ops to loops
  → createHIVMDecomposeOpPass()
  → hivmNormSyncPipeline()            // ← InjectSync 在这里
  → createAddFFTSToSyncBlockSetOpPass()
  → createEnableMultiBufferPass()
  → createLiftLowestStridePass()
```

## 4. Workspace 管理

### 4.1 Workspace 的作用

在 MIX 模式下，AIC 的计算结果需要传递给 AIV 做后处理。数据通路是：

```
AIC: L0C → fixpipe → GM(workspace) → sync_block_set
AIV: sync_block_wait → GM(workspace) → MTE2 → UB → VECTOR 计算 → UB → MTE3 → GM(output)
```

Workspace 是 GM 上的一块共享缓冲区，充当 AIC 和 AIV 之间的数据中转站。

### 4.2 函数参数约定

```mlir
func.func @minicv_mix_aic(
  %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},  // FFTS 基地址
  %arg1: memref<?xi8>,                                              // sync block lock
  %arg2: memref<?xi8, #hivm.address_space<gm>>,                    // workspace (raw bytes)
  %arg3: memref<?xf16, #hivm.address_space<gm>>,                   // 数据参数...
  ...
) attributes {
  SyncBlockLockArgIdx = 0 : i64,   // %arg1 的索引（从非FFTS参数起算）
  WorkspaceArgIdx = 1 : i64,       // %arg2 的索引
  hacc.entry,
  hacc.function_kind = #hacc.function_kind<DEVICE>,
  hivm.func_core_type = #hivm.func_core_type<AIC>,
  hivm.part_of_mix,
  mix_mode = "mix"
}
```

### 4.3 使用 memref.view 访问 Workspace

Workspace 参数是 `memref<?xi8>` 类型的原始字节缓冲区。要将其解释为特定形状和类型的 tensor，使用 `memref.view`：

```mlir
%c0 = arith.constant 0 : index
%reinterpret_cast_ws = memref.view %arg2[%c0][]
    : memref<?xi8, #hivm.address_space<gm>>
    to memref<4x3x64x32xf16, #hivm.address_space<gm>>
```

**不能用 `memref.reinterpret_cast`**（不支持元素类型变更），**不能用 `memref_ext.alloc_workspace`**（不被 `InferHIVMMemScope` pass 识别，会导致编译失败）。

#### memref_ext.AllocWorkspaceOp 失败的原因

`InferHIVMMemScope` pass 会尝试为 `memref<?xi8>` 参数传播地址空间。它遍历参数的所有 user op，只支持实现了 `ViewLikeOpInterface` 等特定 interface 的 op。`memref_ext.alloc_workspace` 不在此列，导致：
```
'func.func' op Failed to propagate memory scope for argument #2
'memref_ext.alloc_workspace' op Unsupported user for root alloc op.
```

**解决方案**：直接给 workspace 参数标注地址空间 `#hivm.address_space<gm>`，并用 `memref.view`（实现了 `ViewLikeOpInterface`）来创建 typed view。

### 4.4 Workspace 大小计算

```python
# Workspace 大小 = 所有核需要共享的中间数据总量
# 例如 vadd: 4 blocks × 3 iterations × 64 × 32 × sizeof(f16)
WS = num_blocks * num_iters * tile_M * tile_K * element_size
# = 4 * 3 * 64 * 32 * 2 = 49152 bytes

# Host 侧总分配 = WS * blockNum (每个 block 有独立 workspace)
totalWorkSpaceSize = WS * blockNum
```

在 `jit_npu.py` 中的 C++ wrapper 代码中：
```cpp
uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
```

### 4.5 Workspace 数据布局

以 vadd 为例，workspace 的 4D 布局为 `[block_id, iter, M_tile, K_tile]`：

```mlir
// 形状: [4, 3, 64, 32] x f16
// block_id: 区分不同 block 的数据
// iter: 区分内循环迭代的数据
// M_tile × K_tile: 单次计算的 tile 大小

// AIC 通过 fixpipe 写入:
%subview = memref.subview %ws[%block_id, %iter, 0, 0] [1, 1, 64, 32] [1, 1, 1, 1]
hivm.hir.fixpipe ins(%l0c_buf) outs(%subview)

// AIV 通过 memref.copy 读出:
%subview = memref.subview %ws[%block_id, %iter, %sub_offset, 0] [1, 1, 32, 32] [1, 1, 1, 1]
memref.copy %subview, %ub_buf  // GM → UB
```

## 5. 同步插入规则

### 5.1 AIC 核内：不需要手动 pipe_barrier

AIC 的操作链 `nd2nz → mmadL1 → fixpipe` 形成串行数据流。各阶段通过共享片上缓冲区（L1、L0C）的硬件隐式依赖保证顺序：

- `mmadL1` 必须等 `nd2nz` 写完 L1 才能读
- `fixpipe` 必须等 `mmadL1` 写完 L0C 才能读

因此 AIC 内部**不需要**显式 `pipe_barrier`。

`sync_block_set` 上绑定的 `<PIPE_FIX>` 保证了信号在 fixpipe 完成后才发出：

```mlir
hivm.hir.mmadL1 ...
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %val
    syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
```

### 5.2 AIV 核内：必须手动 pipe_barrier

AIV 的 MTE2、VECTOR、MTE3 三条流水线**完全独立、异步**，硬件不自动追踪依赖。必须在以下位置插入同步：

#### 规则 1: sync_block_wait 之前

```mlir
hivm.hir.pipe_barrier[<PIPE_ALL>]     // ← 确保前一次迭代的 MTE3 写完
hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %val
```

等待 AIC 通知前，需确保 AIV 上一轮的所有操作（特别是 MTE3 写回 GM）已经完成，避免上一轮的 UB→GM 搬运与新一轮的 GM→UB 搬运在 GM 上产生冲突。

#### 规则 2: memref.copy (GM→UB) 之后、VECTOR 计算之前

```mlir
memref.copy %gm_subview, %ub_buf  // MTE2 pipeline: GM → UB
hivm.hir.pipe_barrier[<PIPE_ALL>]  // ← 等 MTE2 完成
hivm.hir.vexp ins(%ub_buf) ...     // VECTOR pipeline: 计算
```

DMA 搬运（MTE2）和向量计算（VECTOR）使用不同流水线。不加 barrier，VECTOR 可能读到 UB 中尚未搬运完的脏数据。

#### 规则 3: memref.copy (UB→GM) 之后、sync_block_set 之前

```mlir
memref.copy %ub_result, %gm_subview  // MTE3 pipeline: UB → GM
hivm.hir.pipe_barrier[<PIPE_ALL>]     // ← 等 MTE3 完成
hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %val
    syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
```

发出跨核同步信号前，需确保 UB→GM 的数据搬运已经完成。

#### 规则 4: 尾部循环中的 sync_block_wait 之前

```mlir
// 排空尾部迭代
scf.for %i = %c0 to %c3 step %c1 : i32 {
  %flag = ...
  hivm.hir.pipe_barrier[<PIPE_ALL>]     // ← 确保所有流水线排空
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %flag
}
```

#### 规则 5: return 之前

```mlir
hivm.hir.pipe_barrier[<PIPE_ALL>]  // ← 确保所有操作完成
return
```

### 5.3 完整的 AIV 同步模式（vadd 用例）

```mlir
scf.for %outer ... {
  scf.for %inner ... {
    %ub_in = memref.alloc() : memref<...xf16, #hivm.address_space<ub>>
    %ub_out = memref.alloc() : memref<...xf16, #hivm.address_space<ub>>

    // [barrier] 确保上一轮 MTE3 完成
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    // [跨核同步] 等待 AIC 写完 workspace
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %wait_flag

    // [DMA] GM → UB
    memref.copy %gm_input, %ub_in

    // [barrier] 确保 MTE2 完成
    hivm.hir.pipe_barrier[<PIPE_ALL>]

    // [计算] VECTOR pipeline
    hivm.hir.vexp ins(%ub_in) outs(%ub_out)

    // [DMA] UB → GM (写回 workspace)
    memref.copy %ub_out, %gm_workspace

    // [barrier] 确保 MTE3 完成
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    // [跨核同步] 通知 AIC 可以读 workspace
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %set_flag
  }
}
// [尾部排空]
scf.for %i ... {
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %tail_flag
}
// [最终 barrier]
hivm.hir.pipe_barrier[<PIPE_ALL>]
return
```

### 5.4 对比：AIV 的 flash_attn 参考用例 (example_exp.mlir)

flash_attn 中 AIV 的同步模式**没有手动 pipe_barrier**，原因是：

1. 所有操作都在 UB 上（VECTOR pipeline 内）
2. `memref.copy` GM↔UB 与 VECTOR 计算之间的依赖由原生 InjectSync NORMAL 模式处理（因为使用了标准的 `memref.copy` + `hivm.hir.vcast/vmul/...` 模式，IRTranslator 能正确识别）
3. `sync_block_wait`/`sync_block_set` 已经保证了跨核时序

而 vadd 用例中使用了 `memref.alloc` (带 strided layout) 等 IRTranslator 不认识的 op，导致 NORMAL 模式分析失败。

## 6. 编译标志参考

| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--disable-hivm-tensor-compile` | `false` | 跳过 tensor→memref 的 bufferization pipeline |
| `--enable-hivm-inject-barrier-all-sync` | `false` | 启用暴力 barrier_all 模式 (debug only) |
| `--enable-auto-multi-buffer` | `true` | 启用自动多缓冲 |
| `--disable-auto-inject-sync` | `false` | 完全禁用自动同步注入 |
| `--enable-hivm-graph-sync-solver` | `false` | 使用图求解器替代 InjectSync |
| `--enable-triton-kernel-compile` | `false` | 启用 triton kernel 编译路径 |
| `--enable-hivm-compile` | `false` | 启用 HIVM 编译 |

常用组合（手写 MLIR 编译）：
```bash
bishengir-compile input.mlir \
    --target=ascend910b3 \
    --enable-auto-multi-buffer=false \
    --enable-triton-kernel-compile=true \
    --enable-hivm-compile=true \
    --disable-hivm-tensor-compile=true \
    -o output
```

## 7. 踩坑记录

### 7.1 memref_ext.AllocWorkspaceOp 不能在 post-outline 函数中使用

`AllocWorkspaceOp` 设计用于 `SplitMixKernel` 之前的统一函数中，会被 `LowerMemRefExt` (即 `LowerAllocWorkSpace`) pass 处理。在已经 split 过的 AIC/AIV 函数中直接使用会导致 `InferHIVMMemScope` 失败。

### 7.2 memref.reinterpret_cast 不支持元素类型变更

从 `memref<?xi8>` 到 `memref<4x3x64x32xf16>` 必须用 `memref.view`，不能用 `memref.reinterpret_cast`。

### 7.3 Workspace 参数必须标注地址空间

```mlir
// 错误：无地址空间，InferHIVMMemScope 需要传播
%arg2: memref<?xi8>

// 正确：预标注 GM 地址空间，跳过传播
%arg2: memref<?xi8, #hivm.address_space<gm>>
```

### 7.4 UB 分配应在循环内部

将 UB 的 `memref.alloc` 放在循环内部（loop-scoped），而非提升到函数顶部。后者可能导致编译器后续 pass（如 PlanMemory）处理异常。

### 7.5 .so 缓存问题

编译并运行 NPU 程序时，pwd 下会生成 `.so` 文件。修改 MLIR 后重新编译，如果不删除旧 `.so`，有概率加载到缓存的旧版本，导致修改不生效。

### 7.6 sync_block_set 必须带 syn_instr_mode 属性

```mlir
// 错误：缺少 syn_instr_mode，硬件不知道这是 block 内 AIC/AIV 同步
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1

// 正确：显式标注同步模式
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
    syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>
```

C++ 中构造时必须添加：

```cpp
s.addAttribute("tsync_instr_mode",
    hivm::SyncBlockInstrModeAttr::get(ctx,
        hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION));
```

注意 C++ 属性名是 `tsync_instr_mode`（带 `t` 前缀），而汇编格式打印为 `syn_instr_mode`。缺少此属性会导致 NPU 运行时 507057 错误。

### 7.7 AIV 函数必须保留 hacc.entry 属性

MIX 模式 split 后，AIC 和 AIV 函数**都必须**标记 `hacc.entry`。`bishengir-compile` 依据此属性为函数生成正确的内核入口代码（prologue/epilogue）。如果 AIV 缺少 `hacc.entry`，会导致 NPU 运行时 "CCU instruction address check error"（错误码 507015）——本质是生成了无效的指令跳转地址。

### 7.8 AIV 中应保留 memref.copy，不要预转换为 hivm.hir.load/store

AIV 函数中的 `memref.copy`（GM↔UB 搬运）应保持原样，交给 `bishengir-compile` 内部 pipeline 处理。原因：

1. `bishengir-compile` 的内部 pipeline 会先将 `memref.copy` lower 为对应的 HIVM DMA op，然后再运行 InjectSync 来插入正确的同步。
2. 如果在喂给 `bishengir-compile` 之前就将 `memref.copy` 转换为 `hivm.hir.load`/`hivm.hir.store`，可能干扰内部 pass 的假设，导致生成错误的机器代码。
3. AIC 端不存在此问题：`nd2nz`、`fixpipe`、`mmadL1` 本身就是 HIVM op，是 AIC 的原生操作。

实践规则：**`outline-scope` 负责 AIC 端的 op 转换（memref.copy→fixpipe, load→nd2nz），AIV 端只做地址空间映射（cbuf→ub），不转换 DMA op。**

### 7.9 VID guard 只应包裹"写入共享内存"的操作

`insert-vid` pass 为 Vector Sub-Block 插入写入守卫（`scf.if VID==0`），防止两个 sub-block 重复写入同一 GM 地址。判定"写操作"时必须区分方向：

| 操作 | 方向 | 是否 VID guard |
|------|------|---------------|
| `memref.copy` GM→UB | 加载到私有 UB | 否（每个 sub-block 需加载自己的 tile） |
| `memref.copy` UB→GM | 写回共享 GM | 视情况而定（若写入地址含 sub_block_idx 偏移则无需 guard） |
| `hivm.hir.store` | 写回 GM | 是（除非地址已按 sub-block 分离） |

检查方式：判断 `memref.copy` 的 **source** 地址空间。若 source 为 `gm`，则为 GM→local 加载，不应 guard。

```cpp
if (auto srcAS = hivm::getHIVMAddressSpace(copyOp.getSource().getType()))
  if (srcAS == hivm::AddressSpace::GM)
    return false;  // GM→local = load, not a write
```

将 GM→UB 的 load 错误包裹在 VID guard 里，会导致 sub_block_idx=1 的核读到未初始化的 UB 数据，引发计算错误或 NPU 运行时异常。

### 7.10 outline-scope 必须清理 AIV 中的 dead 累加器

MIX split 后，原始函数中的 CUBE-only 缓冲区（如 `mmadL1` 累加器 alloc 和其 `memref.copy cbuf→gm`）会被克隆到 AIC 和 AIV 两份。在 AIV 中：

- `mmadL1` 所在的 CUBE scope 被 `processClone` 删除 → 累加器 alloc 的唯一 HIVM writer 消失
- 地址空间从 `cbuf` 映射为 `ub` → 产生一个函数级别的大 UB 分配（如 64x64xf32=16KB）
- 残留的 `memref.copy ub→gm` 会把**未初始化的 UB 数据**store 到 D 输出的 GM 地址，覆盖 AIC 通过 `fixpipe` 写出的正确结果

修复方式：在 AIV 处理流程中（`remapAllocAddressSpaces` 之后）增加 `handleMemCopiesInAIV`，与 AIC 端的 `handleMemCopiesInAIC` 对称：对 source 没有 HIVM users 的 local→GM copy 直接删除。后续 `eraseUnusedAllocs` 会自动清理无用的 alloc。

### 7.11 Python 测试脚本的维度必须与 MLIR kernel 对齐

`compile_vadd_pipeline.py` 中的 `M, N, K` 和 `WORKSPACE_SIZE` 必须与 MLIR 中 `reinterpret_cast` 的 sizes 严格一致。维度不匹配时，Python 端创建的 tensor 形状与 kernel 按 stride 读写的内存布局不一致，导致越界访问或数据错乱。

验证方法：检查 MLIR 入口函数中 `memref.reinterpret_cast` 的 sizes 字段：
```mlir
// A: (128, 256) → K=256, not 768
%A = memref.reinterpret_cast %arg3 ... sizes: [128, 256] ...
// C: (256, 128)
%C = memref.reinterpret_cast %arg5 ... sizes: [256, 128] ...
// workspace: (4, 64, 32)
%ws = memref.reinterpret_cast %arg7 ... sizes: [4, 64, 32] ...
```

### 7.12 bishengir-compile 编译标志：手写 MLIR 必须关闭 auto-multi-buffer

```bash
bishengir-compile input.mlir \
    --target=Ascend910B2 \
    --enable-auto-multi-buffer=false \       # ← 关键：手写 MLIR 必须关闭
    --enable-hivm-inject-barrier-all-sync \  # ← 用 BARRIERALL 模式
    --enable-triton-kernel-compile=true \
    --enable-hivm-compile=true \
    --disable-hivm-tensor-compile=true \
    -o output
```

`--enable-auto-multi-buffer=true` 会让编译器尝试自动对循环做多缓冲变换。当 IR 已经手动管理了单缓冲的同步协议时，auto-multi-buffer 可能错误地改变循环结构，导致同步失配和 NPU 运行时崩溃。
