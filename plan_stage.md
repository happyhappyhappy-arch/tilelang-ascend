# TileLang-Ascend 自定义 Pass Pipeline —— 分阶段实施规划

> 配合 `plan.md`（设计文档）和 `experience.md`（实战经验）使用。
> 本文档聚焦 **怎么做、做到什么程度算对**。

---

## 总体策略

**穿刺优先、从简到繁、逐层验证**：

1. **先用 vadd 穿刺验证**：vadd 是最简单的 MIX 用例（AIC: matmul → AIV: vexp），
   已在 `memref/vc_output_alloc_ws.mlir` 中完成手写 MLIR 验证（20/20 pass），
   积累的同步和 workspace 经验记录在 `experience.md` 中
2. 每个 pass 实现后用 vadd 用例做单 pass 验证
3. vadd 全链路跑通后，扩展到 flash_attn 用例
4. multibuffer 在功能穿刺完成后再做

**穿刺路径（7 个阶段）**：

```
Stage 0   基础设施搭建
   ↓
Stage 1   cv_annotate              ← 所有后续 pass 的前提
   ↓
Stage 2   insert_workspace         ← workspace 布局 + memref.view
   ↓
Stage 3   inject_block_sync        ← 核间 sync + AIV 核内 pipe_barrier
   ↓
Stage 4   insert_vid               ← 写入守卫
   ↓
Stage 5   outline_scope + infer_task_type
   ↓
Stage 5.5 vadd E2E 验证            ← 与 vc_output_alloc_ws.mlir 对比
   ↓
Stage 6   flash_attn E2E 验证      ← 与 example-reference.mlir 对比
   ↓
Stage 7   simple_multibuffer       ← 可选优化，穿刺后再做
```

### 关键经验（来自 vadd 穿刺验证）

以下发现直接影响 pass 设计，详见 `experience.md`：

| 发现 | 对 pass 设计的影响 |
|------|-------------------|
| `InjectSync` NORMAL 模式不认识 `memref.alloc`/`memref.copy` | Stage 3 必须手动插入 AIV 核内 `pipe_barrier[<PIPE_ALL>]` |
| `memref_ext.AllocWorkspaceOp` 不被 `InferHIVMMemScope` 识别 | Stage 2 使用 `memref.view` 而非 `alloc_workspace` |
| Workspace 参数必须预标注 `#hivm.address_space<gm>` | Stage 5 outline 时确保 arg type 正确 |
| AIC 不需要核内 barrier（硬件隐式依赖） | Stage 3 只对 AIV 插 pipe_barrier |
| UB alloc 应在循环内部（loop-scoped） | Stage 2/5 保持 alloc 在 scf.for 体内 |

---

## Stage 0：基础设施搭建

### 目标

让 7 个 pass 的空壳能编译通过、能在 lower.py 中被调用，确认从 Python → C++ pass 的调用链路畅通。

### 具体任务

| # | 任务 | 涉及文件 |
|---|------|---------|
| 0.1 | 在 `Passes.td` 中定义 7 个 pass（替换现有的 CVSplit/Vectorize） | `tilelangir/include/tilelangir/Transforms/Passes.td` |
| 0.2 | 为每个 pass 创建空壳 .cpp 实现（只打日志、不做变换） | `tilelangir/lib/Transforms/CVAnnotate.cpp` 等 7 个文件 |
| 0.3 | 更新 CMakeLists.txt 添加新的 .cpp | `tilelangir/lib/Transforms/CMakeLists.txt` |
| 0.4 | 在 lower.py 的 `tladapter_passes` 中接入新 pass | `tilelang/engine/lower.py` |
| 0.5 | 编译并跑一个简单用例验证 pass 被调用 | `testing/compile/flash_attn_npuir_dev.py` |

### 验证方法

```bash
# 编译
cd tilelangir && mkdir -p build && cd build && cmake .. && make -j

# 运行测试，观察 pass 日志
python testing/compile/flash_attn_npuir_dev.py 2>&1 | grep "tilelangir"
```

### 完成标准

- [ ] `make` 无报错
- [ ] 运行用例时能看到每个 pass 的日志输出（`[tilelangir-cv-annotate] running...` 等）
- [ ] 输出 IR 与不加 pass 时完全相同（空壳不做任何变换）

---

## Stage 1：cv_annotate — CV 分核标注

### 目标

将函数内的 Cube/Vector 操作识别并聚拢到 `scope.scope` 块中。
这是所有后续 pass 的基础。

### 输入 → 输出

- **输入**：`example_input.mlir`（C/V 操作混合在函数体中）
- **输出**：每个循环迭代内出现 3 个 scope（C0、V0、C1），循环前后各 1-2 个 scope

### 实现步骤

| # | 步骤 | 说明 |
|---|------|------|
| 1.1 | 实现 op 类型分类函数 | 根据 op name 或 interface 判断 Cube/Vector/DMA/Scalar |
| 1.2 | 实现 DMA 归属规则 | `nd2nz` GM→cbuf 归 Cube，`memref.copy` cbuf→GM 归 Vector |
| 1.3 | 遍历函数体，创建 scope.scope 块 | 连续同类型 op 聚拢，类型切换时关闭旧 scope 开新 scope |
| 1.4 | 标量 op 归属 | arith/index_cast 等根据 use chain 归属到下一个 scope |
| 1.5 | 标注 stage_id | 同类型 scope 从 0 开始递增编号 |
| 1.6 | 更新 module/func 属性 | `hivm.module_core_type` → MIX，`hivm.func_core_type` → MIX |

### Op 类型分类参考

```
Cube:  hivm.hir.mmadL1
DMA(Cube): hivm.hir.nd2nz (dst=cbuf), hivm.hir.load (dst=cbuf)
Vector: hivm.hir.vmul, vadd, vsub, vmax, vmin, vexp, vdiv,
        vreduce, vcast, vbrc
DMA(Vector): memref.copy (src=cbuf,dst=gm), hivm.hir.store
Scalar: arith.*, index_cast, memref.alloc, memref.subview, memref.reinterpret_cast
```

### 验证方法

```bash
# 单独运行 cv_annotate pass
tilelangir-opt --tilelangir-cv-annotate example_input.mlir -o stage1_output.mlir

# 人工检查或用脚本验证
```

### 检查清单

- [ ] 循环体内恰好 3 个 `scope.scope`：
  - `{tcore_type = CUBE, stage_id = 0}`（nd2nz K + mmadL1）
  - `{tcore_type = VECTOR, stage_id = 0}`（14 个 softmax op）
  - `{tcore_type = CUBE, stage_id = 1}`（nd2nz V + mmadL1）
- [ ] 循环前 2 个 scope：Cube(nd2nz Q) + Vector(vbrc ×4)
- [ ] 循环后 1 个 scope：Vector(vdiv + vcast + copy)
- [ ] alloc / arith / subview / reinterpret_cast 不在任何 scope 内
- [ ] `hivm.module_core_type = MIX`
- [ ] `hivm.func_core_type = MIX`
- [ ] IR 仍能被 `bishengir-compile` 接受（不会因为 scope.scope 报错）

### 关键风险

- `scope.scope` 的 region 内能否引用外部 SSA value？
  → 需要确认 `scope` 方言的语义：如果是 IsolatedFromAbove 则需要传参
- 标量 op 的归属可能有歧义（被多个 scope 使用）
  → 穿刺阶段：保留在 scope 外，让 scope 内部引用外部 value

---

## Stage 2：insert_workspace — 插入 Workspace

### 目标

将 scope 之间通过 UB memref 直接共享的数据改为通过 workspace (GM) 中转。
这是最复杂的一步。

### 前置条件

Stage 1 完成（IR 中有正确的 scope.scope 标注）

### 输入 → 输出

- **输入**：Stage 1 的输出（scope 标注后的 IR）
- **输出**：scope 间的数据传递全部通过 workspace view

### 分步实现

| # | 步骤 | 复杂度 | 说明 |
|---|------|--------|------|
| 2.1 | 分析跨 scope 数据依赖 | 中 | 遍历每个 scope 的 operands，找到定义在其他 scope 中的 value |
| 2.2 | 计算 workspace 布局 | 低 | 根据跨 scope value 的类型/shape 计算每个 sub-region 的大小和偏移 |
| 2.3 | 创建 per-block workspace view | 低 | `affine.apply` + `memref.view`，偏移 = `block_idx × per_block_size` |
| 2.4 | C→V 通道：插入 fixpipe | 高 | mmadL1 输出改为 local buffer + `hivm.hir.fixpipe` 写 workspace |
| 2.5 | V→C 通道：插入 store/load | 中 | Vector 用 `hivm.hir.store` 写 workspace，Cube 用 `hivm.hir.load` 读 |
| 2.6 | acc_o 累加拆分 | 高 | mmadL1(initC=false) → mmadL1(initC=true) + fresh buf + fixpipe；Vector 侧 load + vadd |
| 2.7 | 生成 host 端回调 | 低 | 创建 `@xxx_infer_workspace_shape_function` 返回 per-block size |
| 2.8 | 消除旧的跨 scope 引用 | 中 | 确保 scope 间不再直接共享 UB memref |

### 建议：先处理简单通道，再处理 acc_o

**第一轮（2.1-2.3 + 2.5）**：只处理 V0→C1 的 scores_cast 通道（最简单，单向 store/load）
**第二轮（2.4）**：处理 C0→V0 的 scores 通道（需要 fixpipe）
**第三轮（2.6）**：处理 C1→V0 的 acc_o 通道（需要累加拆分）

### 验证方法

```bash
tilelangir-opt --tilelangir-cv-annotate --tilelangir-insert-workspace \
  example_input.mlir -o stage2_output.mlir
```

### 检查清单

- [ ] 3 个 workspace view 被正确创建：
  - `memref<64x64xf32>` (scores, 偏移 0)
  - `memref<64x64xf16>` (scores_cast, 偏移 16384)
  - `memref<64x128xf32>` (acc_o, 偏移 24576)
- [ ] 偏移使用 `block_idx * 57344` 计算（per-block）
- [ ] C0 scope 内：mmadL1 输出到 local buffer → `fixpipe {enable_nz2nd}` → workspace
- [ ] V0 scope 内：`hivm.hir.load` ← workspace (scores)，`hivm.hir.store` → workspace (scores_cast)
- [ ] C1 scope 内：`hivm.hir.load` ← workspace (scores_cast)，mmadL1 → fresh buffer → `fixpipe` → workspace (acc_o)
- [ ] C1 的 mmadL1 变为 `initC=true`（`%true` 参数）
- [ ] V0 scope 内新增：`hivm.hir.load` ← workspace (acc_o) + `hivm.hir.vadd` 累加
- [ ] host 回调函数存在且返回 57344
- [ ] scope 之间不再有直接的 UB memref 共享引用

### 关键风险和实战教训

- `fixpipe` 的操作数格式和参数需要正确——参考 `example-reference.mlir` 中的写法
- acc_o 累加拆分改变了算法语义（从 Cube 侧累加变为 Vector 侧累加），需要确保数值正确
- **workspace 参数必须预标注 `#hivm.address_space<gm>`**（实战验证：
  不标注会导致 InferHIVMMemScope pass 失败）
- **必须使用 `memref.view` 创建 typed view**（实战验证：
  `memref.reinterpret_cast` 不支持元素类型变更，`memref_ext.alloc_workspace` 不被 InferHIVMMemScope 识别）
- **UB alloc 应放在循环体内（loop-scoped）**（实战验证：
  提升到函数顶部可能导致 PlanMemory pass 异常）
- 参考文件：`memref/vc_output_alloc_ws.mlir`（vadd，已验证 20/20 pass）、`experience.md` 第 4 节

---

## Stage 3：inject_block_sync — 插入核间同步 + AIV 核内同步

### 目标

1. 在 workspace 读写操作的正确位置插入双向 `SyncBlockSet`/`SyncBlockWait`（**核间同步**）
2. 在 AIV 的 Vector scope 中插入 `pipe_barrier[<PIPE_ALL>]`（**核内同步**）

### 为什么 AIV 需要手动核内同步（关键发现）

bishengir 的 `InjectSync` NORMAL 模式在 `hivmPostBufferizationOptimizationPipeline` 末尾运行。
其 `IRTranslator` 有严格的 op 白名单：`PointerCastOp`、`DestinationStyleOpInterface`、
`memref::LoadOp/StoreOp` 等。

我们生成的 AIV IR 中包含 `memref.alloc`（带 strided layout）和 `memref.copy`，
不在白名单中，导致 NORMAL 模式报错：
`"InjectSync Fail : Unrecognized type of Operation touches local or global buffer!"`

**当前解决方案**：在本 pass 中为 AIV scope 手动插入 `pipe_barrier[<PIPE_ALL>]`。
`BARRIERALL` 模式（`--enable-hivm-inject-barrier-all-sync`）的逻辑本质也是在每个 HIVM op 前插 pipe_barrier，
我们的方案更精细——只在必要的数据依赖点插入。

**AIC 不需要**：AIC 的 `nd2nz → mmadL1 → fixpipe` 由硬件隐式缓冲区依赖保证（L1/L0C 的读写顺序），
且 `sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>]` 绑定了 PIPE_FIX 阶段。

### 前置条件

Stage 2 完成（IR 中有 workspace 读写操作）

### 实现步骤

| # | 步骤 | 说明 |
|---|------|------|
| 3.1 | 建立 workspace 通道的数据流图 | 识别每个 workspace view 的 producer scope 和 consumer scope |
| 3.2 | 分配 Flag ID | 为每条通道分配 "就绪" + "已读完" 两个 Flag 方向，时序不重叠的可复用 |
| 3.3 | 在 producer scope 尾部插入核间 sync | fixpipe/store 之后插入 `SyncBlockSet`（就绪） |
| 3.4 | 在 consumer scope 头部插入核间 sync | load 之前插入 `SyncBlockWait`（就绪） |
| 3.5 | 在 consumer scope 读完后插入核间 sync | load 之后插入 `SyncBlockSet`（已读完） |
| 3.6 | 在 producer scope 写入前插入核间 sync | fixpipe/store 之前插入 `SyncBlockWait`（已读完） |
| 3.7 | 插入初始信号 | 循环前 consumer 侧插入初始 "已读完" SyncBlockSet |
| 3.8 | 插入循环后清理信号 | 循环后 producer 侧等待最后一轮的 "已读完" |
| 3.9 | **AIV 核内 pipe_barrier 插入** | 在 5 个位置插入（详见下方规则） |

### AIV 核内 pipe_barrier 插入规则（来自 vadd 穿刺验证）

详细规则和代码模式参见 `experience.md` 第 5.2 节。

```
AIV scope 内的 pipe_barrier 插入位置：

规则 1: sync_block_wait 之前
   → 确保前一轮 MTE3 写回 GM 已完成

规则 2: memref.copy (GM→UB) 之后、VECTOR 计算之前
   → 确保 MTE2 搬运完成，VECTOR 不读脏数据

规则 3: memref.copy (UB→GM) 之后、sync_block_set 之前
   → 确保 MTE3 搬运完成，跨核信号有效

规则 4: 尾部循环的 sync_block_wait 之前
   → 排空残留操作

规则 5: return 之前
   → 确保所有操作完成
```

### 验证方法

**vadd 用例**（首选）：
```bash
tilelangir-opt --tilelangir-cv-annotate --tilelangir-insert-workspace \
  --tilelangir-inject-block-sync vadd_input.mlir -o stage3_vadd.mlir

# 与手写参考对比
diff stage3_vadd.mlir memref/vc_output_alloc_ws.mlir
```

**flash_attn 用例**：
```bash
tilelangir-opt --tilelangir-cv-annotate --tilelangir-insert-workspace \
  --tilelangir-inject-block-sync example_input.mlir -o stage3_output.mlir

grep -c "sync_block_set" stage3_output.mlir   # 期望: ~10
grep -c "sync_block_wait" stage3_output.mlir  # 期望: ~10
grep -c "pipe_barrier" stage3_output.mlir     # 期望: >0 (仅 AIV scope 内)
```

### 检查清单

**核间同步**：
- [ ] 使用了 4 个 Flag ID (0, 1, 2, 3)
- [ ] Flag 1 被正确复用（scores 就绪、scores_cast 就绪、acc_o 就绪三处）
- [ ] 循环前有 3 个初始 SyncBlockSet（flag 0, 2, 3）
- [ ] C0 scope 内时序正确：wait(0) → fixpipe → set(1)
- [ ] V0 scope 内时序正确：wait(1) → load → set(0) → 计算 → wait(2) → store → set(1) → wait(1) → load → set(3) → vadd
- [ ] C1 scope 内时序正确：wait(1) → load → set(2) → 计算 → wait(3) → fixpipe → set(1)
- [ ] 循环后有 2 个清理 SyncBlockWait（flag 0, 3）
- [ ] sync 指令的 pipe 参数正确：fixpipe 后用 `PIPE_FIX`，load 后用 `PIPE_MTE2`，store 后用 `PIPE_MTE3`
- [ ] sync 指令的 core_type 参数正确：Cube scope 内用 `<CUBE>`，Vector scope 内用 `<VECTOR>`

**核内同步**：
- [ ] AIV scope 中的 `sync_block_wait` 前都有 `pipe_barrier[<PIPE_ALL>]`
- [ ] AIV scope 中的 `memref.copy` (GM→UB) 后、VECTOR op 前有 `pipe_barrier[<PIPE_ALL>]`
- [ ] AIV scope 中的 `memref.copy` (UB→GM) 后、`sync_block_set` 前有 `pipe_barrier[<PIPE_ALL>]`
- [ ] AIV 函数的 `return` 前有 `pipe_barrier[<PIPE_ALL>]`
- [ ] **AIC scope 中没有 pipe_barrier**

### 关键风险

- sync 时序错误会导致死锁或数据损坏，必须精确对照参考输出
- `annotation.mark` 在参考输出中出现在 fixpipe 之后、SyncBlockSet 之前，可能是 bishengir 流水线分析需要
- pipe_barrier 过多会严重影响性能（每个 barrier 等待所有流水线排空），后续可优化为细粒度 barrier

---

## Stage 4：insert_vid — 插入 VID 写入守卫

### 目标

在 Vector scope 中插入 `get_sub_block_idx`，用 `scf.if (VID==0)` 守卫所有写操作。

### 前置条件

Stage 3 完成

### 实现步骤

| # | 步骤 | 说明 |
|---|------|------|
| 4.1 | 在每个 Vector scope 头部插入 VID 获取 | `%vid = hivm.hir.get_sub_block_idx -> i64` |
| 4.2 | 计算守卫条件 | `%is_vid0 = arith.cmpi eq, %vid_idx, %c0 : index` |
| 4.3 | 包裹写 workspace 操作 | `hivm.hir.store` 用 `scf.if %is_vid0` 包裹 |
| 4.4 | 包裹写 GM 输出操作 | 最终的 `hivm.hir.store` / `memref.copy` 用 `scf.if` 包裹，加 `{limit_sub_block_id0}` |

### 验证方法

```bash
tilelangir-opt --tilelangir-cv-annotate --tilelangir-insert-workspace \
  --tilelangir-inject-block-sync --tilelangir-insert-vid \
  example_input.mlir -o stage4_output.mlir

grep "get_sub_block_idx" stage4_output.mlir  # 应出现在每个 Vector scope 中
grep "scf.if" stage4_output.mlir             # 应包裹 store 操作
```

### 检查清单

- [ ] 每个 Vector scope 头部有 `hivm.hir.get_sub_block_idx`
- [ ] 循环内 V0 的 `store scores_cast → workspace` 被 `scf.if %is_vid0` 包裹
- [ ] 循环后 Vector scope 的 `store output → GM` 被 `scf.if %is_vid0 {limit_sub_block_id0}` 包裹
- [ ] Vector 的计算操作（vmul、vreduce 等）**不被** scf.if 包裹
- [ ] Vector 的 workspace load 操作**不被** scf.if 包裹（两个 Sub-Block 都读）

### 关键风险

- 复杂度较低，主要是正确识别哪些 op 是"写操作"
- 注意 `scf.if` 创建了新的 region，需要正确处理 SSA dominance

---

## Stage 5：outline_scope + infer_task_type

### 目标

完成最后两个 pass，完成从 MIX 函数到独立 AIC/AIV 函数的拆分。

### 实现步骤

| # | 步骤 | 说明 |
|---|------|------|
| 5.1 | 实现 `outline_scope` (Clone + Filter + Flatten) | 克隆 MIX 函数 → 过滤对立 scope → 展平保留 scope |
| 5.2 | 实现 `infer_task_type` | 简单 pass：读 func_core_type 属性 → 创建 host 回调函数 |
| 5.3 | **确保 workspace 参数标注 `#hivm.address_space<gm>`** | outline 后 AIC/AIV 函数的 workspace arg 必须保持 GM 标注 |
| 5.4 | 确保 `pipe_barrier` 只保留在 AIV 函数中 | outline 的 Filter 步骤需正确处理 |

### 关键实现细节（来自 vadd 穿刺验证）

`outline_scope` 的 Clone + Filter + Flatten 策略中，**Filter 步骤**需要注意：

1. **scope 过滤**：AIC 副本删除 VECTOR scope，AIV 副本删除 CUBE scope
2. **sync 过滤**：AIC 副本删除 `tcore_type = VECTOR` 的 sync ops，反之亦然
3. **pipe_barrier 保留**：`pipe_barrier[<PIPE_ALL>]` 没有 core type 标记，
   需要根据其位置（是否在 VECTOR scope 内）决定保留或删除
4. **workspace 参数**：两个副本的 `%arg2` 都保持 `memref<?xi8, #hivm.address_space<gm>>` 类型

---

## Stage 5.5：vadd E2E 验证

### 目标

用 vadd 用例串联全部 pipeline，与手写参考 `memref/vc_output_alloc_ws.mlir` 对比验证。
vadd 比 flash_attn 简单得多（仅有 C0→V0 单通道），是验证 pipeline 正确性的最佳起点。

### 参考文件

| 文件 | 说明 |
|------|------|
| `memref/vc_output_alloc_ws.mlir` | **手写参考输出**（已通过 20/20 NPU 测试） |
| `memref/vc_output_manual_sync.mlir` | 手写参考（使用独立 workspace 参数 arg7） |
| `test_alloc_ws.py` | NPU 测试脚本 |
| `compile_vadd_hivmdma.py` | 编译脚本 |

### 验证方法

```bash
# 完整 pipeline (vadd)
tilelangir-opt \
  --tilelangir-cv-annotate \
  --tilelangir-insert-workspace \
  --tilelangir-inject-block-sync \
  --tilelangir-insert-vid \
  --tilelangir-outline-scope \
  --tilelangir-infer-task-type \
  vadd_input.mlir -o vadd_output.mlir

# 结构对比
diff <(rg "(func.func|sync_block|pipe_barrier|memref.view|memref.copy|vexp)" vadd_output.mlir) \
     <(rg "(func.func|sync_block|pipe_barrier|memref.view|memref.copy|vexp)" memref/vc_output_alloc_ws.mlir)

# 编译验证
bishengir-compile vadd_output.mlir --target=ascend910b3 \
    --enable-auto-multi-buffer=false \
    --enable-triton-kernel-compile=true \
    --enable-hivm-compile=true \
    --disable-hivm-tensor-compile=true \
    -o vadd_test

# NPU 运行验证（在有卡服务器上）
python test_alloc_ws.py
```

### 检查清单（对照 vc_output_alloc_ws.mlir）

**模块级别**：
- [ ] `hivm.module_core_type = MIX`
- [ ] 存在 AIC 函数 `@minicv_mix_aic` 和 AIV 函数 `@minicv_mix_aiv`
- [ ] 两个函数的 `%arg2` 类型为 `memref<?xi8, #hivm.address_space<gm>>`

**AIC 函数**：
- [ ] 属性 `hivm.func_core_type = AIC, hivm.part_of_mix, mix_mode = "mix"`
- [ ] workspace view: `memref.view %arg2[%c0][] → memref<4x3x64x32xf16, #hivm.address_space<gm>>`
- [ ] 含有 `hivm.hir.nd2nz` + `hivm.hir.mmadL1` + `sync_block_set/wait`
- [ ] **不含 `pipe_barrier`**
- [ ] 初始 drain 循环发送 `sync_block_set` 信号

**AIV 函数**：
- [ ] 属性 `hivm.func_core_type = AIV, hivm.part_of_mix, mix_mode = "mix"`
- [ ] workspace view: `memref.view %arg2[%c0_ws][] → memref<4x3x64x32xf16, #hivm.address_space<gm>>`
- [ ] UB alloc 在 scf.for 体内（loop-scoped）
- [ ] `memref.copy` GM→UB 后有 `pipe_barrier[<PIPE_ALL>]`
- [ ] `memref.copy` UB→GM 后有 `pipe_barrier[<PIPE_ALL>]`
- [ ] `sync_block_wait` 前有 `pipe_barrier[<PIPE_ALL>]`
- [ ] `return` 前有 `pipe_barrier[<PIPE_ALL>]`
- [ ] 尾部排空循环中有 `pipe_barrier` + `sync_block_wait`

**NPU 运行验证**：
- [ ] `bishengir-compile` 编译成功
- [ ] NPU 上 20/20 passes（0% mismatch）

---

## Stage 6：flash_attn E2E 验证

### 目标

将 vadd 验证通过的 pipeline 扩展到 flash_attn 用例，与 `example-reference.mlir` 对比。

### 检查清单（对照 example-reference.mlir）

**模块级别**：
- [ ] `hivm.module_core_type = MIX`
- [ ] 存在 4 个函数：`@flash_attention_infer_workspace_shape_function`、`@flash_attention_infer_task_type_function`、`@flash_attention_mix_aic`、`@flash_attention_mix_aiv`
- [ ] workspace shape 回调返回 57344
- [ ] task type 回调返回 32 (i8)

**AIC 函数**：
- [ ] 属性 `hivm.func_core_type = AIC`
- [ ] 含有 `hivm.hir.mmadL1` × 2（QK gemm + SV gemm）
- [ ] 含有 `hivm.hir.fixpipe` × 2（scores + acc_o → workspace）
- [ ] 含有 `sync_block_set` 和 `sync_block_wait`
- [ ] **不含 `pipe_barrier`**

**AIV 函数**：
- [ ] 属性 `hivm.func_core_type = AIV`
- [ ] 含有 softmax 相关的 Vector ops
- [ ] 含有 `hivm.hir.get_sub_block_idx` + `scf.if`（VID 写入守卫）
- [ ] 含有 workspace load/store + `pipe_barrier`（核内同步）
- [ ] 含有 `hivm.hir.vadd`（acc_o 累加）
- [ ] 含有 `sync_block_set` 和 `sync_block_wait`

**运行验证**：
- [ ] `bishengir-compile --disable-hivm-tensor-compile=true` 编译成功
- [ ] NPU 上精度验证通过

---

## Stage 7：simple_multibuffer — 循环 tiling 优化（可选）

### 目标

在 cv_annotate 之后、insert_workspace 之前，插入 loop tiling + distribution。

### 前置条件

Stage 6（flash_attn E2E）全部通过。先确保无 multibuffer 的路径完全正确，再加入优化。

### 实现步骤

| # | 步骤 | 说明 |
|---|------|------|
| 6.1 | 识别目标循环 | 包含 scope.scope 的 scf.for |
| 6.2 | loop tiling | 外层步长 = tile_factor，内层 trip count = tile_factor |
| 6.3 | loop distribution | 为每个 scope 生成独立的内层循环 |
| 6.4 | buffer 复制 | 循环内的 alloc 扩展为 tile_factor 份 |
| 6.5 | 处理跨迭代依赖 | acc_o、acc_l、acc_m 等 loop-carried state |

### 验证方法

```bash
# 对比有/无 multibuffer 的输出
tilelangir-opt --tilelangir-cv-annotate --tilelangir-simple-multibuffer \
  example_input.mlir -o stage6_output.mlir

# 检查循环结构
grep "scf.for" stage6_output.mlir
# 期望：1 个外层循环 + 每个 scope 1 个内层循环 = 4 个 scf.for
```

### 检查清单

- [ ] 原循环 `0..8` 变为外层 `0..N/tile_factor` + 内层 `0..tile_factor`
- [ ] 每个 scope (C0, V0, C1) 有独立的内层循环
- [ ] L1 buffer（K_l1, V_l1）有 tile_factor 份
- [ ] scores/scores_cast buffer 有 tile_factor 份
- [ ] 后续 Stage 2-5 的 pass 在新循环结构上仍能正确工作

### 关键风险

- 这是最复杂的变换之一：loop tiling + distribution + buffer replication
- 跨迭代的 loop-carried state（acc_o 等）需要特殊处理
- 建议穿刺阶段先用 tile_factor=1（即不做 tiling），验证骨架正确后再提高

---

## 里程碑总览

| 里程碑 | 内容 | 预估复杂度 | 核心指标 |
|--------|------|-----------|---------|
| **M0** | 基础设施搭建 | ★☆☆☆☆ | pass 空壳编译通过，pipeline 串联能调用 |
| **M1** | cv_annotate | ★★☆☆☆ | 循环体内正确的 scope 标注 |
| **M2** | insert_workspace | ★★★☆☆ | workspace view (memref.view) + GM 标注 |
| **M3** | inject_block_sync | ★★★★☆ | 核间 sync + **AIV 核内 pipe_barrier** |
| **M4** | insert_vid | ★★☆☆☆ | VID=0 写入守卫 |
| **M5** | outline_scope + infer_task_type | ★★☆☆☆ | AIC/AIV 拆分 + host 回调 |
| **M5.5** | **vadd E2E 验证** | ★★☆☆☆ | 输出与 vc_output_alloc_ws.mlir 一致，NPU 20/20 pass |
| **M6** | flash_attn E2E 验证 | ★★★☆☆ | 输出与 example-reference.mlir 一致，NPU 精度通过 |
| **M7** | simple_multibuffer | ★★★★★ | loop tiling + distribution + multi-buffer |

### 建议的优先级和节奏

```
Week 1:   M0 + M1（基础 + cv_annotate）
Week 2:   M2（insert_workspace，使用 memref.view + GM 标注）
Week 3:   M3（最关键一步：核间 sync + AIV pipe_barrier）
Week 4:   M4 + M5（vid + outline，相对简单）
Week 5:   M5.5（vadd E2E 对齐，与 vc_output_alloc_ws.mlir 对比 + NPU 验证）
Week 6:   M6（flash_attn E2E 对齐 + 修 bug）
Week 7+:  M7（simple_multibuffer，按需）
```

### 与之前计划的主要变更

| 变更 | 原计划 | 新计划 | 原因 |
|------|--------|--------|------|
| 核内同步 | 不在范围内，交给 bishengir | **Stage 3 手动插入 pipe_barrier** | InjectSync NORMAL 不认识 memref.alloc/copy |
| workspace 实现 | memref.view (plain memref) | **memref.view + 预标注 GM address_space** | InferHIVMMemScope 传播失败 |
| 验证用例 | 仅 flash_attn | **vadd 优先，flash_attn 后续** | vadd 更简单，已有参考输出 |
| E2E 验证 | 1 个 Stage | **拆为 vadd (M5.5) + flash_attn (M6)** | 分步验证降低风险 |
| M3 复杂度 | ★★★☆☆ | **★★★★☆** | 新增核内同步职责 |

---

## 调试技巧

### 1. 逐 pass 看 IR

```bash
# lower.py 中设置 dump_ir=True，会在每个 pass 后打印 IR
# 或者手动串联 pass：
tilelangir-opt --tilelangir-cv-annotate input.mlir | \
tilelangir-opt --tilelangir-insert-workspace | \
tilelangir-opt --tilelangir-inject-block-sync
```

### 2. 与参考输出对比的关键 grep

```bash
# 函数签名
grep "func.func @" output.mlir

# workspace view
grep "memref.view" output.mlir

# 同步指令
grep "sync_block" output.mlir

# fixpipe
grep "fixpipe" output.mlir

# VID
grep "get_sub_block_idx\|scf.if" output.mlir

# affine map (workspace 偏移)
grep "affine.apply\|affine_map" output.mlir
```

### 3. 常见错误排查

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| Pass 运行时 assert 失败 | scope.scope 的 region 引用了外部 value | 检查 scope 方言是否 IsolatedFromAbove |
| bishengir-compile 报错 | 输出 IR 中有未识别的 op 或属性 | 确认 scope.scope 已被 outline 消除 |
| 运行结果错误 | sync 时序不对 / acc_o 累加拆分有误 | 逐 sync 对照参考输出 |
| 死锁 | Flag ID 冲突或初始信号缺失 | 检查循环前的初始 SyncBlockSet |
| workspace 数据错误 | per-block 偏移计算错误 | 打印 affine_map 参数 |
