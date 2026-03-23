# TileLangIR MIX 模式编译流水线设计文档

## 概览

TileLangIR 编译流水线将 TileLang 前端生成的单体 HIVM IR 逐步变换为 Ascend NPU 的双核（AIC + AIV）可执行代码。整个流程由 8 个 MLIR pass 串联组成，在 `tilelang/engine/lower.py` 中以 MIX 模式触发：

```
canonicalize → adapt-triton-kernel
  → cv-annotate → insert-vid → analyze-cross-scope
  → simple-multibuffer → materialize-workspace
  → inject-block-sync → outline-scope → emit-host-callbacks
```

## 自定义 Op

| Op | 目的 | 关键属性/操作数 |
|---|---|---|
| `tilelangir.cv_scope` | 将 Cube/Vector 算子分组到显式 scope | `tcore_type`(1=AIC, 2=AIV), `stage_id`, `scope_id`, inputs(Variadic) |
| `tilelangir.cv_yield` | `cv_scope` 终结符 | 无操作数 |
| `tilelangir.ws_provider` | 标记 scope 内数据需写到 workspace | `data`(memref), `channel_id`, `flag`, `direction`(C2V/V2C) |
| `tilelangir.ws_consumer` | 消费来自 workspace 的数据 | `channel_id`, `flag`, `direction`; 返回 memref 结果 |

`ws_provider`/`ws_consumer` 通过 `channel_id` 配对，`flag` 用于同步信号编号。

---

## Pass 详解

### 1. tilelangir-cv-annotate

**目的**：将函数体内平坦排列的 HIVM 算子按核类型（Cube / Vector）分组，封装到 `cv_scope` 区域中。

**做法**：
- 用 `HIVMInferCoreTypeInterface` 分类每个算子：`nd2nz`、`mmadL1` 归 Cube；`vexp`、`vmul`、`vadd` 等归 Vector；`arith`、`memref.copy` 等为 neutral。
- 连续同类算子合并为一组，neutral 算子在组边界处断开。如果 neutral 算子的所有使用者都在同一组中，则可吸入该组。
- 为每组创建 `cv_scope`，外部引用的 SSA 值通过 block argument 传入。`scf.for` 的归纳变量（IV）也会被显式捕获，以便后续 pass 利用循环变量信息。
- 设置 `hivm.func_core_type = MIX`，标记函数进入混合模式。

### 2. tilelangir-insert-vid

**目的**：在 Vector scope 中插入 Sub-Block ID 保护，确保写操作仅由 `VID==0` 执行，避免多 Sub-Block 重复写入。

**做法**：
- 遍历所有 `tcore_type=2`（Vector）的 scope。
- 在 scope 开头插入 `hivm.hir.get_sub_block_idx`，计算 `%is_vid0 = (vid == 0)`。
- 对写操作（`hivm.hir.store`、向 GM 的 `memref.copy` 等）用 `scf.if %is_vid0` 包裹。
- 对 loop 外的 `memref.copy`，设置 `limit_sub_block_id0` 属性。

### 3. tilelangir-analyze-cross-scope

**目的**：分析跨 scope 的 memref 数据依赖，标注生产者-消费者关系，并插入 `ws_provider`/`ws_consumer` 通道标记。

**做法**：
- 对同一 block 内的所有 scope，构建 memref 读写依赖图：如果 scope A 写了某个 memref，scope B 读了它，则建立 A→B 的依赖边。
- 判断是否需要跨核传输：当 memref 在 core-local 地址空间（非 GM），且生产者和消费者分属不同核类型时，标记为跨核依赖。
  - 使用 `DestinationStyleOpInterface`（DPS）准确判断 HIVM 算子的读/写操作数。
- 检测循环携带的反向依赖（如 scope 2 写 → scope 1 在下一轮读），若存在双向跨核依赖则设置 `tilelangir.has_bidirectional_ws = true`。
- 基于依赖图计算 `stage_id`（DAG 最长路径）。
- 在 producer scope 末尾插入 `ws_provider`，在 consumer scope 开头插入 `ws_consumer`，以 `channel_id` 配对、`flag` 为同步信号编号。

### 4. tilelangir-simple-multibuffer

**目的**：对循环中的 scope 做多版本展开，实现软件流水线的多缓冲。

**做法**：
- 从 `scf.for` 上的 `tilelangir.num_stages` 属性获取展开因子 `tileFactor`（默认值为 1，不展开）。
- 若存在 `tilelangir.has_bidirectional_ws`，强制 `tileFactor=1`（双向依赖的循环携带状态无法安全并行）。
- 将原循环步长乘以 `tileFactor`，对每个 scope 和 alloc 创建 `tileFactor` 份副本，分配 `version_id` 和 `version_count`。
- 更新 `ws_provider`/`ws_consumer` 的 `flag = channelId * tileFactor + versionId`，确保版本间同步信号不冲突。
- 对使用 `blockIdx` 的 GM workspace（3D memref），放大第一维以容纳多版本。

### 5. tilelangir-materialize-workspace

**目的**：将抽象的 `ws_provider`/`ws_consumer` 具体化为实际的 DMA/拷贝指令。

**做法**：

分为两种处理路径：

**nd2nz 旁路**（C2V 且 data 是 nd2nz 的目标）：
- 不使用 workspace，直接将 producer scope 中的 GM 数据传递给 consumer scope。
- 在 consumer 中创建 `memref.alloc` + `memref.copy`（GM→local），替换 `ws_consumer` 的结果。

**需要 workspace 的通道**：
- 计算各 channel 的 workspace 偏移和总大小，通过函数的 workspace byte-buffer 参数（`memref<?xi8>`）+ `memref.view` 分配子区域。
- C2V provider：`hivm.hir.fixpipe`（cc → GM workspace）。
- C2V consumer：`memref.copy`（GM workspace → UB）。
- V2C provider：`memref.copy`（UB → GM workspace）。
- V2C consumer：`hivm.hir.load`（GM workspace → local，后续由 outline-scope 转为 `nd2nz`）。
- 在函数上设置 `tilelangir.workspace_bytes` 汇总总大小。

### 6. tilelangir-inject-block-sync

**目的**：在跨核 scope 之间插入 `sync_block_set`/`sync_block_wait` 同步指令，保证数据就绪后再消费。

**做法**：
- 收集 loop 内所有 scope，按 `version_id` 分组。
- 对相邻的异核 scope 对（如 CUBE→VECTOR），分配一对 flag：`data_ready` 和 `done_reading`。
- Producer scope 末尾：`sync_block_set(data_ready)`。若 scope 末尾有向 GM 的 `memref.copy`（如 workspace 写入），则延迟 set 到 copy 之后。
- Consumer scope 开头：`sync_block_wait(data_ready)`。
- Consumer scope 末尾：`sync_block_set(done_reading)`。
- Producer 下一轮开头：`sync_block_wait(done_reading)`。
- 在 Vector scope 中插入 `pipe_barrier(PIPE_ALL)`，确保 DMA 流水线完成。
- 在 loop 前初始化 `done_reading` 信号，loop 后等待最终信号。

同步 flag 编号必须 < 16（硬件限制）。

### 7. tilelangir-outline-scope

**目的**：将 MIX 入口函数拆分为两个单核函数：`_mix_aic`（Cube）和 `_mix_aiv`（Vector）。

**做法**：
- 对 MIX 函数做两份克隆。
- AIC 克隆：删除 Vector scope 和对应的 sync 指令，inline Cube scope。
- AIV 克隆：删除 Cube scope 和对应的 sync 指令，inline Vector scope。
- 地址空间重映射：
  - AIC：`address_space<zero>` → `cc`（L0C）；裸 alloc 若为 `mmadL1` 输出则映射到 `cc`，否则映射到 `cbuf`（L1）。
  - AIV：所有非 GM/UB → `ub`（Unified Buffer）。
- 后处理：
  - AIC：将非 GM→GM 的 `memref.copy` 转为 `hivm.hir.fixpipe`；将 `hivm.hir.load` 转为 `hivm.hir.nd2nz`。
  - AIV：处理含 HIVM 用户的 `memref.copy`（保留而非转为 `hivm.hir.store`）。
  - Workspace 参数提升为 GM 地址空间。
- 删除原始 MIX 函数。

### 8. tilelangir-emit-host-callbacks

**目的**：为 NPU runtime 生成 host 端回调函数。

**做法**：
- 查找所有带 `hacc.entry` 的 device 函数。
- 生成 `<entry>_infer_task_type_function`：返回核类型编码（MIX=32, AIC=20, AIV=10）。
- 生成 `<entry>_infer_workspace_shape_function`：返回 `tilelangir.workspace_bytes` 值。
- 回调函数标记为 `hacc.function_kind=HOST`。

---

## 完成情况

### 已完成

- **完整的 MIX 编译链路**：从 TileLang 前端（`T.Pipelined`）到 NPU 可执行 kernel 的端到端编译。
- **vexp (minicv) 用例**：完整通过 MIX 模式精度验证。包含 Cube（nd2nz + mmadL1）和 Vector（vexp）的跨核协作，通过前端显式 workspace 参数传递 V→C 数据，nd2nz 旁路处理 C→V 数据。
- **multi-buffer 支持**：vexp 用例 `num_stages=2` 下精度正确，`msprof` 验证有性能收益。
- **JIT 集成**：`jit_npu.py` 完成 kernel name 提取、MIX 模式编译标志、host callback 函数支持。

### 遗留问题

1. **flash_attn MIX 模式未通过**：
   - **双向依赖问题**：flash_attn 的 online softmax 算法中，`acc_o` 同时需要 Vector 侧修改（`acc_o *= correction`）和 Cube 侧累加（`mmadL1 init=false`）。硬件上 L0C 只能被 `mmadL1` 写入，无法从 GM 加载数据到 L0C，导致 V→C→cc 的累加路径无法实现。
   - **算法层面的绕过方案**：将 `initC=False` 改为 `initC=True` + Vector 侧 `vadd` 累加，可消除双向依赖，但该方案在 NPU 上仍遇到 AICore 执行异常。
   - **根因待查**：可能与 workspace `memref.view` 创建的无显式 strides 类型、fixpipe 的兼容性、或同步时序有关。

2. **multi-buffer 与循环携带状态冲突**：
   - flash_attn 的 `acc_o`、`acc_m`、`acc_l` 等循环携带状态在 Vector scope 中跨迭代修改。multi-buffer 展开会打乱执行顺序（如连续两次 `acc_o *= correction` 而非交替与 `vadd` 执行），导致精度错误。
   - 当前通过 `has_bidirectional_ws` 检测双向跨核依赖来禁用 multi-buffer，但同核内的循环携带依赖（V→V）尚未检测。

3. **`num_stages` 属性传递**：
   - 当前端设置 `num_stages=1` 时，TVM 可能不生成对应的 annotation，导致 `codegen_npuir_dev.cc` 不设置 `tilelangir.num_stages` 属性。已通过将 `SimpleMultiBuffer` 的默认 `tileFactor` 从 2 改为 1 来规避。

4. **fixpipe 类型兼容性**：
   - 通过 workspace `memref.view` 创建的 GM memref 没有显式 strides，与通过 `memref.subview` 创建的有显式 strides 的 memref 在 `bishengir-compile` 中可能走不同的 codegen 路径。vexp 用例因 nd2nz 旁路绕过了 workspace fixpipe，故不受影响。

5. **workspace**:
   - 当前的workspace用的是额外的入参，AscendNPU-IR原生的memref_ext.alloc_workspace没有支持。

6. **核内同步**:
   - 因为未知问题，位于postBufferizationOptimizationPipeline中的InjectSync不生效，目前是由我们手动插入的pipe_barrier来替代。
