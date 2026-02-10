# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import tilelang
import tilelang.language as T

from utils import assert_compile_to_kernel_o_success, get_lowered_tvm_ir

def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable rasterization for better L2 cache locality (Optional)
            # T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.npuir_clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Copy tile of B
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local)

            # relu
            T.npuir_max(C_local, 0, C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel


M = 1024  # M = T.dynamic("m") if you want to use dynamic shape
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

def test_pipelined_num_stages_tvm_ir():
    """校验 num_stages=3 时 TVM IR 中 shared buffer 为 3 份、索引为模 3。不调用 bishengir。"""
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    func = matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32")
    tvm_ir_str = get_lowered_tvm_ir(func, target="npuir")
    assert "(3, 128, 32)" in tvm_ir_str, f"expected (3, 128, 32) in TVM IR, got snippet: {tvm_ir_str[:1500]}"
    assert "(3, 32, 128)" in tvm_ir_str, f"expected (3, 32, 128) in TVM IR, got snippet: {tvm_ir_str[:1500]}"
    assert "% 3" in tvm_ir_str, (
        f"expected mod 3 index in TVM IR, got snippet: {tvm_ir_str[:1500]}"
    )
    print(tvm_ir_str)


def test_flash_attention_compile():
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    """pytest 用例：flash attn 仅编译出 kernel.o，不依赖 torch_npu。"""
    func = matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32")
    o_bytes = assert_compile_to_kernel_o_success(func)
    assert o_bytes is not None and len(o_bytes) > 0


if __name__ == "__main__":
    test_pipelined_num_stages_tvm_ir()
    # test_flash_attention_compile()