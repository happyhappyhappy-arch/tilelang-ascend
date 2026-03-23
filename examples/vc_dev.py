# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

M = 128
N = 128
K = 256
dtype="float16"
inner_dtype="float32"

@tilelang.jit(target="npuir")
def minicv(M, N, K, block_M, block_N, block_K):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def minicv(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((M, K), dtype),
            C: T.Tensor((K, N), dtype),
            D: T.Tensor((M, N), inner_dtype),
            workspace: T.Tensor((m_num * n_num, block_M, block_K), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            blockx = cid // n_num
            bx = blockx * block_M
            blocky = cid % n_num
            by = blocky * block_N
            D_BUF = T.alloc_shared((block_M, block_N), inner_dtype)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):

                A_BUF = T.alloc_shared((block_M//2, block_K), dtype)
                B_BUF = T.alloc_shared((block_M//2, block_K), dtype)
                B_L1 = T.alloc_shared((block_M, block_K), dtype)
                C_BUF = T.alloc_shared((block_K, block_N), dtype)

                T.copy(A[bx + (vid * block_M // 2), k * block_K], A_BUF, size=[block_M//2, block_K])
                T.copy(C[k * block_K, by], C_BUF, size=[block_K, block_N])

                T.vexp(A_BUF, B_BUF)
                T.copy(B_BUF, workspace[cid, vid * block_M // 2, 0], size=[block_M//2, block_K])
                T.copy(workspace[cid, 0, 0], B_L1)

                T.gemm(B_L1, C_BUF, D_BUF, [block_M, block_K, block_N], initC = (k==0))
                
            T.copy(D_BUF[0:block_M, 0:block_N], D[bx : bx + block_M, by : by + block_N])

    return minicv

def test_minicv():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    # os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    # In the futrue, it will be optimized to automatically derive the workspace size.
    os.environ['TILELANG_ASCEND_WORKSPACE_SIZE'] = str(M * N)
    block_M = 64
    block_N = 64
    block_K = 32
    func = minicv(M, N, K, block_M, block_N, block_K)

    v1 = torch.randn(size=[M, K], dtype=eval("torch." + dtype)).npu()
    v2 = torch.zeros(size=[M, K], dtype=eval("torch." + dtype)).npu()
    v3 = torch.randn(size=[K, N], dtype=eval("torch." + dtype)).npu()
    v4 = torch.zeros(size=[M, N], dtype=eval("torch." + inner_dtype)).npu()
    w1 = torch.zeros(size=[8, block_M, block_K], dtype=eval("torch." + dtype)).npu()

    y_ref = (torch.exp(v1.to(torch.float16))) @ (v3.to(torch.float16))
    func(v1, v2, v3, v4, w1)

    print(y_ref)
    print(v4)
    torch.testing.assert_close(v4, y_ref.to(torch.float32), rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    test_minicv()