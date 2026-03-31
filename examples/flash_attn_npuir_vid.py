# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


seq_len = 512
dim = 128

# torch.npu.set_device(0)

@tilelang.jit(out_idx=[-1], target="npuir")
def online_flash_attention(block_M, block_N, dtype="float16", accum_dtype="float32"):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_o = [seq_len, dim]
    block_m = block_M
    block_n = block_N
    block_num = T.ceildiv(seq_len, block_m)
    multi_buffer = 3
    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        Output: T.Tensor(shape_o, dtype),
       
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            offset = cid * block_m

            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)

            T.copy(Q[offset, 0], Q_shared, size=[block_m, dim])
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            socres_l1 = T.alloc_shared([block_m, block_n], dtype)
            acc_o_l0c = T.alloc_fragment([block_m, dim], accum_dtype)

            scores_ub = T.alloc_shared([block_m // 2, block_n], accum_dtype)
            scores_cast = T.alloc_shared([block_m // 2, block_n], dtype)
            acc_m = T.alloc_shared([block_m // 2, 1], accum_dtype)
            acc_l = T.alloc_shared([block_m // 2, 1], accum_dtype)
            acc_o_ub = T.alloc_shared([block_m // 2, dim], accum_dtype)

            local_max = T.alloc_shared([block_m // 2,1], accum_dtype)
            local_sum = T.alloc_shared([block_m // 2,1], accum_dtype)
            new_max = T.alloc_shared([block_m // 2,1], accum_dtype)
            correction = T.alloc_shared([block_m // 2,1], accum_dtype)
            tmp = T.alloc_shared([block_m // 2, block_n], accum_dtype)
            tmp1 = T.alloc_shared([block_m // 2,1], accum_dtype)

            acc_o = T.alloc_shared([block_m // 2, dim], accum_dtype)
            scales = T.alloc_shared([block_m // 2, block_n], accum_dtype)

            scores_shared = T.alloc_shared([block_m, block_n], accum_dtype)
            acc_o_shared = T.alloc_shared([block_m, dim], accum_dtype)

            value_zero = 0
            scale = (1.0 / dim)**0.5
            value_min = -T.infinity(accum_dtype)
            # T.vbrc(value_zero, acc_o)
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(scale, scales)

            for k in T.Pipelined(T.ceildiv(seq_len, block_n)//multi_buffer, num_stages=multi_buffer):

                # cube
                T.copy(K[k * block_n, 0], K_shared, size=[block_n, dim])
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)
                T.copy(scores, scores_shared)

                # vec
                T.copy(scores_shared[vid * (block_m // 2), 0], scores_ub, size=[block_m // 2, block_n])
                T.vmul(scores_ub, scales, scores_ub)
                T.reduce_max(scores_ub, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max ,tmp1)
                T.vexp(tmp1, correction)
                T.vsub(scores_ub, new_max, tmp)
                T.vexp(tmp, scores_ub)
                T.reduce_sum(scores_ub, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                T.vmul(acc_o, correction, acc_o)
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)
                T.vcast(scores_ub, scores_cast, round_mode="rint")
                T.copy(scores_cast, socres_l1[vid * (block_m // 2), 0], size=[block_m // 2, block_n])

                # cube
                T.copy(V[k * block_n, 0], V_shared, size=[block_n, dim])
                T.gemm(socres_l1, V_shared, acc_o_l0c, initC=True)
                T.copy(acc_o_l0c, acc_o_shared)

                # vec
                T.copy(acc_o_shared[vid * (block_m // 2), 0], acc_o_ub, size=[block_m // 2, dim])
                T.vadd(acc_o, acc_o_ub, acc_o)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_m // 2, dim], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            real_m = T.min(block_m // 2, seq_len - cid * block_m - vid * block_m // 2)
            T.copy(O_cast, Output[cid * block_m + vid * block_m // 2, 0], size=[real_m, dim])

    return flash_attention

def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    # os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    kernel = online_flash_attention(64, 64)

    q = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    k = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    v = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    output = torch.randn((seq_len, dim), dtype=torch.float16).npu()

    kernel(q, k, v, output)

    scale = (1.0 / dim)**0.5
    ref_output = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v
    print("output:")
    print(output)
    print("ref_output:")
    print(ref_output)
    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
    print("All check passed.")

if __name__ == "__main__":
    main()