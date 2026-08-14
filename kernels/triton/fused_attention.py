import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  
                    K_block_ptr, V_block_ptr,  
                    start_m, qk_scale,  
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  
                    offs_m, offs_n):
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= qk_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp(m_i - m_i_new)
        p = tl.math.exp(qk - m_i_new[:, None])
        
        acc *= alpha[:, None]
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(tl.float16), v)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  
              stride_qz, stride_qh, stride_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              Z, H, N_CTX,  
              BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q = tl.load(Q_block_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, sm_scale, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n)
    
    m_i += tl.math.log(l_i)
    acc = acc / l_i[:, None]
    
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))

def build_fused_attention_kernel(config):
    def kernel_fn(q, k, v, sm_scale):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        o = torch.empty_like(q)
        BLOCK_M = config.kwargs['BLOCK_M']
        BLOCK_N = config.kwargs['BLOCK_N']
        
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        
        _attn_fwd[grid](
            q, k, v, sm_scale, torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32), o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N,
            num_warps=config.num_warps,
            num_stages=config.num_stages
        )
        return o
    return kernel_fn

def fused_attention_input_factory(shape):
    Z, H, N_CTX, D_HEAD = shape
    def factory():
        q = torch.randn((Z, H, N_CTX, D_HEAD), device='cuda', dtype=torch.float16)
        k = torch.randn((Z, H, N_CTX, D_HEAD), device='cuda', dtype=torch.float16)
        v = torch.randn((Z, H, N_CTX, D_HEAD), device='cuda', dtype=torch.float16)
        sm_scale = 1.0 / (D_HEAD ** 0.5)
        return (q, k, v, sm_scale)
    return factory
