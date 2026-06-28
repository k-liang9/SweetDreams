"""
practice matmul kernel with triton
first splits into dot products of rows & columns
within each dot product, it should 
"""

import torch
import triton
import triton.language as tl
HAS_TRITON = torch.cuda.is_available()

@triton.jit
def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        m, k, n,
        a_stride, b_stride, c_stride,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
    row_id = tl.program_id(axis=0)
    col_id = tl.program_id(axis=1)
    
    # compute offsets
    offs_m = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = col_id * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # allocate answer
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # loop over K in BLOCK_K, load in A tile & B tile, tl.dot, add into acc
    for i in range(tl.cdiv(k, BLOCK_K)):
        offs_k = i * BLOCK_K + tl.arange(0, BLOCK_K)
        a_tile = tl.load(
            a_ptr + a_stride * offs_m[:, None] + offs_k[None, :],
            mask = (offs_m < m)[:, None] & (offs_k < k)[None, :],
            other = 0.0
        )
        b_tile = tl.load(
            b_ptr + b_stride * offs_k[:, None] + offs_n[None, :],
            mask = (offs_k < k)[:, None] & (offs_n < n)[None, :],
            other = 0.0
        )
        acc += tl.dot(a_tile, b_tile)
    
    # store back in C
    tl.store(
        c_ptr + c_stride * offs_m[:, None] + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    )
    

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """python wrapper for tiled matmul"""
    assert a.shape[1] == b.shape[0]
    a, b = a.contiguous(), b.contiguous()
    c = torch.empty((a.shape[0], b.shape[1]), device='cuda', dtype=torch.float32)
    
    grid = (triton.cdiv(a.shape[0], 16), triton.cdiv(b.shape[1], 16))
    _matmul_kernel[grid](
        a, b, c,
        a.shape[0], b.shape[0], b.shape[1],
        a.stride(0), b.stride(0), c.stride(0),
        BLOCK_M=16, BLOCK_K=16, BLOCK_N=16,
    )
    return c

def _check(name, got, expected, atol=1e-5):
    ok = torch.allclose(got, expected, atol=atol)
    print(f'  {name:12s} {"OK" if ok else "MISMATCH"}  (max abs err {(got - expected).abs().max():.2e})')

if __name__ == '__main__':
    if not HAS_TRITON:
        print('Triton/CUDA not available here — these kernels need a GPU.')
        raise SystemExit(0)
    
    a = torch.randn(200, 200, device='cuda')
    b = torch.randn(200, 200, device='cuda')
    _check('matmul', matmul(a, b), a @ b)