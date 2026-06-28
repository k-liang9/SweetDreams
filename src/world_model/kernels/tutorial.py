"""
A hands-on tutorial for writing Triton kernels, from scratch.

Run it directly to see each example execute and check against torch:

    conda activate sweetdreams
    python src/world_model/kernels/tutorial.py

Triton ships with the CUDA torch wheel. These kernels only run on a GPU; on CPU
the script will tell you and skip. Read top to bottom — each section builds on
the previous one.

=============================================================================
MENTAL MODEL (read this first)
=============================================================================

CUDA makes you think per-*thread* (one scalar at a time). Triton makes you think
per-*program*: each program instance handles a whole BLOCK of elements at once,
and you write ordinary-looking array math on those blocks. Triton compiles that
to the threads/vectorization for you.

So writing a kernel is three pieces:

  1. A grid: how many program instances to launch (like a for-loop range that
     runs in parallel). You pick this at launch time.
  2. The @triton.jit kernel: runs once per program instance. It figures out
     *which* slice of data it owns from its program id, loads that slice from
     GPU memory via pointers, computes, and stores the result back.
  3. A Python wrapper: allocates the output, computes the grid, and launches
     the kernel with `kernel[grid](...)`.

Four primitives you'll use constantly inside a kernel:
  - tl.program_id(axis)         -> which program instance am I (an int)
  - tl.arange(0, BLOCK)         -> a vector [0,1,...,BLOCK-1] for indexing
  - tl.load(ptr + offs, mask=)  -> read a block from memory (mask guards OOB)
  - tl.store(ptr + offs, x, mask=) -> write a block back

`mask` matters because BLOCK is a fixed compile-time size but your data length
usually isn't a multiple of it. The mask disables the lanes that would run off
the end of the tensor.
"""

import torch
import triton
import triton.language as tl
HAS_TRITON = torch.cuda.is_available()


# =============================================================================
# EXAMPLE 1 — vector add: c = a + b
# The "hello world" of Triton. Shows program_id, offsets, mask, load/store.
# =============================================================================

@triton.jit
def _add_kernel(
    a_ptr, b_ptr, c_ptr,   # pointers to the first element of each tensor
    n_elements,            # total length (a plain int, runtime value)
    BLOCK: tl.constexpr,   # block size; constexpr = known at compile time
):
    # Which program instance is this? With a 1D grid, program_id(0) goes
    # 0, 1, 2, ... Each one owns a contiguous chunk of BLOCK elements.
    pid = tl.program_id(axis=0)

    # The absolute indices this program is responsible for, e.g. for
    # pid=2, BLOCK=1024 -> [2048 .. 3071].
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # The last program likely runs past the end of the tensor; mask it off.
    mask = offsets < n_elements

    # Gather this block from memory. `a_ptr + offsets` is vectorized pointer
    # arithmetic: one address per lane. Masked-off lanes read nothing.
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Ordinary elementwise math on the block.
    c = a + b

    # Scatter the result back; masked-off lanes write nothing.
    tl.store(c_ptr + offsets, c, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Python wrapper: this is what the rest of your code calls."""
    a, b = a.contiguous(), b.contiguous()
    c = torch.empty_like(a)
    n = a.numel()

    # The grid is a function of the kernel's chosen constants (`meta`). Here it's
    # "ceil(n / BLOCK)" program instances along one axis. The launch syntax
    #   kernel[grid](args...)
    # is Triton's equivalent of CUDA's <<<grid, block>>>.
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _add_kernel[grid](a, b, c, n, BLOCK=1024)
    return c


# =============================================================================
# EXAMPLE 2 — row softmax: one program per row, a full reduction in registers.
# Shows 2D indexing via strides, and reductions (tl.max / tl.sum) along a block.
# This is the pattern attention/layernorm kernels are built on.
# =============================================================================

@triton.jit
def _softmax_kernel(
    x_ptr, out_ptr,
    x_row_stride, out_row_stride,   # how many elements to step to the next row
    n_cols,
    BLOCK: tl.constexpr,            # must be >= n_cols (one block covers a row)
):
    # One program instance per row. program_id(0) is the row index.
    row = tl.program_id(0)

    # Pointer to the start of this row, then offsets across its columns.
    # Strides let you index 2D (or ND) memory: element (row, col) lives at
    # base + row*row_stride + col. (Here columns are contiguous, stride 1.)
    row_start = x_ptr + row * x_row_stride
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # Load the row. Masked-off lanes get -inf so they never win the max and
    # contribute 0 after exp — the standard softmax padding trick.
    x = tl.load(row_start + cols, mask=mask, other=float('-inf'))

    # Numerically stable softmax. tl.max / tl.sum reduce across the block.
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)

    out_start = out_ptr + row * out_row_stride
    tl.store(out_start + cols, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D tensor."""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    # BLOCK must cover a whole row, so round it up to a power of two >= n_cols.
    # num_warps is a launch knob: more warps = more parallel threads per program,
    # worth raising for wide rows.
    BLOCK = triton.next_power_of_2(n_cols)
    grid = (n_rows,)  # one program per row; can be a plain tuple, not just a lambda
    _softmax_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


# =============================================================================
# AUTOTUNING (how matmul.py picks block sizes)
# =============================================================================
# For kernels where the best BLOCK sizes depend on the input shape, decorate the
# kernel with @triton.autotune instead of hardcoding constants. Triton benchmarks
# each Config the first time it sees a new `key` and caches the winner:
#
#   @triton.autotune(
#       configs=[
#           triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
#           triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=2, num_stages=4),
#       ],
#       key=['M', 'N'],   # re-tune only when these change
#   )
#   @triton.jit
#   def _kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr): ...
#
# The autotuned constants are then supplied by Triton, so you DON'T pass BLOCK_M/
# BLOCK_N at launch — but your `grid` lambda can still read them from `meta`.
#
# num_warps  : threads per program instance (parallelism within a program).
# num_stages : software-pipelining depth for the compute/load overlap in loops.


# =============================================================================
# DEBUGGING TIPS
# =============================================================================
# - tl.device_print("label", value) prints from inside a kernel.
# - Set the env var TRITON_INTERPRET=1 to run kernels on CPU in pure Python so
#   you can use pdb/print and get real tracebacks. Slow, but invaluable.
# - Always diff against a torch reference with torch.allclose (see __main__).
# - Off-by-one / garbage outputs are almost always a wrong mask or stride.


def _check(name, got, expected, atol=1e-5):
    ok = torch.allclose(got, expected, atol=atol)
    print(f'  {name:12s} {"OK" if ok else "MISMATCH"}  (max abs err {(got - expected).abs().max():.2e})')


if __name__ == '__main__':
    if not HAS_TRITON:
        print('Triton/CUDA not available here — these kernels need a GPU. '
              'Run on an L40S node inside the sweetdreams env.')
        raise SystemExit(0)

    dev = 'cuda'
    print('Example 1 — vector add')
    a = torch.randn(100_003, device=dev)   # deliberately not a multiple of BLOCK
    b = torch.randn(100_003, device=dev)
    _check('add', add(a, b), a + b)

    print('Example 2 — row softmax')
    x = torch.randn(1024, 781, device=dev)
    _check('softmax', softmax(x), torch.softmax(x, dim=-1))

    print('\nNow open matmul.py / flash_decoding.py and apply the same pattern.')
