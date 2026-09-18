# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split decode/prefill QSA indexer scoring for ROCm.

Port of the NVIDIA tree's ``nvidia/ops/qsa_indexer.py`` (upstream #54513
"Separate prefill and decode paths for QSA indexer" + #54915 "Compact indexer
logits workspace") onto the AMD tree.  Differences from the NVIDIA file:

* bf16 K/V only (no fp8 cache), f32 accumulation, and the score keeps the
  existing AMD ``/ sqrt(head_dim)`` divisor so logits stay comparable with
  ``qsa.qsa_mqa_paged`` (top-k is scale invariant either way).
* RDNA3 (gfx1100, wave32) WMMA only accepts 16-bit ``tl.dot`` operands and
  tile dims >= 16 (see FLA ``chunk_scaled_dot_kkt.py`` ``_CAST_DOT_TO_K_DTYPE``).
  Both kernels dot the bf16 tiles straight from memory; the head axis of the
  query tile is padded so the dot's N dim is at least ``_MIN_DOT_N``
  (Flash-Next has 4 indexer heads: a plain decode row would otherwise be
  N = 1 * 4).  Padded heads are zero-loaded and masked out of the ReLU-sum.
* The ROCm top-k stays ``ops.top_k_per_row_decode`` (no cooperative /
  persistent top-k on ROCm); it only reads the first ``visible_blocks[row]``
  columns, so columns beyond the visible prefix are never written (as
  upstream).
* Expansion reuses ``qsa.expand_qsa_block_indices_cuda`` (AMD output layout:
  ``[rows, token_topk + compress_ratio - 1]``, no trailing count column).

Launch geometry is parameterized by the module constants below; every one
has an env override for sweeps (read once at import).  Defaults = upstream.
"""

from __future__ import annotations

import math
import os

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import HAS_TRITON, tl, triton

from .qsa import (
    _LOGITS_WORKSPACE_BYTES,
    _QSA_COMPACT_LOGITS,
    expand_qsa_block_indices_cuda,
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_opt_int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


# Minimum tl.dot M/N/K on RDNA3 WMMA (and Triton's general floor).
_MIN_DOT_N = 16

# --- decode (uniform, request-major) kernel -------------------------------
# Upstream GB300 tuning: BLOCK_N=64, STAGES=2, num_warps=2 (bf16 cache), and
# TILES_PER_PROG from the _decode_tiles_per_program ladder.  On gfx1100 a
# "warp" is one wave32.
_DECODE_BLOCK_N = _env_int("VLLM_QSA_IDX_DECODE_BLOCK_N", 64)
_DECODE_STAGES = _env_int("VLLM_QSA_IDX_DECODE_STAGES", 2)
_DECODE_WARPS = _env_int("VLLM_QSA_IDX_DECODE_WARPS", 2)
# None -> upstream ladder; an int pins TILES_PER_PROG for every batch size.
_DECODE_TILES_OVERRIDE = _env_opt_int("VLLM_QSA_IDX_DECODE_TILES")

# --- prefill (row-tiled, packed) kernel ------------------------------------
# Upstream GB300 tuning (bf16 cache): TILE_R=64, BLOCK_N=64, K_TILES=16,
# STAGES=2, num_warps=4.  The f32 dot accumulator is
# BLOCK_N x TILE_R x next_pow2(heads) = 64 x 256 for Flash-Next, i.e. 128
# VGPRs/lane at 4 waves -- heavy for RDNA3, so TILE_R / warps are the first
# knobs to sweep.
_PREFILL_TILE_R = _env_int("VLLM_QSA_IDX_PREFILL_TILE_R", 64)
_PREFILL_BLOCK_N = _env_int("VLLM_QSA_IDX_PREFILL_BLOCK_N", 64)
_PREFILL_K_TILES = _env_int("VLLM_QSA_IDX_PREFILL_K_TILES", 16)
_PREFILL_STAGES = _env_int("VLLM_QSA_IDX_PREFILL_STAGES", 2)
_PREFILL_WARPS = _env_int("VLLM_QSA_IDX_PREFILL_WARPS", 4)


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_columns,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MIN_DOT_N: tl.constexpr,
) -> None:
    """Score one request's DECODE_QUERY_LEN rows against TILES_PER_PROG tiles.

    Grid: (num_requests, cdiv(num_columns, BLOCK_N * TILES_PER_PROG)).
    ``q``/``visible``/``logits`` rows are request-major:
    row = request * DECODE_QUERY_LEN + query_offset.
    """
    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    # RDNA3: pad the head axis so the dot's N dim (queries x heads) >= 16.
    NUM_HEADS_PADDED: tl.constexpr = max(
        triton.next_power_of_2(NUM_HEADS), MIN_DOT_N // DECODE_QUERY_LEN_PADDED
    )
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    # [BLOCK_D, queries*heads], bf16 straight from memory (WMMA operand).
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        # bf16 x bf16 -> f32 (WMMA on gfx1100).
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2) / score_divisor
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < num_columns)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    num_columns,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    MIN_DOT_N: tl.constexpr,
) -> None:
    """Score TILE_R packed prefill rows of one request against K_TILES tiles.

    Grid: (num_prefill_requests, cdiv(max_query_len, TILE_R),
    cdiv(num_columns, BLOCK_N * K_TILES)).  Rows [query_offset,
    query_offset + num_rows) of the packed prefill slice are scored into
    ``logits`` rows [0, num_rows) (chunked logits workspace).
    """
    # RDNA3: pad the head axis so the dot's N dim (rows x heads) >= 16.
    NUM_HEADS_PADDED: tl.constexpr = max(
        triton.next_power_of_2(NUM_HEADS), MIN_DOT_N // TILE_R
    )
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    # [BLOCK_D, TILE_R*heads], bf16 straight from memory (WMMA operand).
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        # bf16 x bf16 -> f32 (WMMA on gfx1100); ReLU-sum stays in f32.
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2) / score_divisor
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=store_mask,
        )


def _decode_tiles_per_program(num_requests: int, columns: int) -> int:
    """Upstream ladder: group column tiles per program as the grid grows."""
    if _DECODE_TILES_OVERRIDE is not None:
        return max(1, _DECODE_TILES_OVERRIDE)
    programs = num_requests * triton.cdiv(columns, _DECODE_BLOCK_N)
    if programs < 16384:
        return 1
    if programs < 32768:
        return 2
    if programs < 131072:
        return 4
    return 8


def _check_layouts(
    q: torch.Tensor, k_cache: torch.Tensor, page_table: torch.Tensor
) -> None:
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires a GPU and Triton")
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if q.dtype != k_cache.dtype or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("QSA split indexer wants matching bf16/f16 Q and K cache")
    # The kernels index the head_dim axis and page-table columns with unit
    # stride (as upstream); the AMD caller tensors are contiguous there.
    if q.stride(2) != 1 or k_cache.stride(3) != 1:
        raise ValueError("QSA split indexer needs unit head_dim strides")
    if page_table.ndim != 2 or page_table.stride(1) != 1:
        raise ValueError("QSA split indexer needs a unit-stride page table")


def _score_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    decode_query_len: int,
    logits_width: int,
    score_divisor: float,
) -> torch.Tensor:
    """Logits ``[num_requests * decode_query_len, logits_width]`` (f32)."""
    num_rows = q.shape[0]
    num_requests = num_rows // decode_query_len
    logits = torch.empty((num_rows, logits_width), dtype=torch.float32, device=q.device)
    tiles_per_program = _decode_tiles_per_program(num_requests, logits_width)
    grid = (
        num_requests,
        triton.cdiv(logits_width, _DECODE_BLOCK_N * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        logits_width,
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=_DECODE_BLOCK_N,
        TILES_PER_PROG=tiles_per_program,
        STAGES=_DECODE_STAGES,
        MIN_DOT_N=_MIN_DOT_N,
        num_warps=_DECODE_WARPS,
    )
    return logits


def _score_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
    score_divisor: float,
) -> torch.Tensor:
    """Logits ``[num_queries, logits_width]`` for packed prefill rows."""
    logits = torch.empty(
        (num_queries, logits_width), dtype=torch.float32, device=q.device
    )
    grid = (
        page_table.shape[0],
        triton.cdiv(min(num_queries, max_query_len), _PREFILL_TILE_R),
        triton.cdiv(logits_width, _PREFILL_BLOCK_N * _PREFILL_K_TILES),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        logits.stride(0),
        num_queries,
        query_offset,
        page_table.shape[1],
        logits_width,
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=_PREFILL_TILE_R,
        BLOCK_N=_PREFILL_BLOCK_N,
        K_TILES=_PREFILL_K_TILES,
        STAGES=_PREFILL_STAGES,
        MIN_DOT_N=_MIN_DOT_N,
        num_warps=_PREFILL_WARPS,
    )
    return logits


def _topk_rocm(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    blocks: torch.Tensor,
) -> None:
    # 1-D lengths + next_n=1: row length = visible_blocks[row]; only that
    # prefix of each logits row is read (same contract as qsa_select_paged_tokens).
    ops.top_k_per_row_decode(
        logits,
        1,
        visible_blocks,
        blocks,
        blocks.shape[0],
        logits.stride(0),
        logits.stride(1),
        block_topk,
    )


def qsa_logits_width(
    capacity: int, max_seq_len: int | None, compress_ratio: int
) -> int:
    """Compact logits row width (upstream #54915), rounded up to 64."""
    if max_seq_len is None or not _QSA_COMPACT_LOGITS:
        return capacity
    width = triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 64) * 64
    return min(max(64, width), capacity)


def qsa_select_paged_tokens_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    *,
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_len: int,
    max_query_len: int,
    max_seq_len: int | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score, select, and expand QSA indices with split decode/prefill kernels.

    The batch is decode-first (reordered by the QSA metadata builder): rows
    ``[0, num_decode_tokens)`` belong to ``num_decodes`` requests of
    ``decode_query_len`` tokens each and are scored by the request-major
    uniform kernel; the remaining rows are packed prefill requests
    ``page_table[num_decodes:]`` / ``query_start_loc[num_decodes:]`` scored by
    the row-tiled kernel.  Both write into a logits workspace of width
    ``qsa_logits_width(...)`` and go through ``ops.top_k_per_row_decode``
    and ``expand_qsa_block_indices_cuda``.
    """
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out
    _check_layouts(q, k_cache, page_table)
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    if token_to_req.shape != (rows,) or query_positions.shape != (rows,):
        raise ValueError("QSA request mapping / positions must match query rows")
    if visible_blocks.shape != (rows,):
        raise ValueError("QSA visible_blocks must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("QSA query_start_loc must have num_requests + 1 entries")
    if num_decode_tokens < 0 or num_decode_tokens > rows:
        raise ValueError("QSA decode token count out of range")
    if num_decode_tokens and (
        decode_query_len <= 0 or num_decodes * decode_query_len != num_decode_tokens
    ):
        raise ValueError("QSA decode rows must form a uniform request batch")

    block_topk = token_topk // compress_ratio
    capacity = page_table.shape[1] * k_cache.shape[1]
    logits_width = qsa_logits_width(capacity, max_seq_len, compress_ratio)
    score_divisor = math.sqrt(q.shape[2])
    row_bytes = max(logits_width * 4, 1)

    # Decode requests occupy the leading rows (request-major, uniform length).
    if num_decode_tokens:
        requests_per_chunk = max(
            1, _LOGITS_WORKSPACE_BYTES // (row_bytes * decode_query_len)
        )
        for req_start in range(0, num_decodes, requests_per_chunk):
            req_end = min(req_start + requests_per_chunk, num_decodes)
            row_slice = slice(
                req_start * decode_query_len, req_end * decode_query_len
            )
            logits = _score_decode(
                q[row_slice],
                k_cache,
                page_table[req_start:req_end],
                visible_blocks[row_slice],
                decode_query_len,
                logits_width,
                score_divisor,
            )
            blocks = torch.empty(
                (logits.shape[0], block_topk), dtype=torch.int32, device=q.device
            )
            _topk_rocm(logits, visible_blocks[row_slice], block_topk, blocks)
            expand_qsa_block_indices_cuda(
                blocks,
                query_positions[row_slice],
                sequence_lengths,
                token_to_req[row_slice],
                compress_ratio,
                token_topk,
                out[row_slice],
            )

    # Prefill requests follow the decode rows, packed by query_start_loc.
    if num_decode_tokens < rows:
        prefill_slice = slice(num_decode_tokens, rows)
        prefill_rows = rows - num_decode_tokens
        q_prefill = q[prefill_slice]
        prefill_page_table = page_table[num_decodes:]
        prefill_query_start_loc = query_start_loc[num_decodes:]
        prefill_visible = visible_blocks[prefill_slice]
        prefill_positions = query_positions[prefill_slice]
        prefill_token_to_req = token_to_req[prefill_slice]
        prefill_out = out[prefill_slice]
        rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // row_bytes)
        blocks_buffer = torch.empty(
            (min(prefill_rows, rows_per_chunk), block_topk),
            dtype=torch.int32,
            device=q.device,
        )
        for row_start in range(0, prefill_rows, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, prefill_rows)
            row_slice = slice(row_start, row_end)
            logits = _score_prefill(
                q_prefill,
                k_cache,
                prefill_page_table,
                prefill_query_start_loc,
                prefill_visible,
                max_query_len,
                logits_width,
                query_offset=row_start,
                num_queries=row_end - row_start,
                score_divisor=score_divisor,
            )
            blocks = blocks_buffer[: row_end - row_start]
            _topk_rocm(logits, prefill_visible[row_slice], block_topk, blocks)
            expand_qsa_block_indices_cuda(
                blocks,
                prefill_positions[row_slice],
                sequence_lengths,
                prefill_token_to_req[row_slice],
                compress_ratio,
                token_topk,
                prefill_out[row_slice],
            )
    return out


__all__ = [
    "qsa_logits_width",
    "qsa_select_paged_tokens_split",
]
