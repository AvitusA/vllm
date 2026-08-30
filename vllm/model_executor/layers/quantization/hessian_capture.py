# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated per-layer input-Hessian capture for GPTQ recalibration.

VLLM_HESSIAN_DIR=<dir> enables it. For every AutoGPTQLinearMethod.apply on a
layer whose prefix matches VLLM_HESSIAN_PATTERN (regex, default: the
qwen4_exp side-module set), accumulate H += X^T X (fp32, on device) plus the
token count. Snapshots are written every VLLM_HESSIAN_SNAP calls per layer
(default 64) and at exit — idempotent overwrites, safe under SIGTERM.
Files: <dir>/rank<r>/<prefix>.pt  {"h": [in,in] f32, "n": int}
Row-parallel layers see TP-sharded inputs: the per-rank Hessian matches the
per-rank weight shard, which is exactly what per-shard GPTQ needs.
"""
import atexit
import os
import re

import torch

_DIR = os.environ.get("VLLM_HESSIAN_DIR", "")
_PAT = re.compile(
    os.environ.get(
        "VLLM_HESSIAN_PATTERN",
        r"(linear_attn\.(in_proj_qkvz|out_proj)"
        r"|self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj)"
        r"|shared_expert\.(gate_up_proj|down_proj)"
        r"|input_mix_weight_up"
        r"|\.ple\.(key_proj|value_proj))",
    )
)
_SNAP = int(os.environ.get("VLLM_HESSIAN_SNAP", "64"))

_state: dict[str, list] = {}  # prefix -> [H tensor, n_tokens, calls]
_rank_dir: str | None = None


def enabled() -> bool:
    return bool(_DIR)


def _ensure_rank_dir() -> str:
    global _rank_dir
    if _rank_dir is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            r = get_tensor_model_parallel_rank()
        except Exception:
            r = 0
        _rank_dir = os.path.join(_DIR, f"rank{r}")
        os.makedirs(_rank_dir, exist_ok=True)
        atexit.register(_dump_all)
    return _rank_dir


def _dump(prefix: str) -> None:
    h, n, _ = _state[prefix]
    safe = prefix.replace("/", "_")
    path = os.path.join(_ensure_rank_dir(), f"{safe}.pt")
    tmp = path + ".tmp"
    torch.save({"h": h.cpu(), "n": n}, tmp)
    os.replace(tmp, path)


def _dump_all() -> None:
    for prefix in list(_state):
        try:
            _dump(prefix)
        except Exception:
            pass


@torch.no_grad()
def maybe_capture(layer, x: torch.Tensor) -> None:
    if not _DIR:
        return
    prefix = getattr(layer, "prefix", None)
    if not prefix or not _PAT.search(prefix):
        return
    x2 = x.reshape(-1, x.shape[-1]).float()
    st = _state.get(prefix)
    if st is None:
        h = torch.zeros(
            x2.shape[-1], x2.shape[-1], dtype=torch.float32, device=x2.device
        )
        st = _state[prefix] = [h, 0, 0]
    st[0] += x2.T @ x2
    st[1] += x2.shape[0]
    st[2] += 1
    if st[2] % _SNAP == 0:
        _dump(prefix)
