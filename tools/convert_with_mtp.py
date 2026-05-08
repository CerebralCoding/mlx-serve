#!/usr/bin/env python3
"""
Convert a HuggingFace Qwen3.5/3.6 MoE checkpoint to MLX while preserving
the MTP (Multi-Token Prediction) head as a sidecar `mtp.safetensors`.

The vanilla `mlx_lm.convert` path strips MTP weights (its sanitize step
filters `mtp.` keys, and `save_model` only writes `model.parameters()` which
doesn't include MTP). This script:

  1. Snapshot-downloads the HF model (or accepts a local path).
  2. Stashes the original MTP weights from the safetensors before conversion.
  3. Calls `mlx_lm.convert.convert(...)` for the main trunk — sanitize sees MTP
     present in the source and applies its `+1` shift to all main-model norm
     weights, so the saved trunk is in MLX-style "centered around 1" form.
  4. Quantizes the MTP linears (q/k/v/o, mlp.{gate,up,down}, optionally fc/head)
     to the requested bits/group_size.
  5. Applies the matching `+1` shift to MTP norm weights so the converted output
     mirrors mlx-lm's convention end-to-end. mlx-serve's binder detects this as
     `mtplx_sanitized` layout and skips the load-time +1.
  6. Writes them as `<out>/mtp.safetensors` with `language_model.mtp.*` keys.
  7. Patches `<out>/config.json` to record `mtplx_mtp_quantization` if MTP uses
     different quant params from the main trunk.

Usage:
  python3 tools/convert_with_mtp.py \\
      --hf-path Qwen/Qwen3-VL-30B-A3B-Instruct \\
      --mlx-path ~/.mlx-serve/models/Qwen3.6-30B-mtp-4bit \\
      --q-bits 4 --q-group-size 64

Notes:
  * Only tested against `qwen3_5_moe` / `qwen3_5` architectures. Other model
    families that ship MTP heads (e.g. DeepSeek-V3) would need a different
    sanitize path and aren't supported here.
  * Pass `--mtp-q-bits` / `--mtp-q-group-size` to quantize MTP linears
    differently from the main trunk (see CyanKiwi-style aggressive MTP quant).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.convert import convert as mlx_lm_convert


# Suffixes that get +1-shifted to match Convention B (mlx-lm-sanitized) for
# MTP weights. mlx-lm's own sanitize only shifts the first four; the pre-fc
# norms (`pre_fc_norm_embedding/hidden`) are MTP-specific and don't match
# any sanitize pattern, but the existing `Qwen3.6-27B-mtp` checkpoint shifts
# them anyway — empirically required for the binder's `pre_shifted=true`
# path to read them as MLX-style. `mtp.norm.weight` is intentionally excluded:
# it's already in MLX style upstream (the binder treats it as pre-shifted in
# both layouts).
NORM_SHIFT_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)

# MTP weights to quantize. `mtp.fc.weight` stays bf16 (matching the published
# MTPLX/27B-mtp convention — single dense projection). MoE expert tensors
# need a sanitize-style split before quantize (handled separately below).
LINEAR_2D_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    # Dense MLP (Qwen3.5-4B-style MTP):
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    # MoE MTP (Qwen3.6-35B-A3B-style):
    ".mlp.gate.weight",                          # router
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
    ".mlp.shared_expert_gate.weight",
)


def find_source_dir(hf_path: str) -> Path:
    """Return a local directory containing the HF safetensors. Downloads via
    huggingface_hub.snapshot_download when `hf_path` is a repo identifier."""
    p = Path(hf_path).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    print(f"[INFO] snapshot_download({hf_path!r})")
    local = snapshot_download(repo_id=hf_path)
    return Path(local)


def load_all_weights(src: Path) -> dict[str, mx.array]:
    """Load every weight from every safetensors shard in `src` into one dict."""
    weights: dict[str, mx.array] = {}
    for fn in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(fn)))
    return weights


def strip_to_lm_prefix(key: str) -> str:
    """Normalize an HF MTP key to `language_model.mtp.<rest>` form, regardless
    of whether the source nested it under `model.` or `model.language_model.`
    or stored it bare."""
    # Common upstream variants:
    #   model.mtp.X                    (Qwen3.5 base)
    #   model.language_model.mtp.X    (Qwen3-VL wrapping)
    #   language_model.model.mtp.X    (already partially sanitized)
    #   mtp.X                          (already at root)
    if key.startswith("model.language_model.mtp."):
        return "language_model." + key[len("model.language_model.") :]
    if key.startswith("language_model.model.mtp."):
        return "language_model." + key[len("language_model.model.") :]
    if key.startswith("model.mtp."):
        return "language_model." + key[len("model.") :]
    if key.startswith("mtp."):
        return "language_model." + key
    return key


def shift_if_norm(key: str, value: mx.array) -> mx.array:
    """Apply mlx-lm's +1 shift if `key` ends with one of the norm suffixes."""
    if value.ndim != 1:
        return value
    if any(key.endswith(sfx) for sfx in NORM_SHIFT_SUFFIXES):
        return value + 1.0
    return value


def quantize_linear(
    weight: mx.array, bits: int, group_size: int, mode: str = "affine"
) -> tuple[mx.array, mx.array, mx.array]:
    """Return (packed_weight, scales, biases) for `weight` quantized to MLX
    affine format. Mirrors what `nn.quantize` does internally per linear."""
    return mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)


def prepare_mtp_weights(
    raw: dict[str, mx.array],
    bits: int,
    group_size: int,
) -> dict[str, mx.array]:
    """Take raw HF MTP weights, normalize keys to Convention B, +1-shift norms,
    and quantize the listed linear projections. fc.weight stays bf16. MoE
    expert tensors get split & renamed to match mlx-lm's sanitize convention
    (`experts.gate_up_proj` → `switch_mlp.{gate,up}_proj.weight`,
    `experts.down_proj` → `switch_mlp.down_proj.weight`) before quantize."""
    out: dict[str, mx.array] = {}

    # First pass: collect MTP keys, drop everything else.
    mtp_keys = [k for k in raw if ".mtp." in k or k.startswith("mtp.") or "model.mtp" in k]
    if not mtp_keys:
        raise SystemExit("[ERR] No MTP weights found in source — config may declare them but the safetensors don't ship them.")

    # Index MoE expert tensors first so we can split/rename them in one place.
    # gate_up_proj has shape [E, 2*I, H]; split along axis -2 into gate+up.
    # down_proj has shape [E, H, I]; rename only.
    n_quantized = 0
    handled: set[str] = set()
    for key in list(mtp_keys):
        if key.endswith(".mlp.experts.gate_up_proj"):
            base = strip_to_lm_prefix(key)[: -len(".experts.gate_up_proj")]
            v = raw[key]
            assert v.ndim == 3 and v.shape[-2] % 2 == 0, f"unexpected gate_up_proj shape {list(v.shape)}"
            mid = v.shape[-2] // 2
            gate = v[..., :mid, :]
            up = v[..., mid:, :]
            for name, t in (("gate_proj", gate), ("up_proj", up)):
                wq, sc, bi = quantize_linear(t, bits=bits, group_size=group_size)
                out[f"{base}.switch_mlp.{name}.weight"] = wq
                out[f"{base}.switch_mlp.{name}.scales"] = sc
                out[f"{base}.switch_mlp.{name}.biases"] = bi
            n_quantized += 2
            handled.add(key)
        elif key.endswith(".mlp.experts.down_proj"):
            base = strip_to_lm_prefix(key)[: -len(".experts.down_proj")]
            v = raw[key]
            assert v.ndim == 3, f"unexpected down_proj shape {list(v.shape)}"
            wq, sc, bi = quantize_linear(v, bits=bits, group_size=group_size)
            out[f"{base}.switch_mlp.down_proj.weight"] = wq
            out[f"{base}.switch_mlp.down_proj.scales"] = sc
            out[f"{base}.switch_mlp.down_proj.biases"] = bi
            n_quantized += 1
            handled.add(key)

    # Second pass: regular weights — quantize listed 2D linears, pass through
    # everything else (with +1 shift for relevant norms).
    for key in mtp_keys:
        if key in handled:
            continue
        v = raw[key]
        new_key = strip_to_lm_prefix(key)
        is_linear_2d = (
            v.ndim == 2 and any(new_key.endswith(sfx) for sfx in LINEAR_2D_SUFFIXES)
        )
        if is_linear_2d:
            base = new_key[: -len(".weight")]
            wq, sc, bi = quantize_linear(v, bits=bits, group_size=group_size)
            out[base + ".weight"] = wq
            out[base + ".scales"] = sc
            out[base + ".biases"] = bi
            n_quantized += 1
        else:
            out[new_key] = shift_if_norm(new_key, v)

    print(f"[INFO] Prepared {len(out)} MTP tensors ({n_quantized} quantized linears, "
          f"{len(handled)} MoE-expert keys split)")
    return out


def update_config_for_mtp(
    out_dir: Path,
    main_bits: int,
    main_group: int,
    mtp_bits: int,
    mtp_group: int,
) -> None:
    """If MTP quant differs from main trunk's, record it in config.json under
    `mtplx_mtp_quantization` so mlx-serve's loader pins the right group_size."""
    if main_bits == mtp_bits and main_group == mtp_group:
        return
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["mtplx_mtp_quantization"] = {
        "bits": mtp_bits,
        "group_size": mtp_group,
        "mode": "affine",
        "policy": "convert_with_mtp.py",
        "prequantized": True,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[INFO] Recorded mtplx_mtp_quantization (bits={mtp_bits}, group={mtp_group}) in config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf-path", required=True, help="HF repo id OR local snapshot dir")
    ap.add_argument("--mlx-path", required=True, help="Output directory")
    ap.add_argument("--q-bits", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--q-mode", default="affine")
    ap.add_argument("--mtp-q-bits", type=int, default=None, help="Override bits for MTP linears (defaults to --q-bits)")
    ap.add_argument("--mtp-q-group-size", type=int, default=None, help="Override group_size for MTP linears (defaults to --q-group-size)")
    args = ap.parse_args()

    out = Path(args.mlx_path).expanduser()
    if out.exists():
        print(f"[ERR] {out} already exists. Pass a fresh path or rm it first.", file=sys.stderr)
        return 2

    src = find_source_dir(args.hf_path)
    print(f"[INFO] source: {src}")

    # Stash raw MTP before mlx-lm convert (which would drop them on save).
    raw = load_all_weights(src)
    mtp_q_bits = args.mtp_q_bits if args.mtp_q_bits is not None else args.q_bits
    mtp_q_group = args.mtp_q_group_size if args.mtp_q_group_size is not None else args.q_group_size
    prepared = prepare_mtp_weights(raw, bits=mtp_q_bits, group_size=mtp_q_group)
    del raw  # free before convert() spins up its own model

    # Run vanilla mlx-lm convert for the main trunk. Sanitize will detect MTP
    # presence in the original weights (we still pass `src` not `prepared`)
    # and +1-shift main-model norms accordingly. MTP keys are stripped at save
    # time — that's fine, we re-attach our prepared sidecar next.
    print("[INFO] running mlx_lm.convert for main trunk")
    mlx_lm_convert(
        hf_path=str(src),
        mlx_path=str(out),
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        q_mode=args.q_mode,
    )

    # Write the sidecar MTP file.
    mtp_path = out / "mtp.safetensors"
    mx.save_safetensors(str(mtp_path), prepared, metadata={"format": "mlx"})
    print(f"[INFO] wrote {mtp_path}")

    # Record per-MTP quant override in config.json if it differs from main.
    update_config_for_mtp(out, args.q_bits, args.q_group_size, mtp_q_bits, mtp_q_group)

    print("[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
