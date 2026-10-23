# diffusion_gemma fixtures

- Conformance: `plugin.zig conformanceFixture`, a three-layer f32 pack (packed experts, self-conditioning, encoder
  scalars), built in memory.
- Proof oracle: tapes recorded from the legacy engine (`*.tape`, `engines/mlx/plugin/tape.zig`); `proof_test.zig`
  replays them on tiny random packs written by `sdk/bases/proof/fixture.zig`. Re-record with `frozen = false` and
  `MLX_SERVE_PROOF_RECORD=<repo root>` while the legacy forward exists.
- Real-model parity (legacy path): `MLX_SERVE_DIFFUSION_TRACE=1` traces against mlx-vlm.
