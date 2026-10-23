# prism_hadamard_qwen35 plugin

Owns `prism_hadamard_qwen35` (prism-ml Bonsai 2 packs). Table = `plugins/qwen3_5`'s load/forward with
`Config.hadamard_block`: the dense qwen3_5 trunk, every projection read through the `hadamard` quant format.

- Numerics are the pack's: f16 activations over f16 scales, an f32 GDN state; f32 unpacked tensors narrowed to f16 at
  load (`narrowHadamardPackTables`, run by the driver too); signs stay f32.
- Rotations: each `<linear>.signs` pooled in one `rht.Registry` per model (siblings share a handle, so a projection
  reading the same activation hits the memo); bases `Linear.rht` takes the rotated arm (qmv_nax2, qmv2 with the
  -scale fast path, stock qmm); `TableEmbed` and the MTP head's embedding rotate gathered rows back. A joined group
  keeps its join only where its parts share one vector; q|k|v takes no join and no decode side copy.
- Seams (`seam.zig`): `RotAddNorm` adds, norms and rotates in one dispatch (`rht.normRotate`, block 1024, <= 16 rows),
  seeding the memo; `RotNorm.want_normed` says whether the reader also needs the norm unrotated (a GDN layer, the
  final norm). The GDN decode step fuses the out projection's rotation (`gdn_decode.step`).
- The driver gives the shell a registry over the rotated table and head (`trunk_weight` lends `side`/`block`/`flags`),
  so the engine's draft readouts rotate as on the legacy path; the MTP head is grafted at load (`mtp_graft.zig`,
  served packs too), depth 2 by default (`spec.mtp_default_depth`).
- Proofs (`proof_test.zig`, tapes in `fixtures/`): steps, rollbacks, the lookup loop, a tied pack, batched ticks and a
  batched verify, an image prompt, and the MTP head (steps, picks, rounds) bit-equal to the legacy shell.
- Not served: a DFlash drafter (its head reads the trunk's in f32, `lmHeadForeign`): a pack carrying one stays legacy;
  int8 prefill (the driver refuses it by name); routed experts (the format refuses rotated banks).
