# glm5_next fixtures

No files: the oracle is the legacy forward (`forward.zig`'s `forwardGlm5With` / `forwardGlmBatchedDecode`) on a tiny
random pack written in memory by `fixture.zig` (`glmFiles`, seed 7: 4 layers, KDA at 2 heads of 128, one DSA layer
at the release's 64 heads of 256 over a 512 latent, a 4-token indexer pool with an 8-token budget, 8 experts top 2,
4-bit group 64). Change the writer and the proof re-derives both sides; nothing to regenerate.
Configs claimed here: `src/fixtures/model-configs/glm5_next-*.json` (mlx-community 4-bit, NVIDIA FP4, Sushi 2.4bpw).
Gate model: `ddalcu/GLM-5.3-Flash-MLX-Serve-iQ-MLX-5.1bpw`.
