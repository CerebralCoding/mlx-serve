# Brewfile can't pin versions (and no zig@0.16 formula exists while 0.16 IS
# stable) — the floors are ENFORCED at build time instead:
#   zig  >= 0.16.0  comptime check at the top of build.zig (CI pins 0.16.0
#                   via mlugg/setup-zig in release.yml)
#   webp >= 1.6.0   build.zig verifyBrewDeps
brew "zig"
# mlx + mlx-c are NOT brew deps: pinned submodules (lib/mlx-src, lib/mlxc-src)
# built by scripts/build-mlx.sh so the NAX (M5) kernels ship enabled — the
# brew bottle is compiled at deployment target 26.0 and silently disables them.
brew "webp"
