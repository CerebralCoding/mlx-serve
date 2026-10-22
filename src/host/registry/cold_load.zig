//! Cold-load launch flags, drafter directory resolution and the vision opt-out.

const std = @import("std");
const testing = std.testing;

const scheduler_mod = @import("../../engines/mlx/scheduler.zig");

/// Which drafter directory a COLD load should use.
///
/// `--no-drafter` is a policy — it silences every model, including one whose
/// own dir ships a sidecar `dflash.resolveInDirDrafter` would otherwise find.
/// `--drafter <path>`, by contrast, names a sidecar for the checkpoint it was
/// passed beside: handing it to whatever model is swapped in next would load a
/// mismatched assistant, so it applies only when the entry being loaded IS the
/// launch model (which happens on a reload after eviction). Every other model
/// is served by the in-dir probe.
pub fn coldLoadDrafterDir(
    no_drafter: bool,
    primary_model_dir: []const u8,
    drafter_dir: []const u8,
    entry_path: []const u8,
) []const u8 {
    if (no_drafter) return "";
    if (drafter_dir.len == 0) return "";
    if (!std.mem.eql(u8, primary_model_dir, entry_path)) return "";
    return drafter_dir;
}

/// The drafter a load uses: "" = none, null = the in-dir probe decides.
/// `--no-drafter`/`--drafter` win over the per-model `drafter` setting.
pub fn drafterFor(no_drafter: bool, explicit: []const u8, setting: ?[]const u8) ?[]const u8 {
    if (no_drafter) return "";
    if (explicit.len > 0) return explicit;
    const s = setting orelse return null;
    if (std.mem.eql(u8, s, "off")) return "";
    if (std.mem.eql(u8, s, "auto")) return null;
    return s;
}

/// Should a cold load bring up the checkpoint's vision tower?
pub fn coldLoadVision(has_vision: bool) bool {
    return has_vision and !scheduler_mod.no_vision_global;
}

// ── tests ──

test "coldLoadDrafterDir: --no-drafter wins, an explicit --drafter belongs to its OWN model" {
    // `--drafter <path>` names a sidecar for the checkpoint it was passed
    // with; handing it to whatever model gets swapped in next would load a
    // mismatched assistant. Other models are served by the in-dir probe.
    // Reloading the launch model AFTER an eviction must still get it back.
    try testing.expectEqualStrings("/d", coldLoadDrafterDir(false, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "/d", "/other"));
    // --no-drafter is a policy, not a path: it silences every model, including
    // one whose own dir ships a sidecar the in-dir probe would find.
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/m"));
    try testing.expectEqualStrings("", coldLoadDrafterDir(true, "/m", "/d", "/other"));
    // No --drafter at launch: nothing to carry, the in-dir probe decides.
    try testing.expectEqualStrings("", coldLoadDrafterDir(false, "/m", "", "/m"));
}

test "drafterFor: launch flags win, then the per-model setting, then the in-dir probe" {
    try testing.expectEqualStrings("", drafterFor(true, "/d", "/s").?);
    try testing.expectEqualStrings("/d", drafterFor(false, "/d", "off").?);
    try testing.expectEqualStrings("", drafterFor(false, "", "off").?);
    try testing.expectEqualStrings("/s", drafterFor(false, "", "/s").?);
    try testing.expectEqual(@as(?[]const u8, null), drafterFor(false, "", "auto"));
    try testing.expectEqual(@as(?[]const u8, null), drafterFor(false, "", null));
}

test "the cold-load LoadRequest re-applies EVERY retained launch setting" {
    // Three separate rounds of this bug shipped: prefix-cache, then MTP +
    // llama, then the drafter/ssd group — each time a launch flag reached
    // only `--model` and the cold path (hot switch, /v1/load-model, first
    // request naming an unloaded model) quietly used a struct default. The
    // scan is the class guard: a field retained on the Scheduler for this
    // purpose that no cold-load assignment mentions is the next round.
    // Needles are ++-split so this test's own source can't satisfy the scan.
    const src = @embedFile("../../engines/mlx/scheduler.zig");
    inline for (.{
        "kv_quant_config",           "prefix_cache_capacity",     "prefix_cache_mem_bytes",
        "prefix_cache_disk_bytes",   "ssm_checkpoint_stride",     "ssm_checkpoint_max",
        "mtp_enabled",               "mtp_head_kv_quant",         "mtp_depth",
        "llama",                     "ds4_mtp",                   "ds4_dspark",
        "ds4_ssd_streaming",         "no_drafter",                "draft_block_size",
        "draft_block_size_explicit", "ane_prefill",               "ane_chunk_resolver",
        "ane_headroom_resolver",     "prefix_cache_mem_resolver", "ane_slots",
        "ane",
    }) |field| {
        const needle = "." ++ field ++ " = self" ++ "." ++ field ++ ",";
        try testing.expect(std.mem.indexOf(u8, src, needle) != null);
    }
    // The drafter path is the one retained setting that must NOT be copied
    // straight across — it goes through the ownership rule above.
    const via_rule = "coldLoadDrafterDir(" ++ "self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path)";
    try testing.expect(std.mem.indexOf(u8, src, via_rule) != null);
    // The stale TODO that stood in for the wiring must be gone.
    const old = "Phase E will wire the load-model API" ++ " to set this.";
    try testing.expect(std.mem.indexOf(u8, src, old) == null);
}

test "every HotPrefixCache.initWithMem load site reads the CLAMPED budget, never the raw launch flag" {
    // 2026-08-30 Metal OOM: a 40 GB `--prefix-cache-mem` beside a ~70 GB pack
    // was never validated against what the weights left under the GPU ceiling;
    // the cache filled to its cap and a 143k prefill died uncatchably. The
    // budget must pass through the resolver (server.prefixCacheMemForLoad)
    // before it reaches initWithMem — and a SECOND construction site must not
    // be able to skip the clamp (the cold-load launch flags class).
    const src = @embedFile("../../engines/mlx/load.zig");
    const needle = "HotPrefixCache.initWith" ++ "Mem(";
    var found: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, src, pos, needle)) |at| : (pos = at + needle.len) {
        found += 1;
        const window = src[at..@min(at + 240, src.len)];
        try testing.expect(std.mem.indexOf(u8, window, "clamped_prefix_mem") != null);
        try testing.expect(std.mem.indexOf(u8, window, "params.prefix_cache_mem_bytes") == null);
    }
    try testing.expect(found >= 1);
}

test "coldLoadVision honors the process-wide vision opt-out" {
    scheduler_mod.no_vision_global = false;
    try std.testing.expect(coldLoadVision(true));
    try std.testing.expect(!coldLoadVision(false));
    scheduler_mod.no_vision_global = true;
    defer scheduler_mod.no_vision_global = false;
    try std.testing.expect(!coldLoadVision(true));
}
