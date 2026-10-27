//! Inference-thread admission and execution of native fine-tuning jobs.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const types = @import("../../../host/finetune/types.zig");
const model = @import("../../../sdk/train/model.zig");
const LoadedModel = @import("../../../host/registry/registry.zig").LoadedModel;
const Scheduler = @import("../scheduler.zig").Scheduler;

pub fn frozen(lm: *const LoadedModel) !model.Frozen {
    if (lm.engine != null) return error.TrainingUnsupportedModel;
    const cfg = lm.config orelse return error.TrainingUnsupportedModel;
    const binding = cfg.plugin orelse return error.TrainingUnsupportedModel;
    if (binding.outside or cfg.isMoe() or !@import("../../../sdk/weights/lora_config.zig").supported(cfg.model_type)) return error.TrainingUnsupportedModel;
    const xfm = lm.transformer orelse return error.TrainingUnsupportedModel;
    if (@import("builtin").os.tag != .macos or !mlx.streamOnGpuDevice(xfm.s)) return error.TrainingRequiresMetal;
    const p = xfm.plugin orelse return error.TrainingUnsupportedModel;
    const get = p.table.native_train orelse return error.TrainingUnsupportedModel;
    var result: model.Frozen = undefined;
    const verdict = get(p.model, &result);
    if (!verdict.isOk()) {
        if (@import("std").mem.eql(u8, verdict.why.slice(), "TrainingRequiresFourBitOrBfloat16")) return error.TrainingRequiresFourBitOrBfloat16;
        return error.TrainingUnsupportedModel;
    }
    return result;
}

pub fn plan(sch: *Scheduler, lm: *LoadedModel, out: *types.Plan) !void {
    const base = try frozen(lm);
    if (lm.config.?.plugin.?.overlay_sha256 != null) return error.TrainingRequiresOriginalModel;
    const targets = try base.targets(sch.allocator);
    defer sch.allocator.free(targets);
    out.reserve_bytes = try @import("../../../sdk/train/memory.zig").reserve(base, targets, 8, 512);
    var merge_scratch: u64 = 0;
    for (targets) |t| merge_scratch = @max(merge_scratch, @as(u64, t.input) * t.output * 16);
    // Verification reloads a merged copy; raw and prepared weights can overlap at load.
    const copies = std.math.mul(u64, lm.bytes_resident, 2) catch return error.TrainingMemoryOverflow;
    const verification = std.math.add(u64, copies, merge_scratch) catch return error.TrainingMemoryOverflow;
    out.reserve_bytes = std.math.add(u64, out.reserve_bytes, verification) catch return error.TrainingMemoryOverflow;
    var active: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    const ceiling = @import("../../../host/registry/memory_ceiling.zig").currentGpuMemoryCeiling(lm.config, active);
    out.available_bytes = ceiling -| active -| sch.promise_ledger.outstanding(active);
    if (out.reserve_bytes > out.available_bytes) return error.TrainingDoesNotFit;
}

pub fn run(sch: *Scheduler, lm: *LoadedModel, input: types.Run) !void {
    var bill: types.Plan = .{};
    try plan(sch, lm, &bill);
    const base = try frozen(lm);
    const cfg = lm.config.?;
    const fingerprint = cfg.plugin.?.source_sha256 orelse return error.LoraBaseFingerprintRequired;
    var teacher: @import("teacher.zig").Teacher = .{ .a = input.a, .io = input.io, .lm = lm, .frozen = base, .observer = input.observer };
    try @import("workflow.zig").run(base, &teacher, input, fingerprint, .{});
}
