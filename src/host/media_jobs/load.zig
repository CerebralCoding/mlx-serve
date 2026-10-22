//! Media-gen model load on the inference thread.

const mlx = @import("../../mlx/ffi.zig");
const gen_mod = @import("gen.zig");
const diarization_mod = @import("../../plugins/nemotron_diarization/diarization.zig");
const log = @import("../log.zig");

const publishHotCacheResidency = @import("../../engines/mlx/cache/budget.zig").publishHotCacheResidency;
const Scheduler = @import("../../engines/mlx/scheduler.zig").Scheduler;
const mediaBill = @import("../registry/preflight.zig").mediaBill;
const genLoadResidentBytes = @import("../registry/preflight.zig").genLoadResidentBytes;
const availForLoad = @import("../registry/preflight.zig").availForLoad;
const memInsufficientForLoad = @import("../registry/preflight.zig").memInsufficientForLoad;
const loadRequirementBytes = @import("../registry/preflight.zig").loadRequirementBytes;
const scheduler_mod = @import("../../engines/mlx/scheduler.zig");

/// Media-gen load on the inference thread. Mirrors `doLoadDs4OnInferenceThread`:
/// build the modality engine (its mlx ops bind to this thread's GPU stream from
/// t0), install it on the entry, move the stub config/tok/chat_config over, and
/// mark ready. The MLX/ds4/llama fields stay null; request handlers route
/// through the matching `image_engine`/`audio_engine`/`video_engine` slot.
pub fn doLoadGenOnInferenceThread(sch: *Scheduler, params: anytype, modality: gen_mod.Modality) !void {
    log.info("[gen] loading {s} engine: {s}\n", .{ @tagName(modality), params.model_dir });
    const entry = params.entry;

    // Media preflight, mirroring the MLX path's — a Metal OOM during engine
    // build or the first generation is uncatchable, so refuse up front. The
    // bill is per-BACKEND: a staged-residency model (minimax_h3 frees its text
    // encoder before the DiT loads) is billed its true peak, not the sum of
    // every safetensors in the dir, which refused loads that would have worked.
    if (!scheduler_mod.skip_mem_preflight) {
        // The stub config's model_type is a MODALITY static ("AudioVideo" for
        // every video backend — the per-modality-vs-per-backend class), so the
        // per-BACKEND residency estimate must re-peek the dir's actual type,
        // the same authority the engine dispatch itself uses. Caught live:
        // the H3 boot billed the 64.5 GB sum while claiming "staged".
        const peeked = gen_mod.peekModelType(sch.io, sch.allocator, params.model_dir);
        defer if (peeked) |p| sch.allocator.free(p);
        const backend_type = peeked orelse params.config.model_type;
        const peak = gen_mod.estimatePeakResidentBytes(sch.io, params.model_dir, backend_type);
        const cuda = mlx.cudaAvailable();
        const bill = mediaBill(peak, if (cuda) gen_mod.sumSafetensorsAt(sch.io, params.model_dir) else 0, cuda);
        const avail = availForLoad(sch.io, bill.ram);
        const gb = 1024.0 * 1024.0 * 1024.0;
        log.info("[preflight] media peak ~{d:.2} GB (staged residency), available {d:.2} GB\n", .{
            @as(f64, @floatFromInt(peak)) / gb,
            @as(f64, @floatFromInt(avail)) / gb,
        });
        if (memInsufficientForLoad(bill.ram, avail)) {
            log.err("Insufficient memory for this media model: needs ~{d:.1} GB free ({d:.1} GB for the model plus headroom for warmup buffers) but only {d:.1} GB is available. Unload the chat model or close other apps and retry; pass --skip-mem-preflight to override.\n", .{
                @as(f64, @floatFromInt(loadRequirementBytes(bill.ram))) / gb,
                @as(f64, @floatFromInt(bill.ram)) / gb,
                @as(f64, @floatFromInt(avail)) / gb,
            });
            return error.InsufficientMemory;
        }
        if (bill.gpu > 0) {
            var active: usize = 0;
            _ = mlx.mlx_get_active_memory(&active);
            const gpu_free = mlx.maxRecommendedWorkingSet() -| active;
            log.info("[preflight] media working set ~{d:.2} GB on the GPU, {d:.2} GB free\n", .{
                @as(f64, @floatFromInt(bill.gpu)) / gb,
                @as(f64, @floatFromInt(gpu_free)) / gb,
            });
            if (gpu_free > 0 and bill.gpu > gpu_free) {
                log.err("Insufficient GPU memory for this media model: its working set needs ~{d:.1} GB but only {d:.1} GB of GPU memory is free. Unload other models and retry; pass --skip-mem-preflight to override.\n", .{
                    @as(f64, @floatFromInt(bill.gpu)) / gb,
                    @as(f64, @floatFromInt(gpu_free)) / gb,
                });
                return error.InsufficientMemory;
            }
        }
    }

    // Build the engine FIRST. On failure the engine's own errdefer cleans up
    // its partial state, the slot stays null, and the stub config (still owned
    // by the caller, not yet installed below) is freed by the caller's
    // error path — so we must not touch `entry.config` before this succeeds.
    switch (modality) {
        .image => entry.image_engine = try gen_mod.ImageEngine.load(sch.io, sch.allocator, params.model_dir),
        .audio => entry.audio_engine = try gen_mod.AudioEngine.load(sch.io, sch.allocator, params.model_dir),
        .video => entry.video_engine = try gen_mod.VideoEngine.load(sch.io, sch.allocator, params.model_dir),
        .mesh => entry.mesh_engine = try gen_mod.MeshEngine.load(sch.io, sch.allocator, params.model_dir),
        .decision => entry.decision_engine = try gen_mod.DecisionEngine.load(sch.io, sch.allocator, params.model_dir),
        .diarization => entry.diarization_engine = try diarization_mod.Engine.load(sch.io, sch.allocator, params.model_dir),
    }

    // Install stub CPU state (infallible from here, mirroring the ds4 path).
    entry.releaseRetainedCpuState();
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    entry.weights = null;
    entry.transformer = null;
    entry.vision_encoder = null;
    entry.drafter = null;
    entry.dflash = null;
    entry.drafter_block_size = 0;
    entry.drafter_path = "";
    entry.prefix_cache = null;

    // The SAME per-backend estimator the eviction gate reserved against — see
    // `genLoadResidentBytes`. Peeked here rather than reused from the preflight
    // block above because `--skip-mem-preflight` skips that block entirely, and
    // the residency budget is not a preflight.
    const bytes_resident: u64 = blk: {
        const peeked = gen_mod.peekModelType(sch.io, sch.allocator, params.model_dir);
        defer if (peeked) |p| sch.allocator.free(p);
        const media_peak: u64 = if (peeked) |p|
            gen_mod.estimatePeakResidentBytes(sch.io, params.model_dir, p)
        else
            0;
        const on_disk: u64 = on_disk_blk: {
            if (entry.bytes_on_disk) |b| {
                if (b > 0) break :on_disk_blk b;
            }
            break :on_disk_blk gen_mod.estimateResidentBytes(sch.io, params.model_dir);
        };
        break :blk genLoadResidentBytes(media_peak, on_disk);
    };

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    sch.current_model = entry;
    sch.xfm = null;
    sch.weights = null;
    sch.vision_encoder = null;
    sch.drafter = null;
    sch.dflash = null;
    publishHotCacheResidency(sch);
    if (scheduler_mod.hot_cache_budget_invalidate) |f| f();
}
