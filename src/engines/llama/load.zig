//! Opening a llama.cpp engine and its context from the settings that shape a load: sequences, KV quant, prefill
//! batch and MTP drafts. The scheduler arm and the engine adapter both open here.

const std = @import("std");
const model_discovery = @import("../../host/registry/discovery.zig");
const log = @import("../../host/log.zig");
const arch_llama = if (@import("build_options").llama_engine) @import("bridge.zig") else @import("stub.zig");

/// Launch settings of the embedded llama.cpp engine (`--llama-*`); the adapter's `settings_json`.
pub const Settings = struct {
    /// Sequences per model: requests decoded together in one batch, each
    /// keeping its own prompt KV for prefix reuse.
    seqs: u32 = 4,
    kv_quant: arch_llama.LlamaKvQuant = .off,
    /// Physical prefill batch; 0 = libllama's default.
    ubatch: u32 = 0,
    /// Draft tokens per MTP round when the model ships a head; 0 = off.
    mtp_drafts: u32 = 2,
};

pub fn Opened(comptime bridge: type) type {
    return struct { engine: *bridge.LlamaEngine, ctx: *bridge.LlamaContext };
}

/// Opens the GGUF at `path` on `bridge` (`bridge.zig`, or a test double) with a context of `s.seqs` sequences of
/// `ctx_size` tokens each. An `mtp-*.gguf` head beside the model wins over the trunk's own NextN heads.
pub fn open(comptime bridge: type, io: std.Io, gpa: std.mem.Allocator, path: []const u8, s: Settings, ctx_size: u32) !Opened(bridge) {
    log.info("[llama] opening engine: {s}\n", .{path});
    const mtp_path: ?[]u8 = if (s.mtp_drafts > 0) model_discovery.findLlamaMtpSidecar(io, gpa, path) else null;
    defer if (mtp_path) |p| gpa.free(p);
    if (mtp_path) |p| log.info("[llama] MTP draft head: {s}\n", .{p});
    const engine = try bridge.LlamaEngine.open(gpa, path, .{
        .mtp_path = mtp_path,
        .load_mtp = s.mtp_drafts > 0,
    });
    errdefer engine.close();
    log.info("[llama] engine ready (EOS={d}, n_vocab={d})\n", .{ engine.eosToken(), engine.nVocab() });

    const ctx = try engine.createContext(.{
        .ctx_size = @intCast(ctx_size),
        .n_seq = s.seqs,
        .type_k = s.kv_quant.ggmlType(),
        .type_v = s.kv_quant.ggmlType(),
        .ubatch = s.ubatch,
        .mtp_drafts = s.mtp_drafts,
    });
    log.info("[llama] context: {d} sequences x {d} tokens, KV {s}, MTP drafts {d}\n", .{
        ctx.seqs.len, ctx_size, s.kv_quant.label(), ctx.mtpDrafts(),
    });
    return .{ .engine = engine, .ctx = ctx };
}
