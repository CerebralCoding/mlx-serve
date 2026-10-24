//! Opening a llama.cpp engine and its context from the settings that shape a load: sequences, KV quant, prefill
//! batch and MTP drafts.

const std = @import("std");
const model_discovery = @import("../../host/registry/discovery.zig");
const log = @import("../../sdk/util/log.zig");
const ffi = @import("ffi.zig");

/// Sequences one context holds at most.
pub const max_seqs = 64;

/// Launch settings of the embedded llama.cpp engine (`--llama-*`); the adapter's `settings_json`.
pub const Settings = struct {
    /// Sequences per model: requests decoded together in one batch, each
    /// keeping its own prompt KV for prefix reuse.
    seqs: u32 = 4,
    kv_quant: LlamaKvQuant = .off,
    /// Physical prefill batch; 0 = libllama's default.
    ubatch: u32 = 0,
    /// Draft tokens per MTP round when the model ships a head; 0 = off.
    mtp_drafts: u32 = 2,
};

/// KV quant for the llama.cpp engine. Mapped from the CLI flag onto ggml types;
/// F16 is the dense default, Q8_0 halves KV bytes, Q4_0 quarters them.
pub const LlamaKvQuant = enum(u8) {
    off, // F16 (default)
    q8, // Q8_0 (~2x compression, near-lossless on most archs)
    q4, // Q4_0 (~4x compression, some quality impact)

    pub fn fromString(s: []const u8) ?LlamaKvQuant {
        if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "f16") or std.mem.eql(u8, s, "F16")) return .off;
        if (std.mem.eql(u8, s, "8") or std.mem.eql(u8, s, "q8") or std.mem.eql(u8, s, "Q8_0") or std.mem.eql(u8, s, "q8_0")) return .q8;
        if (std.mem.eql(u8, s, "4") or std.mem.eql(u8, s, "q4") or std.mem.eql(u8, s, "Q4_0") or std.mem.eql(u8, s, "q4_0")) return .q4;
        return null;
    }

    pub fn ggmlType(self: LlamaKvQuant) i32 {
        return switch (self) {
            .off => 0, // shim treats 0 as "use libllama default"
            .q8 => ffi.GgmlType.Q8_0,
            .q4 => ffi.GgmlType.Q4_0,
        };
    }

    pub fn label(self: LlamaKvQuant) []const u8 {
        return switch (self) {
            .off => "F16",
            .q8 => "Q8_0",
            .q4 => "Q4_0",
        };
    }
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
