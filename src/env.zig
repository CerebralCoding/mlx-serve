//! The process environment, in one place. Every variable the code reads is registered in `vars` (kind, type,
//! default, owning folder, the release that first shipped it, one line of doc) and read through the accessors
//! below, which refuse an unregistered name at compile time. Nothing else calls getenv/setenv or walks environ
//! (`zig build lint`, rule env-read).
//!
//! When a value is read: `public` and `tuning` variables once, by `init` at startup (before it, as in tests and
//! the iOS library, on every call); `kill`, `diag` and `fixture` variables on every call, so a call site that
//! latches its answer decides how often. A `diag` variable is also armed by its token in `MLX_SERVE_DIAG`
//! (`MLX_SERVE_STEP_TRACE` is `step-trace`; `token=value` gives a value), read by `init`.

const std = @import("std");
const builtin = @import("builtin");

const is_windows = builtin.os.tag == .windows;

pub const Kind = enum {
    /// Picks one of two arms of a path (a default-on path's `=0`, or an opt-in arm); retires two releases after
    /// `since`, with its losing arm.
    kill,
    /// Traces, profiles, dumps, probes and fault injection; never changes what is served.
    diag,
    /// A number or plan the code should measure instead; becomes a constant or a per-machine table, two releases
    /// after `since`.
    tuning,
    /// A user-facing or system variable; no expiry.
    public,
    /// A test input: fixture paths and env-gated tests; unset, the test skips.
    fixture,
};

pub const Type = enum { flag, int, float, text };

pub const Var = struct {
    kind: Kind,
    type: Type,
    /// What unset means: a value, a named constant, or the arm in words.
    default: []const u8,
    /// The folder under src/ that reads it.
    owner: []const u8,
    /// The release that first shipped it (`-` for the system's own variables).
    since: []const u8,
    doc: []const u8,
};

/// The registration of `name`; a name missing from `vars` is a compile error.
pub fn info(comptime name: @EnumLiteral()) Var {
    const n = @tagName(name);
    if (!@hasDecl(vars, n)) @compileError("env: " ++ n ++ " is not registered in src/env.zig");
    return @field(vars, n);
}

/// The raw value of a registered variable, or null when unset.
pub fn get(comptime name: @EnumLiteral()) ?[*:0]const u8 {
    return named(@tagName(name));
}

/// `get` for a name built at compile time (a plugin's `tag ++ "_ROUTE_TRACE"`, a helper's comptime parameter).
pub fn named(comptime n: []const u8) ?[*:0]const u8 {
    if (!@hasDecl(vars, n)) @compileError("env: " ++ n ++ " is not registered in src/env.zig");
    const z = comptime std.fmt.comptimePrint("{s}", .{n});
    switch (comptime @field(vars, n).kind) {
        .public, .tuning => if (snapshot_taken) return Slot(n).value,
        .diag => {
            if (std.c.getenv(z)) |v| return v;
            return if (snapshot_taken) diag_list.find(comptime diagToken(n)) else null;
        },
        .kill, .fixture => {},
    }
    return std.c.getenv(z);
}

pub fn span(comptime name: @EnumLiteral()) ?[:0]const u8 {
    return if (get(name)) |v| std.mem.span(v) else null;
}

/// Set and not starting with `0`: `FOO=0` is off, `FOO=` and `FOO=yes` are on.
pub fn on(comptime name: @EnumLiteral()) bool {
    comptime expectType(name, .flag);
    return onValue(get(name));
}

/// Set at all, `FOO=0` included.
pub fn present(comptime name: @EnumLiteral()) bool {
    return get(name) != null;
}

/// Exactly `1`.
pub fn one(comptime name: @EnumLiteral()) bool {
    comptime expectType(name, .flag);
    const v = get(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Base 10; `default` when unset, empty or unparseable.
pub fn int(comptime name: @EnumLiteral(), comptime T: type, default: T) T {
    comptime expectType(name, .int);
    return intValue(T, get(name), default);
}

/// A finite, non-negative number; `default` otherwise.
pub fn float(comptime name: @EnumLiteral(), default: f32) f32 {
    comptime expectType(name, .float);
    return floatValue(get(name), default);
}

/// `on`, asked once: the first answer is latched in `cache`.
pub fn onCached(cache: *?bool, comptime name: @EnumLiteral()) bool {
    if (cache.*) |v| return v;
    const v = on(name);
    cache.* = v;
    return v;
}

/// A variable whose name is chosen at run time (`--api-key-env`, a provider's key variable, a plugin's lever
/// through the Host table); read on every call, registered or not.
pub fn byName(name: [*:0]const u8) ?[*:0]const u8 {
    return std.c.getenv(name);
}

/// The process environment as the C runtime holds it, for a child process that inherits it (`posix_spawn`).
pub fn block() [*:null]?[*:0]u8 {
    return std.c.environ;
}

/// Whether any variable named `prefix…` is set and not `0` (the qwen4 profilers, `QWEN4_PROFILE_*`).
pub fn anyOnWithPrefix(prefix: []const u8) bool {
    if (comptime is_windows) return false; // no `environ` in the Windows CRT
    var i: usize = 0;
    while (std.c.environ[i]) |entry| : (i += 1) {
        const s = std.mem.span(entry);
        if (!std.mem.startsWith(u8, s, prefix)) continue;
        const eq = std.mem.indexOfScalar(u8, s, '=') orelse continue;
        if (entry[eq + 1] != '0') return true;
    }
    return false;
}

/// Reads every `public` and `tuning` variable and `MLX_SERVE_DIAG` once; `main` calls it first.
pub fn init() void {
    @setEvalBranchQuota(10_000);
    inline for (comptime std.meta.declarations(vars)) |n| {
        switch (comptime @field(vars, n).kind) {
            .public, .tuning => Slot(n).value = std.c.getenv(n),
            else => {},
        }
    }
    diag_list.load(Slot("MLX_SERVE_DIAG").value);
    snapshot_taken = true;
}

/// Sets a registered variable for this process; `get` answers with it from then on.
pub fn set(comptime name: @EnumLiteral(), value: [*:0]const u8) void {
    _ = comptime info(name);
    _ = setForeign(@tagName(name), value);
    if (snapshot_taken) init();
}

/// Sets a variable another library reads (MLX's distributed rank, ds4's kernel sources); false when the C
/// runtime refuses.
pub fn setForeign(name: [*:0]const u8, value: [*:0]const u8) bool {
    return (if (is_windows) c._putenv_s(name, value) else c.setenv(name, value, 1)) == 0;
}

pub fn unset(comptime name: @EnumLiteral()) void {
    _ = comptime info(name);
    _ = if (is_windows) c._putenv_s(@tagName(name), "") else c.unsetenv(@tagName(name));
    if (snapshot_taken) init();
}

/// Test seam: `name` holds `value` (null: unset) until `restore` puts back what was there.
pub fn override(comptime name: @EnumLiteral(), value: ?[*:0]const u8) Override(name) {
    var o: Override(name) = .{};
    if (get(name)) |v| {
        const s = std.mem.span(v);
        o.len = @min(s.len, o.buf.len - 1);
        @memcpy(o.buf[0..o.len], s[0..o.len]);
        o.buf[o.len] = 0;
        o.had = true;
    }
    if (value) |v| set(name, v) else unset(name);
    return o;
}

pub fn Override(comptime name: @EnumLiteral()) type {
    return struct {
        buf: [512]u8 = undefined,
        len: usize = 0,
        had: bool = false,

        pub fn restore(self: *const @This()) void {
            if (self.had) set(name, self.buf[0..self.len :0].ptr) else unset(name);
        }
    };
}

pub fn onValue(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != '0';
}

pub fn intValue(comptime T: type, raw: ?[*:0]const u8, default: T) T {
    const s = std.mem.span(raw orelse return default);
    if (s.len == 0) return default;
    return std.fmt.parseInt(T, s, 10) catch default;
}

pub fn floatValue(raw: ?[*:0]const u8, default: f32) f32 {
    const s = std.mem.span(raw orelse return default);
    if (s.len == 0) return default;
    const v = std.fmt.parseFloat(f32, s) catch return default;
    return if (std.math.isFinite(v) and v >= 0) v else default;
}

const c = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    extern "c" fn _putenv_s(name: [*:0]const u8, value: [*:0]const u8) c_int;
};

var snapshot_taken = false;
var diag_list: DiagList = .{};

/// One process-lifetime slot per snapshotted variable, keyed by its registration (a decl's address is unique).
fn Slot(comptime n: []const u8) type {
    return SlotOf(&@field(vars, n));
}

fn SlotOf(comptime v: *const Var) type {
    return struct {
        const of = v;
        var value: ?[*:0]const u8 = null;
    };
}

fn expectType(comptime name: @EnumLiteral(), comptime t: Type) void {
    if (info(name).type != t) @compileError("env: " ++ @tagName(name) ++ " is registered as ." ++ @tagName(info(name).type));
}

/// `MLX_SERVE_STEP_TRACE` -> `step-trace`; a name without the prefix keeps all of it (`QWEN4_GDN_CHECK` -> `qwen4-gdn-check`).
fn diagToken(comptime n: []const u8) []const u8 {
    const base = if (std.mem.startsWith(u8, n, "MLX_SERVE_")) n["MLX_SERVE_".len..] else n;
    var out: [base.len]u8 = undefined;
    for (base, 0..) |ch, i| out[i] = if (ch == '_') '-' else std.ascii.toLower(ch);
    const final = out;
    return &final;
}

/// `MLX_SERVE_DIAG`, its commas turned into NULs so each `token=value` value is a C string in place.
const DiagList = struct {
    buf: [1024]u8 = undefined,
    len: usize = 0,

    fn load(self: *DiagList, raw: ?[*:0]const u8) void {
        const s = if (raw) |v| std.mem.span(v) else "";
        self.len = @min(s.len, self.buf.len - 1);
        for (s[0..self.len], self.buf[0..self.len]) |ch, *dst| dst.* = if (ch == ',') 0 else ch;
        self.buf[self.len] = 0;
    }

    /// `token` armed bare reads as `1`; `token=value` reads as `value`.
    fn find(self: *const DiagList, token: []const u8) ?[*:0]const u8 {
        var off: usize = 0;
        while (off < self.len) {
            const item = std.mem.sliceTo(self.buf[off..self.len :0], 0);
            if (std.mem.eql(u8, item, token)) return "1";
            if (item.len > token.len and item[token.len] == '=' and std.mem.eql(u8, item[0..token.len], token))
                return self.buf[off + token.len + 1 .. self.len :0].ptr;
            off += item.len + 1;
        }
        return null;
    }
};

/// Every environment variable the code reads, one line each; sorted by kind, then name.
pub const vars = struct {
    // ── public ──
    pub const CUDA_HOME: Var = .{ .kind = .public, .type = .text, .default = "/usr/local/cuda or /opt/cuda when present", .owner = "mlx", .since = "26.10.2", .doc = "CUDA toolkit root for MLX's JIT on Linux; exported when unset" };
    pub const CUDA_PATH: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "mlx", .since = "26.10.2", .doc = "Windows CUDA toolkit: its bin goes first on PATH, and it counts as CUDA_HOME" };
    pub const HF_TOKEN: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "host/cli", .since = "26.7.1", .doc = "Hugging Face token sent by `pull`" };
    pub const HOME: Var = .{ .kind = .public, .type = .text, .default = "/tmp where a caller needs one", .owner = "host", .since = "-", .doc = "the user's home: ~/.mlx-serve (models, settings, logs, caches)" };
    pub const LTX_GEMMA_DIR: Var = .{ .kind = .public, .type = .text, .default = "the Gemma encoder under ~/.mlx-serve/models", .owner = "plugins/ltx", .since = "26.6.13", .doc = "LTX text encoder (Gemma) dir, absolute" };
    pub const MLXSERVE_DEVICE: Var = .{ .kind = .public, .type = .text, .default = "the GPU", .owner = "mlx", .since = "26.7.2", .doc = "iOS: `cpu` forces the CPU device" };
    pub const MLXSERVE_LOWMEM: Var = .{ .kind = .public, .type = .flag, .default = "by total RAM", .owner = "host/media_jobs", .since = "26.7.2", .doc = "low-memory media mode (1 on, 0 off)" };
    pub const MLX_ENABLE_TF32: Var = .{ .kind = .public, .type = .flag, .default = "on", .owner = "plugins/qwen4_exp/qsa", .since = "26.9.3", .doc = "MLX's own TF32 switch; the QSA score follows it" };
    pub const MLX_MAX_OPS_PER_BUFFER: Var = .{ .kind = .public, .type = .int, .default = "MLX's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.1", .doc = "MLX's ops per command buffer; the dsv41 cell stamps it on its receipt" };
    pub const MLX_PTX_CACHE_DIR: Var = .{ .kind = .public, .type = .text, .default = "$TMPDIR/mlx/<mlx stamp>/ptx", .owner = "mlx", .since = "26.10.3", .doc = "MLX's CUDA PTX cache; exported per MLX build when unset" };
    pub const MLX_SERVE_ANE_CACHE_CAP_GB: Var = .{ .kind = .public, .type = .int, .default = "40", .owner = "engines/ane", .since = "26.10.3", .doc = "ANE compile cache cap, GB (read by lib/ane)" };
    pub const MLX_SERVE_ANE_CACHE_DIR: Var = .{ .kind = .public, .type = .text, .default = "~/.mlx-serve/ane-cache", .owner = "engines/ane", .since = "26.10.3", .doc = "ANE compile cache root (read by lib/ane)" };
    pub const MLX_SERVE_CACHE_LIMIT: Var = .{ .kind = .public, .type = .text, .default = "by RAM", .owner = "mlx", .since = "26.7.12", .doc = "MLX allocator cache cap" };
    pub const MLX_SERVE_DIAG: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = ".", .since = "-", .doc = "comma list of diag tokens (`step-trace`, `decode-graph-dump=/path`): arms each diag variable by name" };
    pub const MLX_SERVE_GPU_CEILING_MB: Var = .{ .kind = .public, .type = .int, .default = "0 (the machine's)", .owner = "host/registry", .since = "26.9.5", .doc = "GPU memory ceiling for the admission guards, MiB" };
    pub const MLX_SERVE_INT8_QK: Var = .{ .kind = .public, .type = .flag, .default = "off", .owner = "sdk/blocks/attention", .since = "26.10.3", .doc = "int8 Q and K causal hd-256 prefill attention (lossy, NAX; --int8-qk)" };
    pub const MLX_SERVE_KERNEL_DIR: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "mlx", .since = "unreleased", .doc = "dir of .metal files that replace the built-in kernel sources by name" };
    pub const MLX_SERVE_KIT: Var = .{ .kind = .public, .type = .text, .default = "~/.mlx-serve/kit", .owner = "host/plugins", .since = "unreleased", .doc = "the plugin kit dir `mlx-serve plugin` builds against" };
    pub const MLX_SERVE_PREFILL_DECODE_SHARE: Var = .{ .kind = .public, .type = .float, .default = "0.25", .owner = "engines/mlx", .since = "26.10.1", .doc = "--prefill-decode-share: wall-time share decode keeps during a prefill (the flag wins)" };
    pub const MLX_SERVE_RAW_DUMP_FILE: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "host/api", .since = "26.7.2", .doc = "path: append every tools request's full pre-parse text (debug log level)" };
    pub const MLX_SERVE_WIRED: Var = .{ .kind = .public, .type = .text, .default = "the default policy", .owner = "mlx", .since = "26.7.12", .doc = "Metal wired-memory policy" };
    pub const PATH: Var = .{ .kind = .public, .type = .text, .default = "empty", .owner = "mlx", .since = "-", .doc = "executable search path; Windows CUDA prepends its bin" };
    pub const SHELL: Var = .{ .kind = .public, .type = .text, .default = "bash or zsh by platform", .owner = "host/launch", .since = "-", .doc = "login shell for `launch`" };
    pub const TEMP: Var = .{ .kind = .public, .type = .text, .default = "/tmp", .owner = "mlx", .since = "-", .doc = "Windows temp dir (the PTX cache root)" };
    pub const TMPDIR: Var = .{ .kind = .public, .type = .text, .default = "/tmp", .owner = "mlx", .since = "-", .doc = "temp dir (the PTX cache root)" };
    pub const USERPROFILE: Var = .{ .kind = .public, .type = .text, .default = "unset", .owner = "sdk/util", .since = "-", .doc = "Windows: the HOME when HOME is unset" };
    pub const XDG_CONFIG_HOME: Var = .{ .kind = .public, .type = .text, .default = "~/.config", .owner = "host/launch", .since = "26.9.3", .doc = "opencode CLI config root for `launch`" };

    // ── kill switches and opt-in arms ──
    pub const DSV41_BILL_VARIANT: Var = .{ .kind = .kill, .type = .text, .default = "conservative", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "DeepSeek-V4.1 streamed bill variant" };
    pub const MINIMAX_H3_GLUE_FUSED: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/minimax_h3", .since = "26.10.2", .doc = "H3 fused glue kernels (=0: the op chain)" };
    pub const MINIMAX_H3_VSA: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "H3 trained VSA sparse attention (=0: dense attention, no gate branch)" };
    pub const MLX_SERVE_ANE_FORCE: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "engines/mlx", .since = "26.8.10", .doc = "allows --ane-prefill on NAX GPUs, where it measured a loss" };
    pub const MLX_SERVE_CUDA_QMV_TC: Var = .{ .kind = .kill, .type = .flag, .default = "on where measured (sm_120)", .owner = "sdk/blocks/qmm", .since = "26.10.3", .doc = "CUDA tensor-core 4-bit GEMV for 2..8 rows" };
    pub const MLX_SERVE_DECODE_ATTN_QUANT_REQUANT: Var = .{ .kind = .kill, .type = .flag, .default = "off", .owner = "sdk/blocks/attention", .since = "26.10.2", .doc = "requantize the decode-attention side copies" };
    pub const MLX_SERVE_DSV41_ATTN_F16: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "DeepSeek-V4.1 f16 attention (=0: f32)" };
    pub const MLX_SERVE_DSV41_CED: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "DeepSeek-V4.1 CED prefill chunks (=0: full forward)" };
    pub const MLX_SERVE_DSV41_HEAD8: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "DeepSeek-V4.1 dense LM head requantized to 8-bit (=0: as shipped)" };
    pub const MLX_SERVE_DSV4_DSPARK: Var = .{ .kind = .public, .type = .flag, .default = "on (drafting; --dspark sets 1 to load the stages)", .owner = "engines/mlx/spec", .since = "26.7.12", .doc = "DeepSeek-V4 DSpark drafting (=0 off)" };
    pub const MLX_SERVE_FORCE_BATCHED: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/mlx", .since = "26.5.6", .doc = "=1: batched decode even with one slot" };
    pub const MLX_SERVE_GDN_STATE_F32: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "engines/mlx", .since = "26.10.2", .doc = "f32 GDN recurrent state (=0: bf16)" };
    pub const MLX_SERVE_GEN_YIELD: Var = .{ .kind = .kill, .type = .flag, .default = "off on CUDA, on elsewhere", .owner = "host/http", .since = "26.10.3", .doc = "chat interleaves with media jobs" };
    pub const MLX_SERVE_H3_RESIDENT: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/minimax_h3", .since = "26.10.2", .doc = "H3 DiT resident between requests" };
    pub const MLX_SERVE_HC_DECODE2: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "sdk/blocks/hc", .since = "26.10.2", .doc = "two-launch hyper-connection decode read" };
    pub const MLX_SERVE_HC_PREFILL_CUDA: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "sdk/blocks/hc", .since = "26.10.3", .doc = "fused hyper-connection prefill on CUDA" };
    pub const MLX_SERVE_MOE_AFFINE4: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/qwen4_exp", .since = "26.10.2", .doc = "qwen4 one-token affine-4 MoE kernels" };
    pub const MLX_SERVE_MOE_BANK_VIEWS: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/qwen4_exp", .since = "26.10.2", .doc = "one-expert bank views for the affine-4 MoE kernels" };
    pub const MLX_SERVE_MOE_CUDA: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "sdk/blocks/moe", .since = "26.10.3", .doc = "fused MoE decode on CUDA" };
    pub const MLX_SERVE_MOE_PREFILL_CUDA: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "sdk/blocks/moe", .since = "26.10.3", .doc = "fused MoE prefill on CUDA" };
    pub const MLX_SERVE_PLE_LAZY_SETTLE: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/qwen4_exp", .since = "26.10.2", .doc = "qwen4 GPU PLE settles its n-gram history lazily (=0 settles per forward)" };
    pub const MLX_SERVE_QSA_DEC_KERNEL: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/qwen4_exp", .since = "26.10.2", .doc = "QSA split+merge decode attention kernel" };
    pub const MLX_SERVE_QSA_MASK_KERNEL: Var = .{ .kind = .kill, .type = .flag, .default = "on", .owner = "plugins/qwen4_exp", .since = "26.10.2", .doc = "QSA one-launch row mask kernel" };
    pub const MLX_SERVE_ROUND_COST_PERSIST: Var = .{ .kind = .public, .type = .flag, .default = "off", .owner = "engines/mlx/spec", .since = "26.8.10", .doc = "=1: persist the measured round-cost table" };
    pub const MLX_SERVE_SSD_WRITE_THROUGH: Var = .{ .kind = .public, .type = .flag, .default = "writeThroughEnabledFromEnv(null)", .owner = "engines/mlx", .since = "26.9.2", .doc = "SSD tier write-through at commit" };

    // ── tuning numbers ──
    pub const DFLASH_GATE_WARMUP: Var = .{ .kind = .tuning, .type = .int, .default = "DFLASH_GATE_WARMUP", .owner = "engines/mlx/spec", .since = "26.8.6", .doc = "rounds before the DFlash yield gate decides" };
    pub const MLX_SERVE_HC_DECODE2_ROWS: Var = .{ .kind = .tuning, .type = .int, .default = "16", .owner = "sdk/blocks/hc", .since = "26.10.2", .doc = "rows the two-launch HC decode read takes (1..16)" };
    pub const MLX_SERVE_LAYA_BATCH_WINDOW_US: Var = .{ .kind = .public, .type = .int, .default = "0", .owner = "host/media_jobs", .since = "26.9.6", .doc = "Laya decision batch window, microseconds" };
    pub const MLX_SERVE_LAYA_MAX_INPUT_TOKENS: Var = .{ .kind = .public, .type = .int, .default = "DecisionLimits default", .owner = "host/media_jobs", .since = "26.9.6", .doc = "Laya input token limit" };
    pub const MLX_SERVE_LAYA_MAX_QUESTIONS: Var = .{ .kind = .public, .type = .int, .default = "DecisionLimits default", .owner = "host/media_jobs", .since = "26.9.6", .doc = "Laya questions per request limit" };
    pub const MLX_SERVE_SIMD_QMM_DECODE: Var = .{ .kind = .tuning, .type = .int, .default = "per GPU tile", .owner = "sdk/blocks/linear", .since = "26.10.2", .doc = "row floor for simd_qmm batched projections (0: off)" };

    // ── diagnostics ──
    pub const ANE_DRAFT_DEBUG: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "keeps the ANE drafter program's intermediate tensors for the parity test" };
    pub const DSV41_CELL_DECODE_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the decode profile's read-outs" };
    pub const DSV41_CELL_DRAFT_ROUTE_HIST: Var = .{ .kind = .diag, .type = .flag, .default = "on in a profile run", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell profile: the draft route histogram" };
    pub const DSV41_CELL_FENCE_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell profile: the chunk timeline alone" };
    pub const DSV41_CELL_GROUP_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell profile: grouped stage timings" };
    pub const DSV41_CELL_HOST_HEAP: Var = .{ .kind = .diag, .type = .flag, .default = "on in a profile run", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell profile: host heap samples" };
    pub const DSV41_CELL_MLX_NEWBUFFER: Var = .{ .kind = .diag, .type = .flag, .default = "on in a profile run", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell profile: MLX new-buffer counts" };
    pub const DSV41_CELL_VERIFY_GPU_TIMELINE: Var = .{ .kind = .diag, .type = .flag, .default = "on in a profile run", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell profile: the verify GPU timeline" };
    pub const DSV41_KERNELS_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel self-check: receipt path" };
    pub const DSV41_KERNEL_DECODE2_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel decode2 gate: per-output receipt path" };
    pub const DSV41_KERNEL_DRAFT_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel draft gate: per-output receipt path" };
    pub const DSV41_KERNEL_OPS_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate: per-output receipt path" };
    pub const DSV41_KERNEL_PREFILL2_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill2 gate: per-output receipt path" };
    pub const DSV41_KERNEL_PREFILL_RECEIPT: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill gate: per-output receipt path" };
    pub const DSV41_SELFCHECK_REPORT: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel self-check: log every result" };
    pub const GLM53_ROUTES: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/glm_moe_dsa/stream", .since = "26.10.3", .doc = "GLM-5.3 profile builds: write each request's decode routes under this dir" };
    pub const GLM53_WAVE_PROBE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/glm_moe_dsa/stream", .since = "26.10.3", .doc = "GLM-5.3: log each prompt layer's MLX peak by stage" };
    pub const GLM5_NEXT_GPU_TIMELINE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next profile builds: the decode GPU timeline" };
    pub const GLM5_NEXT_PREFILL_TIMELINE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "glm5_next profile builds: the prefill timeline" };
    pub const GLM5_NEXT_ROUTE_TRACE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next profile builds: route trace file prefix" };
    pub const HY3D_PAINT_DIR: Var = .{ .kind = .diag, .type = .text, .default = "the pack's paint/ dir", .owner = "host/media_jobs", .since = "26.7.2", .doc = "Hunyuan3D paint-stage model dir override (absolute, with a config.json)" };
    pub const LTX_AUDIO_DIR: Var = .{ .kind = .diag, .type = .text, .default = "the model dir", .owner = "plugins/ltx", .since = "26.7.1", .doc = "dir holding the LTX audio VAE and vocoder files" };
    pub const LTX_AUDIO_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/ltx", .since = "26.7.1", .doc = "LTX audio decoder shape trace on stderr" };
    pub const MIMO_DIAG: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo profile builds: stream source diagnostics (off | missfree)" };
    pub const MIMO_GPU_TIMELINE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo profile builds: the verify GPU timeline for the first request" };
    pub const MIMO_PREFILL_TIMELINE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo profile builds: the prefill timeline" };
    pub const MIMO_ROUTE_TRACE: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo profile builds: route trace file prefix" };
    pub const MINIMAX_H3_ABLATE: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "H3 ablation arms" };
    pub const MLX_SERVE_ANE_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "ANE engine per-call timings in the log" };
    pub const MLX_SERVE_DECODE_FWD_UBENCH: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "engines/mlx", .since = "26.7.12", .doc = "time N decode-width forwards after load (tests/fwd_ubench.sh)" };
    pub const MLX_SERVE_DECODE_FWD_UBENCH_KV: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "engines/mlx", .since = "26.8.11", .doc = "KV tokens prefilled before the forward ubench" };
    pub const MLX_SERVE_DECODE_FWD_UBENCH_S: Var = .{ .kind = .diag, .type = .int, .default = "1", .owner = "engines/mlx", .since = "26.8.11", .doc = "rows per forward in the ubench" };
    pub const MLX_SERVE_DECODE_FWD_UBENCH_TREE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx", .since = "26.10.1", .doc = "16-row ubench forwards as a spec tree" };
    pub const MLX_SERVE_DECODE_GRAPH_DUMP: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "engines/mlx", .since = "26.9.6", .doc = "path: print one decode forward's lazy graph" };
    pub const MLX_SERVE_DECODE_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/diag", .since = "26.7.10", .doc = "per-layer decode profile (syncs every layer)" };
    pub const MLX_SERVE_DFLASH_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/spec/mtp", .since = "26.8.5", .doc = "DFlash per-round trace (exactly 1)" };
    pub const MLX_SERVE_DISPATCH_PROBE: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "engines/mlx/diag", .since = "26.7.12", .doc = "count the dispatches of N decode steps" };
    pub const MLX_SERVE_DSPARK_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.7.12", .doc = "DeepSeek-V4 DSpark per-round profile" };
    pub const MLX_SERVE_DSPARK_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.7.12", .doc = "DeepSeek-V4 DSpark draft trace" };
    pub const MLX_SERVE_DSV4_LAYER_CAP: Var = .{ .kind = .diag, .type = .int, .default = "all layers", .owner = "plugins/deepseek_v4", .since = "26.8.1", .doc = "DeepSeek-V4 decode runs only the first N layers (timing only, garbage output)" };
    pub const MLX_SERVE_DSV4_SKIP_MOE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.8.1", .doc = "set: DeepSeek-V4 decode skips the MoE (timing only, garbage output)" };
    pub const MLX_SERVE_DSV4_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/deepseek_v4", .since = "26.8.1", .doc = "DeepSeek-V4 decode trace (exactly 1)" };
    pub const MLX_SERVE_DUMP_RESTORE: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "engines/mlx/cache", .since = "26.9.3", .doc = "dir: dump restored prefix-cache state" };
    pub const MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "mlx", .since = "26.7.10", .doc = "=1: report no NAX (the pre-NAX GPU-family fallback)" };
    pub const MLX_SERVE_GLM_PROFILE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "GLM-5.3 per-stage forward profile" };
    pub const MLX_SERVE_GQMM_UBENCH: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/blocks/moe", .since = "26.10.2", .doc = "gather-QMM NAX ubench (`sweep`: every tile config)" };
    pub const MLX_SERVE_H3_SPLIT_SHARE: Var = .{ .kind = .diag, .type = .float, .default = "the calibrated cut", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "H3 cluster: share of rows kept on this Mac (0..1)" };
    pub const MLX_SERVE_LAYER_CAP: Var = .{ .kind = .diag, .type = .int, .default = "0 (all layers)", .owner = "engines/mlx/diag", .since = "26.7.12", .doc = "run only the first N layers" };
    pub const MLX_SERVE_MLX_FAULT_CHUNK: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "mlx", .since = "26.9.2", .doc = "inject a Metal OOM at the n-th prefill checkpoint" };
    pub const MLX_SERVE_MLX_FAULT_KIND: Var = .{ .kind = .diag, .type = .text, .default = "the OOM", .owner = "mlx", .since = "26.10.3", .doc = "injected fault kind (`cuda_lost`)" };
    pub const MLX_SERVE_MLX_FAULT_STEP: Var = .{ .kind = .diag, .type = .int, .default = "0", .owner = "mlx", .since = "26.9.2", .doc = "inject a Metal OOM at the n-th decode checkpoint" };
    pub const MLX_SERVE_MTP_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/spec", .since = "26.7.7", .doc = "MTP per-round trace (exactly 1)" };
    pub const MLX_SERVE_MTP_TRACE_SYNC: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/spec/mtp", .since = "26.8.11", .doc = "drain the GPU before each traced MTP verify" };
    pub const MLX_SERVE_PREFILL_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/generate", .since = "26.5.6", .doc = "prefill phase trace at info (exactly 1)" };
    pub const MLX_SERVE_STEP_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "engines/mlx/generate", .since = "26.10.2", .doc = "per-step decode trace" };
    pub const MUSIC3_COST_PROBE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/music3", .since = "26.8.6", .doc = "Music3 depth decoder cost laps" };
    pub const QWEN4_DEBUG_SCORES: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.8.11", .doc = "set: qwen4 QSA indexer scores on stderr" };
    pub const QWEN4_DUMP_QSA_BLOCKS: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.9.3", .doc = "dump the QSA selected blocks" };
    pub const QWEN4_GDN_CHECK: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/blocks/gdn", .since = "26.8.11", .doc = "check the fused GDN arm against the op chain per layer" };
    pub const QWEN4_NO_POOLED: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.8.11", .doc = "set: fail when the QSA pooled-key cache is missing" };
    pub const QWEN4_OPCOUNT: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 forward op counts" };
    pub const QWEN4_PROFILE_ATTN: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.10.1", .doc = "qwen4 attention profile" };
    pub const QWEN4_PROFILE_FWD: Var = .{ .kind = .diag, .type = .text, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 forward profile (1: verify widths, all: every width)" };
    pub const QWEN4_PROFILE_QSA: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp/qsa", .since = "26.9.1", .doc = "qwen4 QSA profile" };
    pub const QWEN4_STANDIN: Var = .{ .kind = .diag, .type = .text, .default = "unset", .owner = "sdk/diag", .since = "26.8.11", .doc = "qwen4 stand-in arms, comma list (gdn, ...)" };
    pub const QWEN4_STREAM_F32: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "set: qwen4 residual stream in f32 (the math oracle)" };
    pub const SUSHI_DECODE_TICK_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: decode tick ubench" };
    pub const SUSHI_EXL3_CHAIN_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.3", .doc = "sushi EXL3 meter: exl3 chain ubench" };
    pub const SUSHI_EXL3_CODEBOOK_AB: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 codebook ab" };
    pub const SUSHI_EXL3_GEMM_ARMS: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.2", .doc = "sushi EXL3 meter: exl3 gemm arms" };
    pub const SUSHI_EXL3_GEMM_COUNTS: Var = .{ .kind = .diag, .type = .text, .default = "a synthetic layer", .owner = "sdk/quant/exl3/sushi", .since = "26.10.2", .doc = "sushi EXL3 GEMM meter: per-expert row counts file" };
    pub const SUSHI_EXL3_LAYER_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 layer ubench" };
    pub const SUSHI_EXL3_LOWK_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.3", .doc = "sushi EXL3 meter: exl3 lowk ubench" };
    pub const SUSHI_EXL3_RG_PREFILL_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "unreleased", .doc = "sushi EXL3 meter: exl3 rg prefill ubench" };
    pub const SUSHI_EXL3_RG_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "unreleased", .doc = "sushi EXL3 meter: exl3 rg ubench" };
    pub const SUSHI_EXL3_SWIGLU_MAXABS: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 swiglu maxabs" };
    pub const SUSHI_EXL3_UNION_HIST: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.1", .doc = "sushi EXL3 meter: exl3 union hist" };
    pub const SUSHI_EXL3_WIDE_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "unreleased", .doc = "sushi EXL3 meter: exl3 wide ubench" };
    pub const SUSHI_GLM_T1_UBENCH: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "sdk/quant/exl3/sushi", .since = "26.10.3", .doc = "sushi EXL3 meter: glm t1 ubench" };
    pub const TTS_TRACE: Var = .{ .kind = .diag, .type = .flag, .default = "off", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "Qwen3-TTS first frames' codes on stderr" };

    // ── test inputs ──
    pub const ACESTEP_COND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_COVER_COND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover DiT velocity matches reference" };
    pub const ACESTEP_COVER_E2E_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_E2E_NOISE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_E2E_TEXT_HIDDEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_NOISE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover DiT velocity matches reference" };
    pub const ACESTEP_COVER_SRC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover e2e denoise matches reference" };
    pub const ACESTEP_COVER_V1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: cover DiT velocity matches reference" };
    pub const ACESTEP_DETOK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ detokenizer matches reference (cos + scale)" };
    pub const ACESTEP_DIT_NOISE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: DiT velocity matches reference" };
    pub const ACESTEP_DIT_V1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: DiT velocity matches reference" };
    pub const ACESTEP_E2E_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: e2e denoise matches reference" };
    pub const ACESTEP_LYRIC_EMBEDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_LYRIC_LEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_REFCOND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: condition encoder with reference audio matches reference" };
    pub const ACESTEP_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/acestep", .since = "26.7.2", .doc = "ACE-Step pack for the live engine tests" };
    pub const ACESTEP_TEXT_HIDDEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: text encoder hidden states match reference" };
    pub const ACESTEP_TEXT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: prompt tokenization matches reference ids" };
    pub const ACESTEP_TEXT_LEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder matches reference" };
    pub const ACESTEP_TOK_HINTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ tokenizer hints + indices match reference" };
    pub const ACESTEP_TOK_INDICES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ tokenizer hints + indices match reference" };
    pub const ACESTEP_TOK_Z: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.8.10", .doc = "acestep oracle: FSQ tokenizer hints + indices match reference" };
    pub const ACESTEP_VAEDEC_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: VAE decode matches reference" };
    pub const ACESTEP_VAEDEC_WAV: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: VAE decode matches reference" };
    pub const ACESTEP_VAEENC_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: condition encoder with reference audio matches reference" };
    pub const ACESTEP_VAEENC_MEAN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/acestep", .since = "26.7.2", .doc = "acestep oracle: VAE encode mean matches reference" };
    pub const ANE_CORES_27B: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_CORES_CTXS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_CORES_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_CORES_NO_GDN: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" };
    pub const ANE_DRAFT_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the drafter program for the ANE runner (ANE_DRAFT_DIR, ANE_DRAFT_PACK = a drafter dir)" };
    pub const ANE_DRAFT_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: the drafter program for the ANE runner (ANE_DRAFT_DIR, ANE_DRAFT_PACK = a drafter dir)" };
    pub const ANE_DUMP_CTXS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_FIRST: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_LANE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_LANES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_LAYERS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_DUMP_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS…" };
    pub const ANE_HEAD_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: head variants for the ANE runner (ANE_HEAD_DIR, ANE_HEAD_PARTS, ANE_HEAD_NOT)" };
    pub const ANE_HEAD_NOT: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: head variants for the ANE runner (ANE_HEAD_DIR, ANE_HEAD_PARTS, ANE_HEAD_NOT)" };
    pub const ANE_HEAD_PARTS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "dump: head variants for the ANE runner (ANE_HEAD_DIR, ANE_HEAD_PARTS, ANE_HEAD_NOT)" };
    pub const ANE_TEST_LANES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "engines/ane", .since = "26.10.3", .doc = "test input read by `testModel`" };
    pub const ANE_TEST_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "26.10.3", .doc = "test input read by `testModel`" };
    pub const CLEF_HTTP_URL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef HTTP: discovery, both APIs, images, invalid inputs, isolation and unload (CLEF_HTTP_URL)" };
    pub const CLEF_TEST_EXPECTED: Var = .{ .kind = .fixture, .type = .text, .default = "tests/fixtures/clef/expected.json", .owner = "plugins/clef", .since = "26.10.2", .doc = "Clef HTTP oracle" };
    pub const CLEF_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef: live joint head agrees with the published MLX loader" };
    pub const CLEF_TEST_REFERENCE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef: live joint head agrees with the published MLX loader" };
    pub const CLEF_TEST_VISION_REFERENCE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/clef", .since = "26.10.2", .doc = "clef: live joint head agrees with the published MLX loader" };
    pub const DFLASH2_ASSISTANT_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.10", .doc = "dflash2 fixture parity: conv block forward + greedy selector path vs z-lab model_mlx" };
    pub const DFLASH2_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.10", .doc = "dflash2 fixture parity: conv block forward + greedy selector path vs z-lab model_mlx" };
    pub const DFLASH_ASSISTANT_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.5", .doc = "dflash fixture parity: encoder + two draft rounds vs transformers reference" };
    pub const DFLASH_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/dflash", .since = "26.8.5", .doc = "dflash fixture parity: encoder + two draft rounds vs transformers reference" };
    pub const DSV41_AR_BASELINE_GB: Var = .{ .kind = .fixture, .type = .float, .default = "the box's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the guard's memory baseline, GB" };
    pub const DSV41_AR_CEILING_GB: Var = .{ .kind = .fixture, .type = .float, .default = "the GPU working set", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the guard's ceiling, GB" };
    pub const DSV41_AR_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: receipt path" };
    pub const DSV41_AR_PHASE: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the served run's phase" };
    pub const DSV41_AR_PROMPT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "the reference's prompt", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: a prompt-id file to run instead" };
    pub const DSV41_AR_PROMPT_TOKENS: Var = .{ .kind = .fixture, .type = .int, .default = "the reference's prompt", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the prompt's length" };
    pub const DSV41_AR_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the Python reference's greedy ids (dump_dsv41_ar_ref.py)" };
    pub const DSV41_AR_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "16", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: decode rows per layer" };
    pub const DSV41_AR_SCHEDULE: Var = .{ .kind = .fixture, .type = .text, .default = "8-row chunks", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: `served` runs the server's schedule" };
    pub const DSV41_AR_SPLIT: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the served run's prompt split" };
    pub const DSV41_AR_STOP_BYTES: Var = .{ .kind = .fixture, .type = .int, .default = "2 GB", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the window's stop below the ceiling, bytes" };
    pub const DSV41_AR_TIER: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the served run's tier" };
    pub const DSV41_BANK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "DeepSeek-V4.1 EXL3 streamed bank dir for the bank-mode tests" };
    pub const DSV41_BENCH_ATTN: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 bench: one-token decode attention as a dependent chain (DSV41_BENCH_ATTN)" };
    pub const DSV41_BENCH_HC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 bench: one hyper-connection stage as a dependent chain (DSV41_BENCH_HC = rows)" };
    pub const DSV41_BENCH_MOEK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 bench: one-token FFN as a dependent chain, composed vs kernels (DSV41_BENCH_MOEK)" };
    pub const DSV41_BILL_CONTEXT: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 bill: the per-context bill table test" };
    pub const DSV41_CELL_ARM: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the arm name stamped on the receipt" };
    pub const DSV41_CELL_BASELINE_GB: Var = .{ .kind = .fixture, .type = .float, .default = "required", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the guard's memory baseline, GB" };
    pub const DSV41_CELL_BILL: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the bill print test" };
    pub const DSV41_CELL_CASE: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the case id stamped on the receipt" };
    pub const DSV41_CELL_CEILING_GB: Var = .{ .kind = .fixture, .type = .float, .default = "required", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the guard's ceiling, GB" };
    pub const DSV41_CELL_CYCLES: Var = .{ .kind = .fixture, .type = .int, .default = "32", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: decode cycles" };
    pub const DSV41_CELL_DECODE_ATTN_SOFTMAX: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the decode attention softmax route (1 | 0)" };
    pub const DSV41_CELL_DECODE_CACHE_BYTES: Var = .{ .kind = .fixture, .type = .int, .default = "the envelope's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the decode cache limit, bytes" };
    pub const DSV41_CELL_DECODE_CACHE_LIMIT_MB: Var = .{ .kind = .fixture, .type = .int, .default = "the envelope's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the decode cache limit, MiB (0 refused)" };
    pub const DSV41_CELL_DECODE_FILL_GRANULE: Var = .{ .kind = .fixture, .type = .text, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the fill's decode granule (row | record)" };
    pub const DSV41_CELL_DECODE_INDEX_TOPK: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the decode indexer top-k route (1 | 0)" };
    pub const DSV41_CELL_DECODE_MXFP8_ROWS: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the decode mxfp8 rows route (1 | 0)" };
    pub const DSV41_CELL_DECODE_SMALLM: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the decode small-M route (1 | 0)" };
    pub const DSV41_CELL_DELTA: Var = .{ .kind = .fixture, .type = .float, .default = "0.3", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the receipt's tolerance" };
    pub const DSV41_CELL_DENSE_RC: Var = .{ .kind = .fixture, .type = .flag, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: a retired lever; set, the cell refuses by name" };
    pub const DSV41_CELL_DEVROUTE: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the device routing (1 | 0)" };
    pub const DSV41_CELL_DRAFT_AHEAD: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the draft ahead (1 | 0)" };
    pub const DSV41_CELL_DRAFT_STAGED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the staged draft (1 | 0)" };
    pub const DSV41_CELL_EMBEDDING_ROWS: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: host embedding rows (1 | 0)" };
    pub const DSV41_CELL_ENGRAM_POSTED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the posted Engram read (1 | 0)" };
    pub const DSV41_CELL_EVENT_GATES: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the typical tier's event-gated waves (1 | 0)" };
    pub const DSV41_CELL_FILL_LADDER: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: run the fill ladder" };
    pub const DSV41_CELL_GROW_FILL: Var = .{ .kind = .fixture, .type = .text, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the grow fill mode" };
    pub const DSV41_CELL_HEAD_MODE: Var = .{ .kind = .fixture, .type = .text, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the head's mode (bf16 | mxfp8)" };
    pub const DSV41_CELL_HEAD_MXFP8_RC: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the mxfp8 head's row-chunked route (1 | 0)" };
    pub const DSV41_CELL_HOIST_FIRST: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the hoisted first layer (1 | 0)" };
    pub const DSV41_CELL_INPUT_STREAM_EARLY_RELEASE: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the input stream's early release (1 | 0)" };
    pub const DSV41_CELL_LAYER_MAJOR: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: K16 layer-major prefill (1 | 0)" };
    pub const DSV41_CELL_MAX_CONTEXT: Var = .{ .kind = .fixture, .type = .int, .default = "the bill's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the server's billed context" };
    pub const DSV41_CELL_MAX_TOKENS: Var = .{ .kind = .fixture, .type = .int, .default = "1024", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: generated ids, the primary included" };
    pub const DSV41_CELL_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: model dir of the in-process bench cell" };
    pub const DSV41_CELL_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: receipt path" };
    pub const DSV41_CELL_PHASE_POLL_MS: Var = .{ .kind = .fixture, .type = .int, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the phase change's settle poll, ms" };
    pub const DSV41_CELL_PHASE_SETTLE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: a retired lever; set, the cell refuses by name" };
    pub const DSV41_CELL_PREDICT_BF16: Var = .{ .kind = .fixture, .type = .flag, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: a retired lever; set, the cell refuses by name" };
    pub const DSV41_CELL_PREFILL_ATTN: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill attention route (1 | 0)" };
    pub const DSV41_CELL_PREFILL_COMBINE: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill combine route (1 | 0)" };
    pub const DSV41_CELL_PREFILL_FUSED_DOWN: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill fused down (1 | 0)" };
    pub const DSV41_CELL_PREFILL_HC: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill hc route (1 | 0)" };
    pub const DSV41_CELL_PREFILL_HCPOST: Var = .{ .kind = .fixture, .type = .text, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the hc post route (fused | region)" };
    pub const DSV41_CELL_PREFILL_HC_POST: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill hc post route (1 | 0)" };
    pub const DSV41_CELL_PREFILL_HOST_SHARED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill host-shared expert (1 | 0)" };
    pub const DSV41_CELL_PREFILL_INDEX: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill indexer route (1 | 0)" };
    pub const DSV41_CELL_PREFILL_INPUT_RELEASE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: a retired lever; set, the cell refuses by name" };
    pub const DSV41_CELL_PREFILL_JOINLESS: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the joinless prefill (1 | 0)" };
    pub const DSV41_CELL_PREFILL_OPROJ: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prefill o-projection route (1 | 0)" };
    pub const DSV41_CELL_PREFILL_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "the admission's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: forced prefill rows (with DSV41_CELL_ROWS)" };
    pub const DSV41_CELL_PREFILL_SHAREDMID: Var = .{ .kind = .fixture, .type = .text, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the shared expert's middle (compiled | eager)" };
    pub const DSV41_CELL_PREFILL_SUB: Var = .{ .kind = .fixture, .type = .text, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prompt's sub-chunk rows (`whole`: one call)" };
    pub const DSV41_CELL_PROFILE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the profiling window test (prompt pass only)" };
    pub const DSV41_CELL_PROMPT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the prompt-id file" };
    pub const DSV41_CELL_PROMPT_TOKENS: Var = .{ .kind = .fixture, .type = .int, .default = "16384", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: prompt length (the context sweep's case)" };
    pub const DSV41_CELL_REQUESTS: Var = .{ .kind = .fixture, .type = .int, .default = "1", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: requests per cell" };
    pub const DSV41_CELL_REQUEST_END: Var = .{ .kind = .fixture, .type = .text, .default = "end", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: `prefill` defers the request end to the next prefill" };
    pub const DSV41_CELL_ROUTED_BANKED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the banked routed GEMV (1 | 0)" };
    pub const DSV41_CELL_ROUTED_FORMS: Var = .{ .kind = .fixture, .type = .text, .default = "stock", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the routed GEMV forms" };
    pub const DSV41_CELL_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "the admission's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: forced decode rows" };
    pub const DSV41_CELL_STOP_BYTES: Var = .{ .kind = .fixture, .type = .int, .default = "2 GB", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the window's stop below the ceiling, bytes" };
    pub const DSV41_CELL_TRANSIENT_RELEASE: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the transient release (1 | 0)" };
    pub const DSV41_CELL_WIDE_BASE_AT_SEED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide feed's base at the seed (1 | 0)" };
    pub const DSV41_CELL_WIDE_COLD_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide read schedule's cold rows" };
    pub const DSV41_CELL_WIDE_DEFER_BASE: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide feed's deferred base (1 | 0)" };
    pub const DSV41_CELL_WIDE_DEPTH: Var = .{ .kind = .fixture, .type = .int, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide read schedule's depth" };
    pub const DSV41_CELL_WIDE_FEED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide read schedule's feed (1 | 0)" };
    pub const DSV41_CELL_WIDE_HOT_FIRST: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide feed's hot-first half (1 | 0)" };
    pub const DSV41_CELL_WIDE_READ_AHEAD: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide feed's read-ahead (1 | 0)" };
    pub const DSV41_CELL_WIDE_SEED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide feed's seed half (1 | 0)" };
    pub const DSV41_CELL_WIDE_SEED_ALIGNED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the wide feed's aligned seed (1 | 0)" };
    pub const DSV41_CELL_WINDOW_RING_HEADROOM: Var = .{ .kind = .fixture, .type = .int, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: a window ring lever over the tier's" };
    pub const DSV41_CELL_WINDOW_RING_MAX_VERIFY: Var = .{ .kind = .fixture, .type = .int, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: a window ring lever over the tier's" };
    pub const DSV41_CELL_WINDOW_RING_SLACK: Var = .{ .kind = .fixture, .type = .int, .default = "the tier's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: a window ring lever over the tier's" };
    pub const DSV41_CELL_WIRED_GB: Var = .{ .kind = .fixture, .type = .float, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: wired memory, GB" };
    pub const DSV41_DSPARK_CYCLES_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 DSpark oracle input from the reference runtime" };
    pub const DSV41_DSPARK_MINI_CONFIG: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 DSpark oracle input from the reference runtime" };
    pub const DSV41_DSPARK_RECEIPT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 DSpark oracle input from the reference runtime" };
    pub const DSV41_DSPARK_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 DSpark oracle input from the reference runtime" };
    pub const DSV41_DSPARK_SUBSET: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 DSpark oracle input from the reference runtime" };
    pub const DSV41_ENGRAM_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 Engram oracle input from the reference runtime" };
    pub const DSV41_ENGRAM_REPLAY: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 Engram oracle input from the reference runtime" };
    pub const DSV41_ENGRAM_REPLAY_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 Engram oracle input from the reference runtime" };
    pub const DSV41_ENGRAM_TOKEN_MAP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 ar harness: the converter's Engram token map" };
    pub const DSV41_HEAD_FIXTURE_DUMP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 parity: the reference's dump" };
    pub const DSV41_KERNELS_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate on the GPU" };
    pub const DSV41_KERNEL_DECODE2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel decode2 gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_DRAFT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel draft gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_OPS_FAMILIES: Var = .{ .kind = .fixture, .type = .text, .default = "every family", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate: the families to replay" };
    pub const DSV41_KERNEL_OPS_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel ops gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_PREFILL2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill2 gate: the reference runtime's fixture dir" };
    pub const DSV41_KERNEL_PREFILL_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 kernel prefill gate: the reference runtime's fixture dir" };
    pub const DSV41_KV_BOUNDED: Var = .{ .kind = .fixture, .type = .flag, .default = "the tier's route", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 DSpark harness: KV lanes bounded to the run's positions" };
    pub const DSV41_LOOKUP_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_M0_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_MODULE_BASELINE_GB: Var = .{ .kind = .fixture, .type = .float, .default = "7.755397656", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 module plan test: the box baseline, GB" };
    pub const DSV41_MODULE_CEILING_GB: Var = .{ .kind = .fixture, .type = .float, .default = "the GPU's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 module plan test: the guard's ceiling, GB" };
    pub const DSV41_MODULE_HEAD: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 module plan test: bill the draft head" };
    pub const DSV41_MODULE_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "the admission's", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 module plan test: forced rows" };
    pub const DSV41_MODULE_WIRED_GB: Var = .{ .kind = .fixture, .type = .float, .default = "3.377741824", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 module plan test: wired memory, GB" };
    pub const DSV41_PARITY2_DUMP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 parity: the reference's dump" };
    pub const DSV41_PARITY_DEVICE: Var = .{ .kind = .fixture, .type = .text, .default = "cpu", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 parity: `gpu` runs it on the GPU" };
    pub const DSV41_PARITY_DIGESTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 parity: the reference's dump" };
    pub const DSV41_PARITY_DUMP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 parity: the reference's dump" };
    pub const DSV41_PARITY_REPORT: Var = .{ .kind = .fixture, .type = .text, .default = "<source>.zig-report.json", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 parity: report path" };
    pub const DSV41_PARITY_REPORT_ONLY: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 parity: a difference reports instead of failing" };
    pub const DSV41_PHASE0B_MLX: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "streamed-bank MLX device tests (a GPU-lock-held run)" };
    pub const DSV41_PHASE0_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed/expert", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PHASE1_ROUTE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PHASE2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PHASE4A_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed/expert", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_PPL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_PPL_HEAD8: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_PPL_TEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_PPL_WINDOWS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" };
    pub const DSV41_ROUND: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 DSpark round: draft, verify and finish timed apart, greedy (DSV41_ROUND)" };
    pub const DSV41_ROUND_OFFSET: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 DSpark round: draft, verify and finish timed apart, greedy (DSV41_ROUND)" };
    pub const DSV41_ROUND_TEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 DSpark round: draft, verify and finish timed apart, greedy (DSV41_ROUND)" };
    pub const DSV41_SCHEDULE_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 oracle fixture from the reference runtime" };
    pub const DSV41_SELFCHECK_DEVICE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "dsv41 pre-ship gate: the kernels' self-check plan on the device" };
    pub const DSV41_STEP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 decode step: serial one-token steps as the server pipelines them (DSV41_STEP)" };
    pub const DSV41_STEP_CTX: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 decode step: serial one-token steps as the server pipelines them (DSV41_STEP)" };
    pub const DSV41_TINY: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.2", .doc = "dsv41 fixture: prefill, chunked prefill, spans and decode match DeepSeek's reference (DSV41_TINY)" };
    pub const DSV41_VERIFY_BASELINE_GB: Var = .{ .kind = .fixture, .type = .float, .default = "required", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 pre-ship gate: the box baseline, GB" };
    pub const DSV41_VERIFY_CEILING_GB: Var = .{ .kind = .fixture, .type = .float, .default = "required", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 pre-ship gate: the guard's ceiling, GB" };
    pub const DSV41_VERIFY_DEVICE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 pre-ship gate: the served Module's construction on the device" };
    pub const DSV41_VERIFY_ROWS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 verify width: time of a verify block per draft count (DSV41_VERIFY_ROWS)" };
    pub const DSV41_VERIFY_STOP_BYTES: Var = .{ .kind = .fixture, .type = .int, .default = "2 GiB", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 pre-ship gate: the window's stop, bytes" };
    pub const DSV41_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "test input read by `fixtureEnv`" };
    pub const DSV41_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v41", .since = "26.10.3", .doc = "dsv41 vision: tower, aligner and span match the reference" };
    pub const DSV41_WAVE_TABLE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 bill: the prompt wave table test" };
    pub const DSV4_MINI: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec", .since = "26.7.12", .doc = "dsv4: nextPld on a chokepoint-disabled generator stays serial (DSV4_MINI)" };
    pub const DSV4_REAL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/deepseek_v4", .since = "26.7.12", .doc = "dsv4: forwardPrefill on the REAL mirror continues 'The capital of France is' (DSV4_REAL)" };
    pub const DUPES_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: the receipt path a pack run writes" };
    pub const DUPES_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: the pack a run loads through the engine (resident or streamed)" };
    pub const DUPES_PROMPT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: a file of prompt ids (whitespace or commas)" };
    pub const DUPES_RECEIPTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: two receipt paths, comma-separated, to compare" };
    pub const DUPES_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "32", .owner = "sdk/weights/streamed", .since = "unreleased", .doc = "dupes oracle: greedy steps a pack run takes after the prompt" };
    pub const FLINT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flint", .since = "26.10.3", .doc = "flint fixture: one-shot and chunked prefill + decode vs rec_mlx's model (FLINT_MODEL, FLINT_FIXTURE)" };
    pub const FLINT_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flint", .since = "26.10.3", .doc = "flint fixture: one-shot and chunked prefill + decode vs rec_mlx's model (FLINT_MODEL, FLINT_FIXTURE)" };
    pub const FLUX_DECODED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux VAE reproduces decoded image" };
    pub const FLUX_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux end-to-end pipeline matches reference image" };
    pub const FLUX_IMGIDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_MASK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux end-to-end pipeline matches reference image" };
    pub const FLUX_NOISE0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_PACKED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux VAE reproduces decoded image" };
    pub const FLUX_PE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const FLUX_TXTIDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/flux2", .since = "26.6.13", .doc = "flux DiT reproduces noise_step0" };
    pub const GLB_FIXTURE_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/mesh", .since = "26.7.2", .doc = "glb: write swift fixture" };
    pub const GLB_TEXTURED_FIXTURE_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/mesh", .since = "26.7.2", .doc = "glb: write textured swift fixture" };
    pub const GLM53_EXL3_BENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm_moe_dsa/stream/glm53_exl3", .since = "26.10.3", .doc = "glm53 EXL3 bench: the pack dir" };
    pub const GLM53_EXL3_TINY_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm_moe_dsa/stream/glm53_exl3", .since = "26.10.3", .doc = "glm53 EXL3: write the tiny fixture's pack to this dir" };
    pub const GLM53_MTP_PARITY: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm_moe_dsa/stream", .since = "26.10.3", .doc = "glm53 MTP parity: the reference dump dir" };
    pub const GLM53_PARITY: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm_moe_dsa/stream", .since = "26.10.3", .doc = "glm53 parity: the reference dump dir" };
    pub const GLM5NEXT_DETERMINISM_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next determinism device test" };
    pub const GLM5NEXT_DSA_DECODE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next dsa decode device test" };
    pub const GLM5NEXT_EXL3_KBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next EXL3 kernel bench" };
    pub const GLM5NEXT_FP8_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next fp8 device test" };
    pub const GLM5NEXT_FUSION_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next fusion device test" };
    pub const GLM5NEXT_GLUE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next glue device test" };
    pub const GLM5NEXT_KDA_DECODE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next kda decode device test" };
    pub const GLM5NEXT_KDA_FUSED_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next kda fused device test" };
    pub const GLM5NEXT_KDA_PERCORE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next kda percore device test" };
    pub const GLM5NEXT_KDA_PROLOGUE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next kda prologue device test" };
    pub const GLM5NEXT_NVFP4_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next nvfp4 device test" };
    pub const GLM5NEXT_NVFP4_VECTORS: Var = .{ .kind = .fixture, .type = .text, .default = "required with GLM5NEXT_NVFP4_GPU", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next NVFP4 parity: the converter's vector dir" };
    pub const GLM5_BANK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next bill tests: the streamed bank dir" };
    pub const GLM5_BILL_BASELINES_GB: Var = .{ .kind = .fixture, .type = .text, .default = "10.6,12.0,12.6,13.0", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next bill: baselines, GB, comma list" };
    pub const GLM5_BILL_CEILING: Var = .{ .kind = .fixture, .type = .int, .default = "the box's", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next bill: the ceiling, bytes" };
    pub const GLM5_BILL_DRAFT_HEAD: Var = .{ .kind = .fixture, .type = .text, .default = "the settings'", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next bill: the draft head (bf16 | q4 | q8)" };
    pub const GLM5_BILL_MTP: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next bill: with the MTP lane" };
    pub const GLM5_BILL_ROWS: Var = .{ .kind = .fixture, .type = .text, .default = "unset", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next bill: a <prompt>/<decode> rows pin" };
    pub const GLM5_BILL_WAVE_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "the default wave rows", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next bill: the EXL3 prompt wave rows" };
    pub const GLM5_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next fixture: one-shot and chunked prefill + decode vs mlx-vlm glm5_next (GLM5_MODEL…" };
    pub const GLM5_HEAD_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "random rows", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next MTP draft head test: the pack whose lm_head rows it reads" };
    pub const GLM5_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next fixture: one-shot and chunked prefill + decode vs mlx-vlm glm5_next (GLM5_MODEL…" };
    pub const GLM5_MTP_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next MTP fixture: the pack's mtp.0 layer drafts as mlx-vlm's drafter, one-shot and row by row…" };
    pub const GLM5_NEXT_BANK_VECTORS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next/stream", .since = "26.10.3", .doc = "glm5_next NVFP4 bank: the converter's test vectors" };
    pub const GLM5_REAL_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm5_next verify windows of 2..16 rows track the same rows decoded one at a time (GLM5_REAL_MODEL)" };
    pub const GLM5_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "test input read by `fixtureEnv`" };
    pub const GLM5_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/glm5_next", .since = "26.10.2", .doc = "glm vision: tower and merger match the reference" };
    pub const H3_VSA_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "h3 vsa: the gather arm equals FastVideo's reference on its own fixture" };
    pub const H3_VSA_UBENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.10.3", .doc = "h3 vsa kernel: ubench against dense attention at a 720p clip's packed shape" };
    pub const HY3DP_DINO_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: DINO features + ImageProjModel context match reference" };
    pub const HY3DP_DINO_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: DINO features + ImageProjModel context match reference" };
    pub const HY3DP_DINO_PROJ: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: DINO features + ImageProjModel context match reference" };
    pub const HY3DP_FWD_SMOKE: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet: full forward runs end-to-end (shape + finite)" };
    pub const HY3DP_REF_DOWN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_REF_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_REF_MID: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_REF_UP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: dual-UNet ref cache matches reference" };
    pub const HY3DP_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE encode matches reference" };
    pub const HY3DP_UNET_LATENTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_NORMAL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_POSITION: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_POSMAPS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_UNET_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint-unet oracle: full 2.5D step matches reference" };
    pub const HY3DP_VAE_DEC_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE decode matches reference" };
    pub const HY3DP_VAE_DEC_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE decode matches reference" };
    pub const HY3DP_VAE_ENC_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE encode matches reference" };
    pub const HY3DP_VAE_ENC_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d-paint oracle: VAE encode matches reference" };
    pub const HY3D_DENOISE_CTX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_GUIDANCE: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_INIT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DENOISE_STEPS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: denoise loop matches reference" };
    pub const HY3D_DINO_IN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DINO features match reference" };
    pub const HY3D_DINO_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DINO features match reference" };
    pub const HY3D_E2E_GRID: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_E2E_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_E2E_NVERT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_E2E_RES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: e2e SDF grid matches reference" };
    pub const HY3D_SDF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: latent to SDF matches reference" };
    pub const HY3D_SDF_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: latent to SDF matches reference" };
    pub const HY3D_SDF_PTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: latent to SDF matches reference" };
    pub const HY3D_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DINO features match reference" };
    pub const HY3D_VEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3D_VEL_CTX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3D_VEL_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3D_VEL_SIGMA: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/hunyuan3d", .since = "26.7.2", .doc = "hy3d oracle: DiT velocity matches reference" };
    pub const HY3_MODEL_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/chat", .since = "26.7.8", .doc = "renderChatTemplate: REAL Hy3 chat_template.jinja renders without fallback (HY3_MODEL_DIR)" };
    pub const INKLING_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/inkling", .since = "26.7.12", .doc = "test input read by `loadInklingFixtures`" };
    pub const KEV_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/kev", .since = "26.10.1", .doc = "kev: oracle against kev's MLX backend (KEV_TEST_MODEL + KEV_FIXTURES)" };
    pub const KEV_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/kev", .since = "26.10.1", .doc = "kev: oracle against kev's MLX backend (KEV_TEST_MODEL + KEV_FIXTURES)" };
    pub const KOKORO_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/kokoro", .since = "26.7.12", .doc = "Kokoro reference fixtures dir" };
    pub const KOKORO_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/kokoro", .since = "26.7.12", .doc = "Kokoro pack for the live tests" };
    pub const KREA_DECODED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE reproduces decoded image" };
    pub const KREA_E2E: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_H: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_SEED: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_E2E_W: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea end-to-end pipeline matches reference image" };
    pub const KREA_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_MASK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_PE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE encoder round-trips through the decoder" };
    pub const KREA_TE_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea text encoder reproduces hidden states" };
    pub const KREA_VAE_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE reproduces decoded image" };
    pub const KREA_VAE_LATH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea VAE reproduces decoded image" };
    pub const KREA_VEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_CTX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_IMG: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_POS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const KREA_VEL_VALID: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/krea", .since = "26.6.13", .doc = "krea DiT reproduces velocity" };
    pub const LAGUNA_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/blocks/rope", .since = "26.7.10", .doc = "laguna yarn parity vs modeling_laguna.py (LAGUNA_FIXTURES)" };
    pub const LAYA_BENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/laya", .since = "26.9.6", .doc = "laya: forward latency breakdown (LAYA_BENCH=1)" };
    pub const LAYA_BENCH_GAP_MS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/laya", .since = "26.9.6", .doc = "laya: forward latency breakdown (LAYA_BENCH=1)" };
    pub const LAYA_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.9.6", .doc = "byte_fallback: real Laya tokenizer matches HF `tokenizers` on out-of-vocab scripts (LAYA_TEST_MODEL)" };
    pub const LAYA_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.9.6", .doc = "byte_fallback: real Laya tokenizer matches HF `tokenizers` on out-of-vocab scripts (LAYA_TEST_MODEL)" };
    pub const LFM2_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/lfm2", .since = "26.8.6", .doc = "lfm2 vision live: tower parity vs the executed reference" };
    pub const LFM2_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/lfm2", .since = "26.8.6", .doc = "lfm2 vision live: tower parity vs the executed reference" };
    pub const LLAMA_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/llama", .since = "26.5.7", .doc = "test input read by `testModelPath`" };
    pub const LOGPROBS_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/grammar", .since = "26.9.6", .doc = "constrained generation returns one logprob entry per token, paired with that token" };
    pub const LTX_ADA_EMB: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT adaLN conditioning reproduces AdaLayerNormSingle" };
    pub const LTX_ADA_PARAMS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT adaLN conditioning reproduces AdaLayerNormSingle" };
    pub const LTX_AUDIO_ENC_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE encoder matches the reference latent (cos)" };
    pub const LTX_AUDIO_ENC_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE encoder matches the reference latent (cos)" };
    pub const LTX_AUDIO_ENC_WAVE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE encoder matches the reference latent (cos)" };
    pub const LTX_AUDIO_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE decode matches the reference mel (cos)" };
    pub const LTX_AUDIO_MEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE decode matches the reference mel (cos)" };
    pub const LTX_AUDIO_MEL_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "audio VAE decode matches the reference mel (cos)" };
    pub const LTX_AUDIO_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/ltx", .since = "26.7.1", .doc = "LTX audio VAE + vocoder dir for the live tests" };
    pub const LTX_AUDIO_WAVE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "vocoder matches the reference waveform (cos)" };
    pub const LTX_AUDIO_WAVE_MIN: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "vocoder matches the reference waveform (cos)" };
    pub const LTX_B0_A2VG: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AAP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_APP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_ATEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AVA: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_AVV: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_OUTA: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_OUTV: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_V2AG: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VAP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VPP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_B0_VTEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" };
    pub const LTX_CONNT_AIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONNT_AOUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONNT_VIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONNT_VOUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorTransform reproduces Embeddings1DConnector" };
    pub const LTX_CONN_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorProject reproduces TextEmbeddingProjection" };
    pub const LTX_CONN_STACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorProject reproduces TextEmbeddingProjection" };
    pub const LTX_CONN_VIDEO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx connectorProject reproduces TextEmbeddingProjection" };
    pub const LTX_CONVIN_X: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx decoderConv3d matches reference conv_in" };
    pub const LTX_CONVIN_Y: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx decoderConv3d matches reference conv_in" };
    pub const LTX_DIFFVAE_CONTEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae decode runs the whole ladder on the shipped checkpoint" };
    pub const LTX_DIFFVAE_PEAK: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae decode peak stays inside the budget the residency bill promises" };
    pub const LTX_DIFFVAE_PIXELS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_STAGE0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_STAGE1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_STAGE2: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_VPRED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_DIFFVAE_XT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.8.6", .doc = "diffvae parity: every stage reproduces the reference decoder" };
    pub const LTX_ENC_H: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_ENC_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_ENC_PIXELS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_ENC_W: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx VAE encoder reproduces reference VideoEncoder" };
    pub const LTX_FRAMES_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT latent-to-frames reproduces reference decode" };
    pub const LTX_FRAMES_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT latent-to-frames reproduces reference decode" };
    pub const LTX_FWD_ALAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_ATEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_AVEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_VLAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_VTEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_FWD_VVEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditForward reproduces LTXModel velocity" };
    pub const LTX_GEMMA_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S24: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_GEMMA_S48: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx gemmaCapture reproduces reference hidden states" };
    pub const LTX_NSA_PARAMS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditNormedSA reproduces compute_video_normed_sa" };
    pub const LTX_NSA_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditNormedSA reproduces compute_video_normed_sa" };
    pub const LTX_NSA_VH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditNormedSA reproduces compute_video_normed_sa" };
    pub const LTX_POS_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_POS_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_ACOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_ASIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_AXCOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_AXSIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VCOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VSIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VXCOS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_ROPE_VXSIN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditRope matches reference precompute_rope_freqs" };
    pub const LTX_SAMP_COND_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_COND_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_FINAL_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_FINAL_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NEG_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NEG_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NOISE_A: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_NOISE_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT ditSampleCfg reproduces guided denoise loop" };
    pub const LTX_SAMP_SIGMAS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT dynamicShiftSchedule matches reference" };
    pub const LTX_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx loader: connector + vae_decoder key map" };
    pub const LTX_TEXT_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT encodeTextLtx reproduces PromptEncoder" };
    pub const LTX_TEXT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT encodeTextLtx reproduces PromptEncoder" };
    pub const LTX_TEXT_VIDEO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx DiT encodeTextLtx reproduces PromptEncoder" };
    pub const LTX_UP_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx latent upsampler reproduces reference LatentUpsampler" };
    pub const LTX_UP_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.7.1", .doc = "ltx latent upsampler reproduces reference LatentUpsampler" };
    pub const LTX_VAE_DECODED: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx vaeDecode reproduces reference VideoDecoder" };
    pub const LTX_VAE_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.9.1", .doc = "ltx preview: the Latent2RGB preview resembles the decoded frame" };
    pub const LTX_VAE_LATENT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/ltx", .since = "26.6.13", .doc = "ltx vaeDecode reproduces reference VideoDecoder" };
    pub const MAGEFLOW_DIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow DiT forward parity (env-gated)" };
    pub const MAGEFLOW_DIT_MASKED_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow DiT masked parity (env-gated)" };
    pub const MAGEFLOW_E2E_BF16_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow end-to-end DiT-loop parity bf16 (env-gated)" };
    pub const MAGEFLOW_EDIT_E2E_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow edit loop parity bf16 (env-gated)" };
    pub const MAGEFLOW_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow VAE decode parity (env-gated)" };
    pub const MAGEFLOW_TE_EDIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow TE edit parity (env-gated)" };
    pub const MAGEFLOW_TE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow TE conditioning parity (env-gated)" };
    pub const MAGEFLOW_VAE_ENC_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow VAE encode parity (env-gated)" };
    pub const MAGEFLOW_VAE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow VAE decode parity (env-gated)" };
    pub const MAGEFLOW_VIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mage_flow", .since = "26.7.11", .doc = "MageFlow ViT parity (env-gated)" };
    pub const MIMO_BANK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo bill tests: the streamed bank dir" };
    pub const MIMO_BANKVIEW_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: bankview gpu" };
    pub const MIMO_BF16_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo pack test: the pack dir" };
    pub const MIMO_BILL_BASELINES_GB: Var = .{ .kind = .fixture, .type = .text, .default = "9.2,10.0,10.6,12.0", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo bill: baselines, GB, comma list" };
    pub const MIMO_BILL_CEILING: Var = .{ .kind = .fixture, .type = .int, .default = "the box's", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo bill: the ceiling, bytes" };
    pub const MIMO_BILL_PROMPT_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "unset", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo bill: forced prompt rows" };
    pub const MIMO_BILL_ROWS: Var = .{ .kind = .fixture, .type = .int, .default = "unset", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo bill: forced decode rows" };
    pub const MIMO_DEC_ATTN_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: dec attn gpu" };
    pub const MIMO_FP8RES_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: fp8res gpu" };
    pub const MIMO_FP8RES_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo pack test: the pack dir" };
    pub const MIMO_GLUE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: glue gpu" };
    pub const MIMO_HITPASS_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: hitpass gpu" };
    pub const MIMO_MOE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: moe gpu" };
    pub const MIMO_MTP_MLX: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: mtp mlx" };
    pub const MIMO_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo pack test: the pack dir" };
    pub const MIMO_PREFILL_COMBINE_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: prefill combine gpu" };
    pub const MIMO_Q4_BANK_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo q4 bank: fixture path" };
    pub const MIMO_Q4_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 gpu" };
    pub const MIMO_Q4_KV8: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 kv8" };
    pub const MIMO_Q4_MATRIX: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 matrix" };
    pub const MIMO_Q4_PADDED_ATTENTION: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 padded attention" };
    pub const MIMO_Q4_REAL_SHAPE: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 real shape" };
    pub const MIMO_Q4_SDK_STREAM_GPU: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 sdk stream gpu" };
    pub const MIMO_Q4_SORTED: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 sorted" };
    pub const MIMO_Q4_TILED: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: q4 tiled" };
    pub const MIMO_READ_AHEAD_MLX: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo device test: read ahead mlx" };
    pub const MIMO_ROUTE_TRACE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "a temp dir", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo route trace policy fixture: output prefix" };
    pub const MIMO_ROUTE_TRACE_FIXTURE_SPLIT: Var = .{ .kind = .fixture, .type = .text, .default = "a temp dir", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo route trace policy fixture: output prefix" };
    pub const MIMO_V2_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2", .since = "26.10.2", .doc = "mimo_v2 batched decode: a slot in a two-slot tick tracks the fp32 reference as well as serial…" };
    pub const MIMO_V2_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2", .since = "26.10.2", .doc = "mimo_v2 sliding ring: a chunked prefill past the kept rows, then decode, matches the full-buffer…" };
    pub const MIMO_V2_MTP_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2", .since = "26.10.2", .doc = "mimo mtp heads track the torch rendering of the MiMo-V2 MTP layer across rounds, drafts and…" };
    pub const MIMO_X19_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/mimo_v2/stream", .since = "26.10.3", .doc = "mimo pack test: the pack dir" };
    pub const MINICPM5_MODEL_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/chat", .since = "26.9.1", .doc = "renderChatTemplate: REAL MiniCPM5 chat_template.jinja multi-turn tool history renders without…" };
    pub const MINICPM_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_5", .since = "26.10.3", .doc = "test input read by `fixtureEnv`" };
    pub const MINICPM_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_5", .since = "26.10.3", .doc = "minicpm vision: tower stages and merger match the reference" };
    pub const MINIMAX_H3_AUDIO_ENC_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.3", .doc = "minimax h3 audio live: encoder parity vs the executed reference" };
    pub const MINIMAX_H3_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_FRAMES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_GEMM_UBENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.3", .doc = "minimax h3 live: projection gemm ubench" };
    pub const MINIMAX_H3_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 audio live: encoder parity vs the executed reference" };
    pub const MINIMAX_H3_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_PROMPT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_SDPA_UBENCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: sdpa shape ubench" };
    pub const MINIMAX_H3_SIZE: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_SPLIT_OVERFLOW: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.10.2", .doc = "minimax h3 live: the SwiGLU fallback reads every row of a [S, 28672] fc1 output past 2^31" };
    pub const MINIMAX_H3_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MINIMAX_H3_VAE_ENC_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 vae live: encoder parity vs the executed reference" };
    pub const MINIMAX_H3_VIT_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/minimax_h3", .since = "26.8.3", .doc = "minimax h3 live: vision tower parity vs the executed reference" };
    pub const MINIMAX_H3_WIDTH: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/minimax_h3", .since = "26.8.2", .doc = "minimax h3 live: generates a clip" };
    pub const MLX_CUDA_QMV_MAX_ROWS: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "mlx", .since = "26.10.3", .doc = "fp quantized matmul on GB10: 8 rows equal each row alone, bit for bit, and no worse than the tiled…" };
    pub const MLX_SERVE_GDN_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/gdn", .since = "26.7.7", .doc = "GDN µbench: sequential kernel vs bare qmm at 27B shapes (attribution; MLX_SERVE_GDN_UBENCH=1)" };
    pub const MLX_SERVE_GGUF_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/quant/gguf", .since = "26.10.1", .doc = "real GGUF: stands in for a model dir, and --engine turns it off (set MLX_SERVE_GGUF_TEST_MODEL)" };
    pub const MLX_SERVE_GOLDEN_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/testing", .since = "unreleased", .doc = "golden-output dir for the golden tests" };
    pub const MLX_SERVE_KERNEL_CONTRACT_OFFLINE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/testing/kernel_contract", .since = "unreleased", .doc = "kernel contract offline: each record's M1/M2 limit is what the toolchain compiles" };
    pub const MLX_SERVE_KIT_PLUGINS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/plugins", .since = "unreleased", .doc = "dir of kit-built plugin dylibs the loader test opens" };
    pub const MLX_SERVE_KVQ_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/mlx/cache", .since = "26.7.12", .doc = "kv-quant decode attention ubench" };
    pub const MLX_SERVE_LAGUNA_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "plugins/laguna", .since = "26.7.12", .doc = "Laguna-XS decode µbench: bf16 attention projections (MLX_SERVE_LAGUNA_UBENCH=1)" };
    pub const MLX_SERVE_MOE_GATHER_TRACE: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.7.10", .doc = "trace in the MoE gather ubench" };
    pub const MLX_SERVE_MOE_GATHER_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.7.10", .doc = "MoE decode gather µbench (MLX_SERVE_MOE_GATHER_UBENCH=1)" };
    pub const MLX_SERVE_MOE_WSUM_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.10.1", .doc = "MoE prefill expert reduction ubench" };
    pub const MLX_SERVE_PREFILL_QMM_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.7.10", .doc = "prefill qmm µbench: stock qmm vs dequant+GEMM at 27B prefill shapes (MLX_SERVE_PREFILL_QMM_UBENCH=1)" };
    pub const MLX_SERVE_PROOF_RECORD: Var = .{ .kind = .fixture, .type = .text, .default = "unset: proofs replay their tapes", .owner = "engines/mlx/plugin", .since = "unreleased", .doc = "record proof tapes into this dir instead of replaying them" };
    pub const MLX_SERVE_TEST_ANE_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ane", .since = "unreleased", .doc = "test input read by `realPack`" };
    pub const MLX_SERVE_TEST_DS4_GGUF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/ds4", .since = "unreleased", .doc = "test input read by `realGguf`" };
    pub const MLX_SERVE_TEST_LLAMA_GGUF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/llama", .since = "unreleased", .doc = "test input read by `realGguf`" };
    pub const MLX_SERVE_TEST_SHARD: Var = .{ .kind = .fixture, .type = .text, .default = "unset: every test", .owner = "build/test_runner.zig", .since = "unreleased", .doc = "k/n: this test process runs every n-th test from k; a `sharded:` test splits its own work the same way" };
    pub const MLX_SERVE_VERIFY_WIDTH_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.8.5", .doc = "spec-verify width sweep µbench (MLX_SERVE_VERIFY_WIDTH_UBENCH=1)" };
    pub const MLX_SERVE_VQMM_FN_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.10.2", .doc = "verifyQmm Flash Next shapes µbench: stock vs plain vs NAX per width (MLX_SERVE_VQMM_FN_UBENCH=1)" };
    pub const MLX_SERVE_VQMM_MIXED_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.7.10", .doc = "verifyQmm mixed-width NAX µbench (MLX_SERVE_VQMM_MIXED_UBENCH=1)" };
    pub const MLX_SERVE_VQMM_PARITY_DEBUG: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.8.10", .doc = "test input read by `expectVerifyQmmNoWorseThanStock`" };
    pub const MLX_SERVE_VQMM_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/linear", .since = "26.7.7", .doc = "verifyQmm µbench: kernel vs stock per 27B shape (MLX_SERVE_VQMM_UBENCH=1)" };
    pub const MOE_UBENCH: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/blocks/moe", .since = "26.10.2", .doc = "moe gather kernels: microbench (MOE_UBENCH=1)" };
    pub const MOSS_IMAGE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/moss_image", .since = "26.10.3", .doc = "MOSS-Image: the prompt is the release template over one trimmed user turn, thinking on" };
    pub const MOSS_IMAGE_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/moss_image", .since = "26.10.3", .doc = "MOSS-Image: the prompt is the release template over one trimmed user turn, thinking on" };
    pub const MTPLX_DSV41_BOX_BASELINE_GB: Var = .{ .kind = .fixture, .type = .float, .default = "unset", .owner = "plugins/deepseek_v41/stream", .since = "26.10.2", .doc = "dsv41 cell: the Python tier's box baseline, stamped on the receipt" };
    pub const MTP_LMHEAD_TEST_FILE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/blocks/linear", .since = "26.9.3", .doc = "verify vocabulary projections preserve solo logits across request widths (MTP_LMHEAD_TEST_FILE)" };
    pub const MTP_REPLAY_REPORT: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "engines/mlx/spec", .since = "26.9.5", .doc = "print the MTP depth-policy replay report" };
    pub const MTP_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx/spec/mtp", .since = "26.9.3", .doc = "plain to prime resumes the same MTP history and next draft as legacy hidden capture" };
    pub const MUSE_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/muse_glimmer", .since = "26.8.5", .doc = "muse vision live: tower parity vs the executed reference" };
    pub const MUSE_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/muse_glimmer", .since = "26.8.5", .doc = "muse vision live: tower parity vs the executed reference" };
    pub const MUSIC3_COST_FRAMES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/music3", .since = "26.8.6", .doc = "music3 frame cost probe (env-gated, informational)" };
    pub const MUSIC3_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/music3", .since = "26.8.6", .doc = "Music3 reference fixtures dir" };
    pub const MUSIC3_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/music3", .since = "26.8.6", .doc = "Music3 pack for the live tests" };
    pub const MUSIC3_WAV_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/music3", .since = "26.8.6", .doc = "music3 wav dump harness (env-gated, fixed seed)" };
    pub const NEMOTRON_DIARIZATION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_diarization", .since = "26.10.3", .doc = "nemotron diarization: features, speaker activity and the cache match the references" };
    pub const NEMOTRON_DIARIZATION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_diarization", .since = "26.10.3", .doc = "nemotron diarization: features, speaker activity and the cache match the references" };
    pub const PIXAL3D_DINOV3: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d oracle: crop, DINOv3 tokens, NAF and the projection reproduce every stage's conditioning…" };
    pub const PIXAL3D_FACES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "Pixal3D reference fixtures dir" };
    pub const PIXAL3D_FOV_DEG: Var = .{ .kind = .fixture, .type = .float, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_IMAGE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d oracle: crop, DINOv3 tokens, NAF and the projection reproduce every stage's conditioning…" };
    pub const PIXAL3D_NAF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "naf oracle: the upsampled map at sampled pixels matches the reference (PIXAL3D_FIXTURES…" };
    pub const PIXAL3D_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_RES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_TEX: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_VIEWS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "pixal3d end to end: a cutout (PIXAL3D_IMAGE) or posed views (PIXAL3D_VIEWS) become a textured GLB…" };
    pub const PIXAL3D_WEIGHTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/pixal3d", .since = "26.10.3", .doc = "test input read by `ditOracle`" };
    pub const QV_GH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const QV_GW: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const QWEN3_ALIGNER_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-aligner: the prompt, the classes and the times match the reference" };
    pub const QWEN3_ALIGNER_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-aligner: the prompt, the classes and the times match the reference" };
    pub const QWEN3_ASR_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-asr: features and audio tower match the reference" };
    pub const QWEN3_ASR_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_asr", .since = "26.10.3", .doc = "qwen3-asr: features and audio tower match the reference" };
    pub const QWEN4_EMBEDDED_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx", .since = "26.10.2", .doc = "local oQ Qwen4 metadata defers convention until loaded weights" };
    pub const QWEN4_EXTERNAL_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "engines/mlx", .since = "26.10.2", .doc = "local legacy Qwen4 metadata keeps external folded default" };
    pub const QWEN4_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 batched decode: one forwardMoeBatchedDecode tick == two serial ticks (GDN + PLE history + QSA…" };
    pub const QWEN4_FIXTURE_BF16: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 fixture bf16: residual streams track a bf16 reference (QWEN4_FIXTURE_BF16)" };
    pub const QWEN4_FIXTURE_ROWS: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "sdk/testing", .since = "26.8.11", .doc = "print per-row parity numbers" };
    pub const QWEN4_FIXTURE_VISION: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "qwen4 fixture vision: tower parity, pre-tile splice + M-RoPE prefill/decode, QSA mask at 3-D block…" };
    pub const QWEN4_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen4_exp", .since = "26.8.11", .doc = "row-axis verify: every row of a group verify is byte-identical to its solo verify" };
    pub const QWEN4_YARN_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.9.1", .doc = "qwen4 yarn parity vs HF and vLLM (QWEN4_YARN_FIXTURES)" };
    pub const QWEN_IMAGE_E2E_GUIDANCE: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_E2E_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage load leaves the DiT and VAE resident (env-gated)" };
    pub const QWEN_IMAGE_E2E_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_E2E_SIZE: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_E2E_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage e2e on a real pack (env-gated)" };
    pub const QWEN_IMAGE_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage edit expandImagePads expansion counts" };
    pub const QWEN_IMAGE_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen_image", .since = "26.9.5", .doc = "QwenImage edit expandImagePads expansion counts" };
    pub const QWEN_MID_SYSTEM_MODEL_DIR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/chat", .since = "26.10.2", .doc = "mid-system: real Qwen pack loaded through ChatConfig preserves tool-round prefixes" };
    pub const QWEN_PREPROCESS_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.7.11", .doc = "qwen preprocessing parity vs reference pixel_values (QWEN_PREPROCESS_FIXTURE)" };
    pub const QWEN_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const QWEN_VISION_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "sdk/vision", .since = "26.6.12", .doc = "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" };
    pub const RADIO_VISION_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_h", .since = "26.10.3", .doc = "radio full checkpoint: image/video embeddings and EVS match the official fp32 oracle" };
    pub const RADIO_VISION_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/nemotron_h", .since = "26.10.3", .doc = "radio full checkpoint: image/video embeddings and EVS match the official fp32 oracle" };
    pub const REPLAY_DUMP_SOFT: Var = .{ .kind = .fixture, .type = .flag, .default = "off", .owner = "host/chat", .since = "26.7.5", .doc = "tool traffic replay: captured agent traffic survives parse + schema coercion" };
    pub const SA3_CROSS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_DIT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SA3_GLOBAL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SA3_PATCH: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SA3_T5: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "test input read by `testEngine`" };
    pub const SA3_TOK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" };
    pub const SA3_X: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/stable_audio3", .since = "26.10.2", .doc = "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" };
    pub const SINGLE_MTP_TEST_CONTEXT: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/qwen4_exp", .since = "26.9.3", .doc = "single-stream verify optimizations preserve native outputs and captures (QWEN4_TEST_MODEL)" };
    pub const SUSHI_EXL3_K3_FIXTURE: Var = .{ .kind = .fixture, .type = .text, .default = "src/sdk/quant/exl3/sushi/fixtures/exl3_k3_linear.safetensors", .owner = "sdk/quant/exl3/stream", .since = "26.10.2", .doc = "sushi K3 mul1 fixture for mlx-stream's EXL3 parity" };
    pub const TOKENIZER_PARITY_CASES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.10.2", .doc = "tokenizer parity: encode() matches HF `tokenizers` ids (TOKENIZER_PARITY_MODEL)" };
    pub const TOKENIZER_PARITY_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "host/tokenizer", .since = "26.10.2", .doc = "tokenizer parity: encode() matches HF `tokenizers` ids (TOKENIZER_PARITY_MODEL)" };
    pub const TRELLIS2_DUMP: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_FACES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_FIXTURES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "TRELLIS.2 reference fixtures dir" };
    pub const TRELLIS2_IMAGE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_LOAD: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_PACK: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_RES: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_SS_DEC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 oracle: the sparse-structure decoder matches the reference logits (TRELLIS2_SS_DEC…" };
    pub const TRELLIS2_STEPS: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_TEX: Var = .{ .kind = .fixture, .type = .int, .default = "the test's own", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "trellis2 end to end: an RGBA cutout becomes a textured GLB (TRELLIS2_PACK, TRELLIS2_IMAGE…" };
    pub const TRELLIS2_WEIGHTS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the tests skip", .owner = "plugins/trellis2", .since = "26.10.3", .doc = "TRELLIS.2 weights dir" };
    pub const TTS_FRAMES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "talker+predictor reproduce greedy oracle codes" };
    pub const TTS_INPUT_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "talker+predictor reproduce greedy oracle codes" };
    pub const TTS_PARITY_8BIT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.7.2", .doc = "8-bit checkpoint parity vs bf16 (greedy, cloned voice)" };
    pub const TTS_PARITY_BF16: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.7.2", .doc = "8-bit checkpoint parity vs bf16 (greedy, cloned voice)" };
    pub const TTS_QUANT_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.7.2", .doc = "8-bit quantized checkpoint loads and produces non-silent audio" };
    pub const TTS_REF_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "codec decoder reproduces reference audio" };
    pub const TTS_REF_CODES: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "talker+predictor reproduce greedy oracle codes" };
    pub const TTS_SPK_EMB: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "speaker encoder (ECAPA-TDNN) matches the Python oracle" };
    pub const TTS_SPK_MEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "speaker encoder (ECAPA-TDNN) matches the Python oracle" };
    pub const TTS_SPK_REF: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "speaker encoder (ECAPA-TDNN) matches the Python oracle" };
    pub const TTS_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "generateCodes yields at frame boundaries and stops on cancel" };
    pub const TTS_WAV_OUT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/qwen3_tts", .since = "26.6.13", .doc = "end-to-end native TTS produces non-silent audio" };
    pub const YUE2_AR_LOGITS0: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_AR_LOGITS1: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_AR_NEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_AR_PREFIX: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 AR path: prefill and a cached decode step match the reference logits" };
    pub const YUE2_NAR_AR: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_CODEC: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_PREFIX_LEN: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_STATE: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_NAR_V: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 NAR path: velocity and the seeded 4-step solve match the reference" };
    pub const YUE2_TEST_MODEL: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "test input read by `testEngine`" };
    pub const YUE2_TOK_IDS: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 tokenizer: the tiktoken ranks encode like the reference" };
    pub const YUE2_TOK_TEXT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 tokenizer: the tiktoken ranks encode like the reference" };
    pub const YUE2_VAE_AUDIO: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 VAE: two tiles of random latents decode like the reference" };
    pub const YUE2_VAE_LAT: Var = .{ .kind = .fixture, .type = .text, .default = "unset: the test skips", .owner = "plugins/yue2", .since = "26.10.2", .doc = "yue2 VAE: two tiles of random latents decode like the reference" };
    pub const _GPU_WINDOW_LOCKED: Var = .{ .kind = .fixture, .type = .flag, .default = "unset: the GPU test refuses", .owner = "plugins", .since = "26.10.2", .doc = "set by the GPU window orchestrator: the GPU-lock-held child may run device tests" };
};

// ── tests ──

const testing = std.testing;

test "value readers: on is set-and-not-0, int and float fall back on empty or garbage" {
    try testing.expect(!onValue(null));
    try testing.expect(!onValue("0"));
    try testing.expect(!onValue("0x"));
    try testing.expect(onValue(""));
    try testing.expect(onValue("1"));
    try testing.expect(onValue("on"));
    try testing.expectEqual(@as(usize, 7), intValue(usize, null, 7));
    try testing.expectEqual(@as(usize, 7), intValue(usize, "", 7));
    try testing.expectEqual(@as(usize, 7), intValue(usize, "x", 7));
    try testing.expectEqual(@as(usize, 42), intValue(usize, "42", 7));
    try testing.expectEqual(@as(f32, 0.5), floatValue("x", 0.5));
    try testing.expectEqual(@as(f32, 0.5), floatValue("-1", 0.5));
    try testing.expectEqual(@as(f32, 0.5), floatValue("inf", 0.5));
    try testing.expectEqual(@as(f32, 2.5), floatValue("2.5", 0.5));
}

test "MLX_SERVE_DIAG: a bare token reads 1, token=value reads the value, a prefix matches nothing" {
    var d: DiagList = .{};
    d.load("step-trace,decode-graph-dump=/tmp/g,empty=");
    try testing.expectEqualStrings("1", std.mem.span(d.find("step-trace").?));
    try testing.expectEqualStrings("/tmp/g", std.mem.span(d.find("decode-graph-dump").?));
    try testing.expectEqualStrings("", std.mem.span(d.find("empty").?));
    try testing.expect(d.find("step") == null);
    try testing.expect(d.find("decode-graph") == null);
    d.load(null);
    try testing.expect(d.find("step-trace") == null);
    try testing.expectEqualStrings("step-trace", comptime diagToken("MLX_SERVE_STEP_TRACE"));
    try testing.expectEqualStrings("qwen4-gdn-check", comptime diagToken("QWEN4_GDN_CHECK"));
}

test "registry: diag tokens are unique and every entry carries an owner, a release and a doc" {
    comptime {
        @setEvalBranchQuota(1_000_000);
        const names = std.meta.declarations(vars);
        for (names, 0..) |a, i| {
            const v = @field(vars, a);
            if (v.owner.len == 0 or v.since.len == 0 or v.doc.len == 0) @compileError(a ++ ": owner, since and doc are required");
            if (v.kind != .diag) continue;
            for (names[i + 1 ..]) |b| {
                if (@field(vars, b).kind == .diag and std.mem.eql(u8, diagToken(a), diagToken(b)))
                    @compileError(a ++ " and " ++ b ++ " share a diag token");
            }
        }
    }
}

test "init snapshots public variables, set refreshes the snapshot, MLX_SERVE_DIAG arms a diag by token" {
    const dump = override(.MLX_SERVE_RAW_DUMP_FILE, "/a");
    const trace = override(.MLX_SERVE_STEP_TRACE, null);
    const diag = override(.MLX_SERVE_DIAG, "step-trace");
    defer {
        diag.restore();
        trace.restore();
        dump.restore();
        snapshot_taken = false;
    }
    try testing.expect(!on(.MLX_SERVE_STEP_TRACE)); // before init the list is not read
    init();
    try testing.expectEqualStrings("/a", span(.MLX_SERVE_RAW_DUMP_FILE).?);
    set(.MLX_SERVE_RAW_DUMP_FILE, "/b");
    try testing.expectEqualStrings("/b", span(.MLX_SERVE_RAW_DUMP_FILE).?);
    try testing.expect(on(.MLX_SERVE_STEP_TRACE));
    set(.MLX_SERVE_STEP_TRACE, "0"); // the variable itself wins over the list
    try testing.expect(!on(.MLX_SERVE_STEP_TRACE));
}

test "override restores what was there, set or unset" {
    {
        const o = override(.MLX_SERVE_DSV4_DSPARK, "0");
        try testing.expectEqualStrings("0", span(.MLX_SERVE_DSV4_DSPARK).?);
        const inner = override(.MLX_SERVE_DSV4_DSPARK, null);
        try testing.expect(get(.MLX_SERVE_DSV4_DSPARK) == null);
        inner.restore();
        try testing.expectEqualStrings("0", span(.MLX_SERVE_DSV4_DSPARK).?);
        o.restore();
    }
}
