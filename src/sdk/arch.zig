//! The `arch` kind: one erased table per arch, built at comptime from the plugin's namespace. Resolved once: `claims`
//! at discovery, `parse` at config parse; the host copies `caps` into its own tables there. A hook that fails
//! refuses the model by name.

const std = @import("std");
const peek = @import("peek.zig");
const check = @import("check.zig");

const Allocator = std.mem.Allocator;

/// Facts the host copies once at load into its own tables.
pub const Caps = struct {
    /// G1: per-request decode state lives on the arch's module, not the host's KVCache: no prefix-cache restore,
    /// no batched decode, single-flight admission, no KVCache snapshot or rewind.
    owns_decode_state: bool = false,
    /// The host sends the whole prompt in one forward; the arch chunks it.
    prefill_whole_prompt: bool = false,
    /// The prompt forward returns the last row's logits: no separate one-row forward.
    prefill_yields_last_logits: bool = false,
    /// The proposal's opt-in; an arch that owns its decode state is never batched.
    batches_decode: bool = false,
    /// The loader reads the resident weights past the page cache unless the model setting says otherwise.
    residents_past_page_cache: bool = false,
};

/// The parsed model's facts the host keeps on its own config.
pub const Shell = struct { num_experts: u32 = 0, num_layers: u32 = 0 };

pub const Arch = struct {
    name: []const u8,
    caps: Caps,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,
    /// The arch's own config (refusals by name through `diag`); owned by the host's config, freed by `free_config`.
    parse: *const fn (gpa: Allocator, p: *const peek.ConfigPeek, diag: *peek.Diag) anyerror!*anyopaque,
    free_config: *const fn (gpa: Allocator, cfg: *anyopaque) void,
    shell: *const fn (cfg: *const anyopaque) Shell,

    /// The table of `T`, a namespace declaring the arch (a missing or mistyped declaration is a compile error
    /// naming it): name, caps, claims, Config, parse, freeConfig, shell.
    pub fn of(comptime T: type) Arch {
        comptime {
            const w = "arch " ++ @typeName(T);
            check.nameDecl(w, T);
            check.valueDecl(w, T, "caps", Caps);
            if (T.caps.owns_decode_state and T.caps.batches_decode) @compileError(w ++ ": batches_decode with owns_decode_state");
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
            check.typeDecl(w, T, "Config");
            check.fnDecl(w, T, "parse", &.{ Allocator, *const peek.ConfigPeek, *peek.Diag }, *T.Config);
            check.fnDecl(w, T, "freeConfig", &.{ Allocator, *T.Config }, void);
            check.fnDecl(w, T, "shell", &.{*const T.Config}, Shell);
        }
        const W = struct {
            fn cfgOf(cfg: *anyopaque) *T.Config {
                return @ptrCast(@alignCast(cfg));
            }
            fn constCfg(cfg: *const anyopaque) *const T.Config {
                return @ptrCast(@alignCast(cfg));
            }
            fn parse(gpa: Allocator, p: *const peek.ConfigPeek, diag: *peek.Diag) anyerror!*anyopaque {
                return try T.parse(gpa, p, diag);
            }
            fn freeConfig(gpa: Allocator, cfg: *anyopaque) void {
                T.freeConfig(gpa, cfgOf(cfg));
            }
            fn shell(cfg: *const anyopaque) Shell {
                return T.shell(constCfg(cfg));
            }
        };
        return .{
            .name = T.name,
            .caps = T.caps,
            .claims = T.claims,
            .parse = W.parse,
            .free_config = W.freeConfig,
            .shell = W.shell,
        };
    }
};
