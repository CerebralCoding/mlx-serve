//! sdk/blocks/gdn/gate, split out of engines/mlx/transformer.zig (P1, moves only).
//! Owns compileGdnGate, gdnGateClosureCallback, computeGdnGate, gdnChainGateDtype, gdnGateChain.

const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("decode.zig");
const log = @import("../../../host/log.zig");
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;

/// Compile the GatedDeltaNet gating chain (astype→exp→add→astype→exp→log1p→
/// multiply→negative→exp→astype, ~10 dispatches) into one fused kernel.
/// shapeless=true: pure elementwise chain, same trace for any [B,S,Hv].
pub fn compileGdnGate(self: *Transformer) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &gdnGateClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, true);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_gdn_gate = compiled;
        log.info("GDN gate compiled (kernel fusion enabled); recurrent state {s}, gate {s}\n", .{ @tagName(self.ssmStateDtype()), @tagName(gdn_decode.gateDtype(self.config.actDtype(), self.ssmStateDtype())) });
    }
}

pub fn gdnGateClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    var A_log = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(A_log);
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
    defer _ = mlx.mlx_array_free(a);
    var dt_bias = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&dt_bias, input, 2) != 0) return -1;
    defer _ = mlx.mlx_array_free(dt_bias);

    const g = gdnGateChain(A_log, a, dt_bias, gdnChainGateDtype(mlx.mlx_array_dtype(a), self.ssmStateDtype()), self.s) catch return -1;
    const out_arr = [_]mlx.mlx_array{g};
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
    _ = mlx.mlx_array_free(g);
    return 0;
}

/// Apply the compiled GDN gate closure if available, else the raw chain.
/// Returns owned g (the state's dtype, shape of `a`).
pub fn computeGdnGate(self: *const Transformer, A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array) !mlx.mlx_array {
    if (self.compiled_gdn_gate) |compiled| {
        const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        if (mlx.mlx_vector_array_size(out_vec) == 1) {
            var g = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&g, out_vec, 0));
            return g;
        }
    }
    return gdnGateChain(A_log, a, dt_bias, gdnChainGateDtype(mlx.mlx_array_dtype(a), self.ssmStateDtype()), self.s);
}

/// MoE routing chain (negate→argpartition→slice→softmax→take→sum→expand→divide).
/// Free-function variant of `Transformer.moeRoutingUncompiled` so unit tests can
/// exercise the pure subgraph without constructing a full Transformer. Returns
/// owned `inds` (int32, [..., k]) and `norm_scores` (bf16, [..., k]) — caller
/// must free both.
/// GatedDeltaNet gating chain: g = exp(-exp(A_log) * softplus(a + dt_bias)),
/// computed in float32 and returned at `g_dtype`, the recurrent state's. Mirrors
/// mlx-lm's `compute_g` (which is `@mx.compile`d). Pure — serves as both the
/// compiled-closure body and the uncompiled fallback. Returns owned array.
/// The composed chain's gate dtype: an f32 state under bf16 activations takes an f32 gate;
/// every other pack keeps the bf16 gate this chain has always returned.
pub fn gdnChainGateDtype(act: mlx.mlx_dtype, state: mlx.mlx_dtype) mlx.mlx_dtype {
    return if (act == .bfloat16) state else .bfloat16;
}

pub fn gdnGateChain(A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array, g_dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var A_log_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(A_log_f32);
    try mlx.check(mlx.mlx_astype(&A_log_f32, A_log, .float32, s));
    var exp_A = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_A);
    try mlx.check(mlx.mlx_exp(&exp_A, A_log_f32, s));

    // softplus(a + dt_bias) = log1p(exp(a + dt_bias)); an f32 gate adds in f32 too,
    // as the reference's `a.float() + dt_bias` does.
    var a_plus_dt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_plus_dt);
    if (g_dtype == .float32) {
        var a32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a32);
        var dt32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt32);
        try mlx.check(mlx.mlx_astype(&a32, a, .float32, s));
        try mlx.check(mlx.mlx_astype(&dt32, dt_bias, .float32, s));
        try mlx.check(mlx.mlx_add(&a_plus_dt, a32, dt32, s));
    } else try mlx.check(mlx.mlx_add(&a_plus_dt, a, dt_bias, s));
    var a_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_f32);
    try mlx.check(mlx.mlx_astype(&a_f32, a_plus_dt, .float32, s));
    var exp_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_a);
    try mlx.check(mlx.mlx_exp(&exp_a, a_f32, s));
    var sp_inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp_inner);
    try mlx.check(mlx.mlx_log1p(&sp_inner, exp_a, s));

    var neg_decay = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_decay);
    try mlx.check(mlx.mlx_multiply(&neg_decay, exp_A, sp_inner, s));
    var neg_neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_neg);
    try mlx.check(mlx.mlx_negative(&neg_neg, neg_decay, s));
    var g_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_f32);
    try mlx.check(mlx.mlx_exp(&g_f32, neg_neg, s));
    var g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_astype(&g, g_f32, g_dtype, s));
    return g;
}
