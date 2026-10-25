//! Per-token logprob records as an engine reports them and the API surfaces render them (OpenAI format).

/// One alternative: a token id and its logprob.
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// The returned token's logprob and its top alternatives (`top_logprobs` is its owner's to free).
pub const LogprobResult = struct {
    token_logprob: f32,
    top_logprobs: []TokenLogprob,
};
