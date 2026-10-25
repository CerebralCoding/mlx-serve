//! Errors that cross a compile-unit boundary travel by name: every unit numbers its own error set, so a returned
//! `anyerror` value means something else on the other side.

const std = @import("std");

/// Every error the host tells apart from an engine call. Anything else arrives as `error.EngineFailed`, its name in
/// `last` for the message.
pub const Known = error{
    EmptyPrompt,
    EngineBusy,
    EngineUnsupported,
    ForeignBytesReaderBusy,
    GenerationFailed,
    GenerationOutOfMemory,
    GpuContextLost,
    InsufficientMemory,
    LoadFailed,
    NoDefaultModel,
    NotEnoughMemory,
    NotLoaded,
    OutOfMemory,
    PrefillDoesNotFit,
    PromptLogprobsUnavailable,
    Shutdown,
    UnknownModelId,
    WriteFailed,
    QueueFull,
    ModelBusy,
    NotEmbeddingModel,
    NoVisionEncoder,
    Timeout,
    EmbedFailed,
    VisionEncodeFailed,
    ModelNotReady,
    SlotFailed,
    Open,
    VideoUnsupported,
    EngineFailed,
};

/// The name of the last unknown error this thread received.
pub threadlocal var last: []const u8 = "";

/// The error `name` stands for on this side.
pub fn toError(name: []const u8) Known {
    inline for (@typeInfo(Known).error_set.?) |e| {
        if (std.mem.eql(u8, name, e.name)) return @field(Known, e.name);
    }
    last = name;
    return error.EngineFailed;
}

/// Turns a call's result name (null = success) into this side's error.
pub fn check(name: ?[]const u8) Known!void {
    if (name) |n| return toError(n);
}

test "err_name: a known name comes back as the same error; an unknown one keeps its name" {
    try std.testing.expectEqual(error.PrefillDoesNotFit, toError("PrefillDoesNotFit"));
    try std.testing.expectEqual(error.EngineFailed, toError("InvalidQwen4Indexer"));
    try std.testing.expectEqualStrings("InvalidQwen4Indexer", last);
    try check(null);
}
