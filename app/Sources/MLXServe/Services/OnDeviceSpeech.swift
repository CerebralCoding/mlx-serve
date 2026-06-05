import Foundation

/// Voice mode's privacy contract: speech is transcribed **on-device** so audio
/// never leaves the Mac and recognition works with no internet. `SFSpeechRecognizer`
/// can fall back to Apple's servers, so we force on-device and, when the on-device
/// model genuinely isn't available, refuse rather than go online behind the user's
/// back. This pure helper holds that decision so it can be unit-tested without the
/// Speech framework.
enum OnDeviceSpeech {
    /// Returns `nil` when on-device recognition is available (caller proceeds and
    /// forces `requiresOnDeviceRecognition = true`), or a user-facing error message
    /// when it isn't — so Voice mode fails loudly instead of streaming audio to
    /// Apple's servers.
    static func unavailableMessage(supportsOnDevice: Bool, locale: String) -> String? {
        guard !supportsOnDevice else { return nil }
        return "On-device speech recognition isn't installed for \(locale). " +
            "Voice mode keeps your audio on your Mac, so it won't fall back to Apple's servers. " +
            "Add the language under System Settings → Keyboard → Dictation (or General → Language & Region), then try again."
    }
}
