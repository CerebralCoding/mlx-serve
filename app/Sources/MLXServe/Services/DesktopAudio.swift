import AVFoundation
import CoreAudio

/// Everything the Mac plays, through a Core Audio process tap, collected as
/// 16 kHz mono like `AudioRecorder`. macOS asks for System Audio Recording
/// the first time it starts.
final class DesktopAudio {
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "mlx-serve.desktop-audio")
    private var samples: [Float] = []
    private var tap = AudioObjectID(kAudioObjectUnknown)
    private var device = AudioObjectID(kAudioObjectUnknown)
    private var proc: AudioDeviceIOProcID?

    /// Collect from `offset` samples into the recording; silence fills the gap.
    func start(at offset: Int) throws {
        stop()
        lock.withLock { samples += repeatElement(0, count: max(0, offset - samples.count)) }
        do {
            let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
            description.uuid = UUID()
            description.isPrivate = true
            try check(AudioHardwareCreateProcessTap(description, &tap))
            var format = AudioStreamBasicDescription()
            var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
            var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat, mScope: kAudioObjectPropertyScopeGlobal,
                                                     mElement: kAudioObjectPropertyElementMain)
            try check(AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &format))
            let composition: [String: Any] = [
                kAudioAggregateDeviceNameKey: "MLX-Serve transcript",
                kAudioAggregateDeviceUIDKey: UUID().uuidString,
                kAudioAggregateDeviceIsPrivateKey: true,
                kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: description.uuid.uuidString]],
            ]
            try check(AudioHardwareCreateAggregateDevice(composition as CFDictionary, &device))
            guard let input = AVAudioFormat(streamDescription: &format),
                  let converter = AVAudioConverter(from: input, to: AudioRecorder.targetFormat) else { throw CocoaError(.featureUnsupported) }
            // Both channels into the mono track, not the left one alone.
            converter.downmix = true
            try check(AudioDeviceCreateIOProcIDWithBlock(&proc, device, queue) { [weak self] _, data, _, _, _ in
                guard let self, let buffer = AVAudioPCMBuffer(pcmFormat: input, bufferListNoCopy: data, deallocator: nil),
                      let out = AudioRecorder.resample(buffer, with: converter), let ch = out.floatChannelData else { return }
                lock.withLock { self.samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength))) }
            })
            try check(AudioDeviceStart(device, proc))
        } catch {
            stop()
            throw error
        }
    }

    /// Stop collecting; what was collected stays for `take()`.
    func stop() {
        if let proc {
            AudioDeviceStop(device, proc)
            AudioDeviceDestroyIOProcID(device, proc)
        }
        if device != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(device) }
        if tap != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tap) }
        (proc, device, tap) = (nil, AudioObjectID(kAudioObjectUnknown), AudioObjectID(kAudioObjectUnknown))
    }

    /// Stop, and hand over everything collected.
    func take() -> [Float] {
        stop()
        return lock.withLock {
            defer { samples = [] }
            return samples
        }
    }

    private func check(_ status: OSStatus) throws {
        if status != noErr { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
    }
}
