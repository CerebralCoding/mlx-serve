import AppKit
import SwiftUI

/// The tray's transcript switch: on records the microphone, off sends the
/// recording to the local Qwen3-ASR and opens it as Markdown.
@MainActor
final class TrayTranscript: ObservableObject {
    static let shared = TrayTranscript()
    static let asrRepo = "mlx-community/Qwen3-ASR-0.6B-8bit"
    static let folder = URL(fileURLWithPath: NSString(string: "~/.mlx-serve/transcripts").expandingTildeInPath)

    enum Phase: Equatable { case idle, recording(since: Date), transcribing, note(String) }
    @Published private(set) var phase = Phase.idle
    /// Not observed: its meter would redraw the tray many times a second.
    private let recorder = AudioRecorder()
    private let desktopAudio = DesktopAudio()

    /// Also record what the Mac plays, from the moment it is ticked; remembered.
    @Published var desktop = UserDefaults.standard.bool(forKey: "transcriptDesktopAudio") {
        didSet {
            UserDefaults.standard.set(desktop, forKey: "transcriptDesktopAudio")
            if case .recording(let since) = phase { desktop ? startDesktop(since) : desktopAudio.stop() }
        }
    }

    func start(voice: VoiceModeController) async {
        guard await AudioRecorder.requestPermission() else { phase = .note("Microphone access is off"); return }
        // Voice mode would hear, and answer, what is being transcribed.
        if voice.isActive { voice.end() }
        do {
            try recorder.start()
            let since = Date()
            phase = .recording(since: since)
            if desktop { startDesktop(since) }
        } catch {
            phase = .note(error.localizedDescription)
        }
    }

    /// A tap that will not start (no System Audio Recording access) unticks the box.
    private func startDesktop(_ since: Date) {
        do {
            try desktopAudio.start(at: Int(Date().timeIntervalSince(since) * AudioRecorder.targetSampleRate))
        } catch {
            desktop = false
        }
    }

    func stop(server: String, model: String) async {
        let desk = desktopAudio.take()
        guard case .recording(let since) = phase, let pcm = recorder.stop() else { phase = .idle; return }
        phase = .transcribing
        let samples = Self.mix(AudioReference.floatSamples(from: pcm), desk)
        let rate = Int(AudioRecorder.targetSampleRate)
        let stamp = DateFormatter()
        stamp.locale = Locale(identifier: "en_US_POSIX")
        stamp.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let title = stamp.string(from: since)
        let file = Self.folder.appendingPathComponent(title.replacingOccurrences(of: ":", with: "."))
        try? FileManager.default.createDirectory(at: Self.folder, withIntermediateDirectories: true)
        do {
            var segments: [(start: Double, text: String)] = []
            var start = 0
            for end in Self.pieceEnds(samples, rate: rate) {
                let wav = AudioReference.wavData(fromMonoFloat: Array(samples[start..<end]), sampleRate: rate)
                let offset = Double(start) / Double(rate)
                segments += try await Self.transcribe(wav, model: model, server: server).map { (offset + $0.start, $0.text) }
                start = end
            }
            segments.removeAll { $0.text.isEmpty }
            guard !segments.isEmpty else { phase = .note("Nothing was heard"); return }
            try Self.markdown(title: title, segments: segments).write(to: file.appendingPathExtension("md"), atomically: true, encoding: .utf8)
            NSWorkspace.shared.open(file.appendingPathExtension("md"))
            phase = .idle
        } catch {
            // The recording stays beside the transcripts, so a failure loses nothing.
            try? AudioReference.wavData(fromMonoFloat: samples, sampleRate: rate).write(to: file.appendingPathExtension("wav"))
            phase = .note(error.localizedDescription)
        }
    }

    /// The microphone with the desktop audio on top, for as long as the microphone ran.
    nonisolated static func mix(_ mic: [Float], _ desktop: [Float]) -> [Float] {
        desktop.isEmpty ? mic : mic.indices.map { mic[$0] + ($0 < desktop.count ? desktop[$0] : 0) }
    }

    /// Where to cut a recording into requests the server takes (20 minutes at
    /// most): the quietest 100 ms of the last 5 s before each limit.
    nonisolated static func pieceEnds(_ samples: [Float], rate: Int, maxSeconds: Int = 1200) -> [Int] {
        let limit = maxSeconds * rate, window = rate / 10
        var ends: [Int] = [], start = 0
        while samples.count - start > limit {
            var cut = start + limit, quietest = Float.infinity
            for p in stride(from: start + limit - 5 * rate, to: start + limit - window, by: window / 2) {
                let loudness = samples[p..<p + window].reduce(0) { $0 + abs($1) }
                if loudness < quietest { quietest = loudness; cut = p + window / 2 }
            }
            ends.append(cut)
            start = cut
        }
        return ends + [samples.count]
    }

    /// The transcript file: when it started, then the text, each segment led by
    /// its time in the recording when there is more than one.
    nonisolated static func markdown(title: String, segments: [(start: Double, text: String)]) -> String {
        let body = segments.count == 1 ? segments[0].text
            : segments.map { "**\(clock($0.start))** \($0.text)" }.joined(separator: "\n\n")
        return "# \(title)\n\n\(body)\n"
    }

    nonisolated static func clock(_ seconds: Double) -> String {
        let s = Int(seconds)
        return s < 3600 ? String(format: "%02d:%02d", s / 60, s % 60) : String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
    }

    struct Segment: Decodable { let start: Double; let text: String }
    private struct Verbose: Decodable { let segments: [Segment] }
    private struct Failure: Decodable {
        struct Detail: Decodable { let message: String }
        let error: Detail
    }

    /// One `verbose_json` request: what the server heard, timed from the start of `wav`.
    nonisolated static func transcribe(_ wav: Data, model: String, server: String) async throws -> [Segment] {
        let boundary = UUID().uuidString
        var body = Data()
        for (name, value) in [("model", model), ("response_format", "verbose_json")] {
            body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\nContent-Type: audio/wav\r\n\r\n".utf8))
        body.append(wav)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        var request = URLRequest(url: URL(string: server + "/v1/audio/transcriptions")!, timeoutInterval: 600)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let message = (try? JSONDecoder().decode(Failure.self, from: data))?.error.message ?? "Transcription failed"
            throw NSError(domain: "Transcript", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
        }
        return try JSONDecoder().decode(Verbose.self, from: data).segments
    }
}

/// A third switch beside Voice and the Quick Launcher; its folder button opens the saved transcripts.
struct TranscriptTrayRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager
    @ObservedObject private var transcript = TrayTranscript.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            row
            if recording {
                Toggle("Also record desktop audio", isOn: $transcript.desktop)
                    .toggleStyle(.checkbox)
                    .font(.app(.caption))
                    .padding(.leading, 26)
            }
        }
    }

    private var row: some View {
        // The aligner's config says qwen3_asr too.
        let asr = appState.localModels.first { $0.modelType == "qwen3_asr" && DownloadManager.isAsrPack($0.path) }
        return TrayFeatureRow(icon: "record.circle", title: "Transcript", subtitle: subtitle(asr != nil), isOn: recording, tint: .red) {
            Button {
                try? FileManager.default.createDirectory(at: TrayTranscript.folder, withIntermediateDirectories: true)
                NSWorkspace.shared.open(TrayTranscript.folder)
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.plain)
            .help("Open the transcripts folder")
            if let asr {
                Toggle("", isOn: Binding(get: { recording }, set: { on in
                    Task {
                        if on { await transcript.start(voice: appState.voice) }
                        else { await transcript.stop(server: appState.server.baseURL, model: asr.path) }
                    }
                }))
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
                // Stopping always works; starting needs the server.
                .disabled(transcript.phase == .transcribing || !recording && appState.server.status != .running)
            } else {
                Button("Download") { downloads.start(repoId: TrayTranscript.asrRepo) { appState.refreshModels() } }
                    .controlSize(.small)
                    .disabled(downloads.downloads[TrayTranscript.asrRepo]?.status == .downloading)
            }
        }
    }

    private var recording: Bool {
        if case .recording = transcript.phase { return true }
        return false
    }

    private func subtitle(_ hasAsr: Bool) -> String {
        switch transcript.phase {
        case .idle: return hasAsr ? "Record, then open as Markdown" : "Needs Qwen3-ASR"
        case .recording(let since): return L10n.format("Recording since %@", since.formatted(date: .omitted, time: .shortened))
        case .transcribing: return "Transcribing…"
        case .note(let note): return note
        }
    }
}
