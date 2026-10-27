import Combine
import Foundation

final class FineTuneClient {
    let baseURL: URL
    private let session: URLSession

    init(baseURL: URL, session: URLSession? = nil) {
        self.baseURL = baseURL
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 86_400
        self.session = session ?? URLSession(configuration: config)
    }

    func createRequest(model: String, name: String, topic: String, keepEverythingElse: Bool) throws -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent("v1/finetune"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": model, "name": name, "topic": topic, "keep_everything_else": keepEverythingElse,
        ])
        return request
    }

    func jobRequest(_ id: String, method: String = "GET", suffix: String? = nil) throws -> URLRequest {
        guard id.hasPrefix("ft-"), !id.dropFirst(3).isEmpty,
              id.dropFirst(3).utf8.allSatisfy({ (48...57).contains($0) }) else { throw URLError(.badURL) }
        var url = baseURL.appendingPathComponent("v1/finetune").appendingPathComponent(id)
        if let suffix { url.appendPathComponent(suffix) }
        var request = URLRequest(url: url)
        request.httpMethod = method
        return request
    }

    func send(_ request: URLRequest) async throws -> FineTuneJob {
        try JSONDecoder().decode(FineTuneJob.self, from: await responseData(request))
    }

    func create(modelPath: String, name: String, topic: String, keepEverythingElse: Bool) async throws -> FineTuneJob {
        var load = URLRequest(url: baseURL.appendingPathComponent("v1/load-model"))
        load.httpMethod = "POST"
        load.setValue("application/json", forHTTPHeaderField: "Content-Type")
        load.httpBody = try JSONSerialization.data(withJSONObject: APIClient.loadModelBody(id: modelPath, setDefault: false))
        let data = try await responseData(load)
        guard let body = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = body["model"] as? [String: Any],
              let id = model["id"] as? String, !id.isEmpty else { throw URLError(.cannotParseResponse) }
        return try await send(createRequest(model: id, name: name, topic: topic, keepEverythingElse: keepEverythingElse))
    }

    private func responseData(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw APIError.badStatus(code: (response as? HTTPURLResponse)?.statusCode ?? 0,
                                     detail: APIError.errorDetail(fromBody: data))
        }
        return data
    }

    func status(_ id: String) async throws -> FineTuneJob { try await send(jobRequest(id)) }
    func cancel(_ id: String) async throws -> FineTuneJob { try await send(jobRequest(id, method: "DELETE")) }
    func discard(_ id: String) async throws -> FineTuneJob { try await send(jobRequest(id, method: "DELETE", suffix: "adapter")) }
    func apply(_ id: String) async throws -> FineTuneJob {
        try await send(jobRequest(id, method: "POST", suffix: "adapter"))
    }

    func reload(model: String) async throws {
        for endpoint in ["unload-model", "load-model"] {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/\(endpoint)"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let body = endpoint == "load-model" ? APIClient.loadModelBody(id: model, setDefault: false) : ["model": model]
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            _ = try await responseData(request)
        }
    }

    func events(_ id: String) -> AsyncThrowingStream<FineTuneJob, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                do {
                    var request = try self.jobRequest(id, suffix: "events")
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    let (bytes, response) = try await self.session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                        var data = Data()
                        for try await byte in bytes { data.append(byte); if data.count >= 16_384 { break } }
                        throw APIError.badStatus(code: (response as? HTTPURLResponse)?.statusCode ?? 0,
                                                 detail: APIError.errorDetail(fromBody: data))
                    }
                    var parser = FineTuneEventParser()
                    // AsyncBytes.lines omits the empty lines that delimit SSE events.
                    for try await byte in bytes {
                        try Task.checkCancellation()
                        if let job = try parser.byte(byte) {
                            continuation.yield(job)
                            if job.status.terminal { continuation.finish(); return }
                        }
                    }
                    throw URLError(.networkConnectionLost)
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in producer.cancel() }
        }
    }
}

struct FineTuneEventParser {
    private var data = ""
    private var pending = Data()
    mutating func byte(_ byte: UInt8) throws -> FineTuneJob? {
        if byte == 10 {
            if pending.last == 13 { pending.removeLast() }
            guard let text = String(data: pending, encoding: .utf8) else { throw URLError(.cannotDecodeContentData) }
            pending.removeAll(keepingCapacity: true)
            return try line(text)
        }
        pending.append(byte)
        if pending.count > 1_048_576 { throw URLError(.dataLengthExceedsMaximum) }
        return nil
    }
    mutating func line(_ line: String) throws -> FineTuneJob? {
        if line.isEmpty {
            defer { data = "" }
            return data.isEmpty ? nil : try JSONDecoder().decode(FineTuneJob.self, from: Data(data.utf8))
        }
        if line.hasPrefix("data:") {
            var value = line.dropFirst(5)
            if value.first == " " { value = value.dropFirst() }
            if !data.isEmpty { data += "\n" }
            data += value
            if data.utf8.count > 1_048_576 { throw URLError(.dataLengthExceedsMaximum) }
        }
        return nil
    }
}

@MainActor
final class FineTuneController: ObservableObject {
    @Published var state = FineTuneState()
    @Published private(set) var managing = false
    @Published private(set) var saved = false
    var request: CustomizeRequest?
    private var client: FineTuneClient?
    private var monitor: Task<Void, Never>?

    func attach(_ job: FineTuneJob, client: FineTuneClient) {
        self.client = client
        saved = false
        state.receive(job)
        watch()
    }

    func watch() {
        guard let client, let id = state.job?.id, state.busy else { return }
        monitor?.cancel()
        monitor = Task { [weak self] in
            for attempt in 0..<3 {
                do {
                    let snapshot = try await client.status(id)
                    guard !Task.isCancelled else { return }
                    self?.state.receive(snapshot)
                    if snapshot.status.terminal { return }
                    for try await job in client.events(id) {
                        guard !Task.isCancelled else { return }
                        self?.state.receive(job)
                    }
                    return
                } catch {
                    guard !Task.isCancelled else { return }
                    if case APIError.badStatus(code: 404, detail: _) = error {
                        self?.state.jobMissing()
                        return
                    }
                    self?.state.error = "Progress disconnected: \(error.localizedDescription)"
                    if attempt < 2 { try? await Task.sleep(for: .seconds(2)) }
                }
            }
        }
    }

    func stop() async {
        guard let client, let id = state.job?.id else { return }
        state.requestStop()
        do { state.receive(try await client.cancel(id)) }
        catch {
            if case APIError.badStatus(code: 404, detail: _) = error { state.jobMissing() }
            else { state.stopFailed(error.localizedDescription) }
        }
    }

    func apply() async -> Bool {
        guard let client, let job = state.job, !managing else { return false }
        managing = true
        defer { managing = false }
        do {
            if !job.adapterApplied { state.receive(try await client.apply(job.id)) }
            try await client.reload(model: job.model)
            state.error = nil
            saved = true
            return true
        } catch { state.error = error.localizedDescription; return false }
    }

    func discard() async -> Bool {
        guard let client, let job = state.job, !managing else { return false }
        managing = true
        defer { managing = false }
        do { state.receive(try await client.discard(job.id)); return true }
        catch { state.error = error.localizedDescription; return false }
    }
}
