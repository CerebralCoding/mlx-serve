import XCTest
@testable import MLXCore

final class FineTuneClientTests: XCTestCase {
    private let snapshot = """
    {"id":"ft-1","model":"m","name":"n","status":"completed","stage":"verify",
    "step":1,"total":1,"cancel_requested":false,"preview":"done","adapter_discarded":false,"adapter_applied":false}
    """

    func testTransportDeliversTerminalSSEAndServerErrors() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FineTuneProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let client = FineTuneClient(baseURL: URL(string: "http://127.0.0.1:9999")!, session: session)
        let json = snapshot.replacingOccurrences(of: "\n", with: "")
        FineTuneProtocol.respond = { request in
            request.url?.path.hasSuffix("events") == true
                ? (200, "text/event-stream", ": keepalive\n\nevent: complete\ndata: \(json)\n\n")
                : (409, "application/json", "{\"error\":{\"message\":\"FineTuneOverlayExists\"}}")
        }
        var received: [FineTuneJob] = []
        for try await job in client.events("ft-1") { received.append(job) }
        XCTAssertEqual(received.count, 1)
        XCTAssertEqual(received.first?.status, .completed)
        do {
            _ = try await client.apply("ft-1")
            XCTFail("A conflict must not look like a save")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("FineTuneOverlayExists"))
        }
    }

    @MainActor
    func testApplyConflictKeepsResultAndDiscardWaitsForServer() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FineTuneProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let client = FineTuneClient(baseURL: URL(string: "http://127.0.0.1:9999")!, session: session)
        let controller = FineTuneController()
        controller.state.begin()
        controller.attach(try JSONDecoder().decode(FineTuneJob.self, from: Data(snapshot.utf8)), client: client)
        FineTuneProtocol.respond = { _ in (409, "application/json", "{\"error\":{\"message\":\"FineTuneOverlayExists\"}}") }
        let saved = await controller.apply()
        XCTAssertFalse(saved)
        XCTAssertFalse(controller.saved)
        XCTAssertEqual(controller.state.job?.name, "n")
        XCTAssertNotNil(controller.state.error)
        let discarded = snapshot.replacingOccurrences(of: "\"adapter_discarded\":false", with: "\"adapter_discarded\":true")
        FineTuneProtocol.respond = { _ in (200, "application/json", discarded) }
        let removed = await controller.discard()
        XCTAssertTrue(removed)
        XCTAssertEqual(controller.state.job?.adapterDiscarded, true)
    }

    func testCreationUsesRegistryIdReturnedForLocalPath() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FineTuneProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let client = FineTuneClient(baseURL: URL(string: "http://127.0.0.1:9999")!, session: session)
        let finished = snapshot
        FineTuneProtocol.respond = { request in
            let body = (try? JSONSerialization.jsonObject(with: request.httpBody ?? Data())) as? [String: Any]
            if request.url?.path == "/v1/load-model" {
                XCTAssertEqual(body?["model"] as? String, "/models/local-pack")
                return (200, "application/json", "{\"model\":{\"id\":\"registered-alias\"}}")
            }
            XCTAssertEqual(request.url?.path, "/v1/finetune")
            XCTAssertEqual(body?["model"] as? String, "registered-alias")
            return (202, "application/json", finished)
        }
        _ = try await client.create(modelPath: "/models/local-pack", name: "roses", topic: "Roses", keepEverythingElse: true)
    }
    func testRequestRoutesAndKeepEverythingElse() throws {
        let client = FineTuneClient(baseURL: URL(string: "http://127.0.0.1:9999")!)
        let request = try client.createRequest(model: "org/base", name: "roses", topic: "Answer gardening questions", keepEverythingElse: false)
        XCTAssertEqual(request.url?.path, "/v1/finetune")
        XCTAssertEqual(request.httpMethod, "POST")
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody)) as? [String: Any])
        XCTAssertEqual(body["keep_everything_else"] as? Bool, false)
        XCTAssertEqual(body["model"] as? String, "org/base")
        XCTAssertEqual(try client.jobRequest("ft-12", method: "DELETE").url?.path, "/v1/finetune/ft-12")
        XCTAssertEqual(try client.jobRequest("ft-12", method: "DELETE", suffix: "adapter").url?.path, "/v1/finetune/ft-12/adapter")
        XCTAssertThrowsError(try client.jobRequest("../elsewhere", method: "DELETE"))
    }

    func testSSECommentsAndEventBoundaries() throws {
        var parser = FineTuneEventParser()
        XCTAssertNil(try parser.line(": heartbeat"))
        XCTAssertNil(try parser.line("event: progress"))
        XCTAssertNil(try parser.line("data: {\"id\":\"ft-1\",\"model\":\"m\",\"name\":\"n\",\"status\":\"running\","))
        XCTAssertNil(try parser.line("data: \"stage\":\"synth\",\"step\":1,\"total\":128,\"cancel_requested\":false,\"preview\":\"rose\",\"adapter_discarded\":false,\"adapter_applied\":false}"))
        XCTAssertEqual(try parser.line("")?.preview, "rose")
        XCTAssertNil(try parser.line(""))
        XCTAssertNil(try parser.line("data: invalid"))
        XCTAssertThrowsError(try parser.line(""))
    }
    @MainActor
    func testApplyReloadsTheModelAndRetriesReloadFailureWithoutReinstalling() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FineTuneProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let client = FineTuneClient(baseURL: URL(string: "http://127.0.0.1:9999")!, session: session)
        let controller = FineTuneController()
        controller.attach(try JSONDecoder().decode(FineTuneJob.self, from: Data(snapshot.utf8)), client: client)
        let installed = snapshot.replacingOccurrences(of: "\"adapter_applied\":false", with: "\"adapter_applied\":true")
        var routes: [String] = []
        var failLoad = true
        FineTuneProtocol.respond = { request in
            let path = request.url!.path
            routes.append(path)
            XCTAssertEqual(request.httpMethod, "POST")
            if path.hasSuffix("adapter") { return (200, "application/json", installed) }
            let body = (try? JSONSerialization.jsonObject(with: request.httpBody ?? Data())) as? [String: Any]
            XCTAssertEqual(body?["model"] as? String, "m")
            if path == "/v1/load-model", failLoad {
                return (503, "application/json", "{\"error\":{\"message\":\"Model busy\"}}")
            }
            return (200, "application/json", "{}")
        }
        let first = await controller.apply()
        XCTAssertFalse(first)
        XCTAssertFalse(controller.saved)
        XCTAssertEqual(controller.state.job?.adapterApplied, true)
        XCTAssertNotNil(controller.state.error)
        XCTAssertEqual(routes, ["/v1/finetune/ft-1/adapter", "/v1/unload-model", "/v1/load-model"])
        routes.removeAll()
        failLoad = false
        let retry = await controller.apply()
        XCTAssertTrue(retry)
        XCTAssertTrue(controller.saved)
        XCTAssertNil(controller.state.error)
        XCTAssertEqual(routes, ["/v1/unload-model", "/v1/load-model"])
    }
}

private final class FineTuneProtocol: URLProtocol {
    static var respond: ((URLRequest) -> (Int, String, String))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        var captured = request
        if captured.httpBody == nil, let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while true {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                data.append(buffer, count: count)
            }
            captured.httpBody = data
        }
        let (code, type, body) = Self.respond!(captured)
        let response = HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: ["Content-Type": type])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
