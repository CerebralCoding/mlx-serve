import XCTest
@testable import MLXCore

final class AdapterModelsTests: XCTestCase {
    func testSelectionIsBoundToLocalBaseAndPreservesZeroStrength() throws {
        let pick = ChatAdapterSelection(model: "org/base", name: "roses", scale: 0)
        XCTAssertEqual(pick.forModel("org/base", local: true), pick)
        XCTAssertNil(pick.forModel("other", local: true))
        XCTAssertNil(pick.forModel("org/base", local: false))
        var defaults = APIClient.RequestDefaults()
        defaults.adapter = pick
        for tools in [nil, "[]"] as [String?] {
            let data = try APIClient.chatRequestBody(messages: [], maxTokens: 32, temperature: 0,
                enableThinking: false, toolsJSON: tools, defaults: defaults, modelId: pick.model)
            let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(body["lora"] as? String, "roses")
            XCTAssertEqual(body["lora_scale"] as? Double, 0)
        }
    }

    func testAdaptersAreDiscoveredOnModelRows() {
        let info = APIClient.parseModelInfo(["id": "org/base", "adapters": ["roses", "concise"]])
        XCTAssertEqual(info.adapters, ["roses", "concise"])
    }

    func testInvalidAdapterMetadataDoesNotHideChats() throws {
        let store = try ChatStore(path: ":memory:", legacyHistoryPath: nil, legacySidebar: nil)
        let chat = ChatSession(title: "Preserve me")
        store.save([chat])
        try store.database.run("INSERT INTO chat_adapters VALUES (?, ?)", [.text(chat.id.uuidString), .text("invalid JSON")])
        XCTAssertEqual(store.load().map(\.id), [chat.id])
        XCTAssertNil(store.load().first?.adapter)
    }

    func testTrainingStateDoesNotFinishOnStopOrAcceptStaleUpdates() throws {
        var state = FineTuneState()
        state.begin()
        XCTAssertTrue(state.busy)
        state.receive(try job(status: "running"))
        state.requestStop()
        XCTAssertTrue(state.busy)
        XCTAssertTrue(state.stopping)
        state.receive(try job(status: "running", id: "ft-2"))
        XCTAssertEqual(state.job?.id, "ft-1")
        state.receive(try job(status: "cancelled"))
        XCTAssertFalse(state.busy)
        state.receive(try job(status: "running"))
        XCTAssertEqual(state.job?.status, .cancelled)
    }

    func testMissingServerJobReleasesTheTrainingUI() throws {
        var state = FineTuneState()
        state.begin()
        state.receive(try job(status: "running"))
        state.jobMissing()
        XCTAssertFalse(state.busy)
        XCTAssertNil(state.job)
        XCTAssertNotNil(state.error)
    }

    func testStageETAAndRefusalEntry() {
        XCTAssertEqual(FineTunePresentation.eta(nil), "Estimating…")
        XCTAssertEqual(FineTunePresentation.eta(125), "2:05")
        XCTAssertEqual(FineTunePresentation.eta(3600), "60:00")
        XCTAssertTrue(FineTunePresentation.isRefusal("I'm sorry, but I can't help with that."))
        XCTAssertFalse(FineTunePresentation.isRefusal("You can prune roses in spring."))
        XCTAssertFalse(FineTunePresentation.validName("../escape"))
        XCTAssertTrue(FineTunePresentation.validName("my-roses_2"))
    }

    func testAdapterPersistsPerChatAndCanBeCleared() throws {
        let store = try ChatStore(path: ":memory:", legacyHistoryPath: nil, legacySidebar: nil)
        var first = ChatSession(title: "First")
        let second = ChatSession(title: "Second")
        first.adapter = ChatAdapterSelection(model: "org/base", name: "roses", scale: 0.75)
        store.save([first, second])
        XCTAssertEqual(store.load().first { $0.id == first.id }?.adapter, first.adapter)
        XCTAssertNil(store.load().first { $0.id == second.id }?.adapter)
        first.adapter = nil
        store.save([first, second])
        XCTAssertNil(store.load().first { $0.id == first.id }?.adapter)
    }

    private func job(status: String, id: String = "ft-1") throws -> FineTuneJob {
        try JSONDecoder().decode(FineTuneJob.self, from: Data("""
        {"id":"\(id)","model":"org/base","name":"roses","status":"\(status)",
         "stage":"train","step":2,"total":100,"loss":1.2,"eta_seconds":125,
         "cancel_requested":false,"preview":"A rose","result":null,"error_name":null,
         "adapter_discarded":false}
        """.utf8))
    }
}
