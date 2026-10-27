import XCTest
@testable import MLXCore

final class AdapterModelsTests: XCTestCase {
    func testDenseTrainingEligibilityMatchesTheServer() {
        for arch in ["llama", "qwen3", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "qwen3_5", "qwen3_5_text"] {
            var model = LocalModel(id: "base", name: "base", path: "/models/base", sizeFormatted: "1 GB",
                modelType: arch, source: .mlxServe, kind: .base)
            model.quantBits = 4
            XCTAssertTrue(FineTunePresentation.supports(model), arch)
            model.quantBits = 8
            XCTAssertFalse(FineTunePresentation.supports(model), arch)
            model.quantBits = 16
            XCTAssertTrue(FineTunePresentation.supports(model), arch)
            model.numExperts = 128
            XCTAssertFalse(FineTunePresentation.supports(model), arch)
            model.numExperts = nil
            model.quantFile = "model.gguf"
            XCTAssertFalse(FineTunePresentation.supports(model), arch)
        }
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
    }

    private func job(status: String, id: String = "ft-1") throws -> FineTuneJob {
        try JSONDecoder().decode(FineTuneJob.self, from: Data("""
        {"id":"\(id)","model":"org/base","name":"roses","status":"\(status)",
         "stage":"train","step":2,"total":100,"loss":1.2,"eta_seconds":125,
         "cancel_requested":false,"preview":"A rose","result":null,"error_name":null,
         "adapter_discarded":false,"adapter_applied":false}
        """.utf8))
    }
}
