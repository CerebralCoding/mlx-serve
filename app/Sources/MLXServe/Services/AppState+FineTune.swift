import Foundation

extension AppState {
    func customizeModel(path: String, title: String, prompt: String = "", chatId: UUID? = nil) {
        if let existing = fineTune.request,
           fineTune.state.busy || (fineTune.state.job?.status == .completed
                && fineTune.state.job?.adapterDiscarded == false && !fineTune.saved) {
            customizeRequest = existing
        } else {
            fineTune.state = FineTuneState()
            let request = CustomizeRequest(path: path, title: title, chatId: chatId, prompt: prompt)
            fineTune.request = request
            customizeRequest = request
        }
    }

    func startFineTune(_ request: CustomizeRequest, topic: String, keepEverythingElse: Bool) {
        guard !fineTune.state.busy else { return }
        fineTune.state.begin()
        Task {
            guard await useModelAndAwaitReady(atPath: request.path) else {
                fineTune.state.startFailed("The model could not be loaded. Check the server status and try again.")
                return
            }
            let api = APIClient()
            api.host = serverOptions.host
            let client = FineTuneClient(baseURL: api.serverURL(port: server.port, path: "/"))
            do {
                let job = try await client.create(modelPath: request.path,
                    name: "custom-\(UUID().uuidString.lowercased())", topic: topic,
                    keepEverythingElse: keepEverythingElse)
                fineTune.attach(job, client: client)
            } catch { fineTune.state.startFailed(error.localizedDescription) }
        }
    }

    func setChatAdapter(_ adapter: ChatAdapterSelection?, chatId: UUID?) {
        guard let id = chatId, let index = chatSessions.firstIndex(where: { $0.id == id }) else { return }
        chatSessions[index].adapter = adapter
        saveChatHistory()
    }

    var activeChatAdapter: ChatAdapterSelection? {
        chatSessions.first { $0.id == activeChatId }?.adapter?
            .forModel(selectedModelPath, local: !useAppleModel && server.lanChatModelId == nil)
    }
}
