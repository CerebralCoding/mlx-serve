import SwiftUI

struct CustomizeSheet: View {
    let request: CustomizeRequest
    @ObservedObject var controller: FineTuneController
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var topic: String
    @State private var keepEverythingElse = true
    @State private var name = "my-custom-model"
    @State private var strength = 1.0

    init(request: CustomizeRequest, controller: FineTuneController) {
        self.request = request
        self.controller = controller
        _topic = State(initialValue: request.prompt)
    }

    private var unsupported: Bool {
        guard let model = appState.localModels.first(where: { $0.path == request.path }) else { return false }
        return !FineTunePresentation.supports(model)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Customize").font(.app(.title2, weight: .semibold))
                Text(request.title).font(.app(.body)).foregroundStyle(.secondary)
            }
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if controller.state.busy {
                        TrainingProgressCard(state: controller.state,
                            stop: { Task { await controller.stop() } }, retry: controller.watch)
                    } else if let job = controller.state.job, let result = job.result, !job.adapterDiscarded {
                        TrainingResultCard(result: result, name: $name, strength: $strength,
                            busy: controller.managing, save: save, discard: discard)
                    } else {
                        if controller.state.job?.status == .cancelled {
                            Text("Training stopped. You can try again.").font(.app(.body))
                        } else if controller.state.job?.adapterDiscarded == true {
                            Text("Adapter discarded.").font(.app(.body))
                        }
                        input
                    }
                    if let error = controller.state.error {
                        Text(error).font(.app(.callout)).foregroundStyle(.red).textSelection(.enabled)
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            HStack {
                if controller.state.busy {
                    Text("Training continues while this sheet is hidden.")
                        .font(.app(.caption)).foregroundStyle(.secondary)
                }
                Spacer()
                Button(controller.state.busy ? "Hide" : "Close") { dismiss() }
                    .font(.app(.body)).keyboardShortcut(.cancelAction)
            }
        }
        .padding(24)
        .frame(width: 660, height: 610)
        .interactiveDismissDisabled(controller.managing)
    }

    private var input: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What should it learn to answer?").font(.app(.headline))
            Text("Describe a topic or paste a few example prompts. Your Mac will create training examples and test the result.")
                .font(.app(.body)).foregroundStyle(.secondary)
            TextEditor(text: $topic).font(.app(.body)).frame(height: 140)
                .padding(8).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
                .accessibilityLabel("Topic or example prompts")
            Toggle("Keep everything else", isOn: $keepEverythingElse).font(.app(.body))
            Text("Include general answers during training to reduce unintended changes. Side effects are checked either way.")
                .font(.app(.caption)).foregroundStyle(.secondary)
            if unsupported {
                Text("Customization supports dense Gemma 4, Qwen3, Qwen3.5/3.8 and Llama MLX models in 4-bit or bf16. This model is not supported yet.")
                    .font(.app(.callout)).foregroundStyle(.secondary)
            }
            HStack {
                Text("Time estimate appears once training starts.").font(.app(.caption)).foregroundStyle(.secondary)
                Spacer()
                Button("Train") { appState.startFineTune(request, topic: topic, keepEverythingElse: keepEverythingElse) }
                    .font(.app(.body)).buttonStyle(.borderedProminent)
                    .disabled(unsupported || topic.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || topic.utf8.count > 2048)
            }
            if topic.utf8.count > 2048 {
                Text("Shorten your description to 2,048 UTF-8 bytes or fewer.").font(.app(.caption)).foregroundStyle(.red)
            }
        }
    }

    private func save() {
        Task {
            guard await controller.save(name: name), let job = controller.state.job else { return }
            if let chatId = request.chatId {
                appState.applyChatModelPick(request.path)
                appState.setChatAdapter(.init(model: request.path, name: job.name, scale: strength), chatId: chatId)
            }
            await appState.server.refreshModels()
            dismiss()
        }
    }

    private func discard() {
        Task {
            guard let name = controller.state.job?.name, await controller.discard() else { return }
            for chat in appState.chatSessions where chat.adapter?.name == name {
                appState.setChatAdapter(nil, chatId: chat.id)
            }
            await appState.server.refreshModels()
            dismiss()
        }
    }
}
