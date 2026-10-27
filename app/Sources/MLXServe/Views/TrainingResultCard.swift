import SwiftUI

struct TrainingResultCard: View {
    let result: FineTuneJob.Result
    let installed: Bool
    let busy: Bool
    let apply: () -> Void
    let discard: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(result.verified ? "Improvement verified" : "Improvement not verified",
                  systemImage: result.verified ? "checkmark.circle" : "exclamationmark.circle")
                .font(.app(.headline)).foregroundStyle(result.verified ? Color.green : .orange)
            Text(result.recommendation).font(.app(.body))
            Text(result.prompt).font(.app(.headline)).textSelection(.enabled)
            HStack(alignment: .top, spacing: 16) {
                example("Before", result.before)
                example("After", result.after)
            }
            Text("Held-out refusals: \(result.refusedBefore) → \(result.refusedAfter) of \(result.heldoutPrompts)")
                .font(.app(.callout))
            Text("Side effects: mean KL \(result.meanKl, specifier: "%.3f"), peak \(result.maxKl, specifier: "%.3f"). Lower is better.")
                .font(.app(.caption)).foregroundStyle(.secondary)
            Divider()
            Text(installed ? "Customization saved. Reload the model to use it." : "Apply this customization to the model and reload it. It will affect every chat using this model.")
                .font(.app(.caption)).foregroundStyle(.secondary)
            HStack {
                Button("Discard", role: .destructive, action: discard).font(.app(.body))
                    .disabled(installed)
                Spacer()
                if busy { ProgressView().controlSize(.small) }
                Button(installed ? "Reload Model" : "Apply & Reload", action: apply).font(.app(.body)).buttonStyle(.borderedProminent)
            }.disabled(busy)
        }
    }

    private func example(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.app(.headline))
            Text(text.isEmpty ? "No example available" : text).font(.app(.body)).textSelection(.enabled)
        }.frame(maxWidth: .infinity, alignment: .leading)
            .padding(12).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }
}
