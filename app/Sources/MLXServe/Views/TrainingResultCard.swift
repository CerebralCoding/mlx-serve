import SwiftUI

struct TrainingResultCard: View {
    let result: FineTuneJob.Result
    @Binding var name: String
    @Binding var strength: Double
    let busy: Bool
    let save: () -> Void
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
            HStack {
                Text("Strength").font(.app(.body))
                Slider(value: $strength, in: 0...2, step: 0.05).accessibilityLabel("Adapter strength")
                Text(strength, format: .number.precision(.fractionLength(2))).font(.app(.body)).monospacedDigit()
            }
            Text("0 uses the original model. The comparison above was measured at strength 1.")
                .font(.app(.caption)).foregroundStyle(.secondary)
            TextField("Save as", text: $name).font(.app(.body)).textFieldStyle(.roundedBorder)
            Text("Use letters, numbers, dots, hyphens or underscores; do not start with a dot.")
                .font(.app(.caption)).foregroundStyle(.secondary)
            HStack {
                Button("Discard", role: .destructive, action: discard).font(.app(.body))
                Spacer()
                if busy { ProgressView().controlSize(.small) }
                Button("Save as", action: save).font(.app(.body)).buttonStyle(.borderedProminent)
                    .disabled(!FineTunePresentation.validName(name))
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
