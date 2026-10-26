import SwiftUI

struct TrainingProgressCard: View {
    let state: FineTuneState
    let stop: () -> Void
    let retry: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                ForEach([FineTuneJob.Stage.synth, .train, .verify], id: \.rawValue) { stage in
                    Text(stage.title).font(.app(.caption, weight: .semibold))
                        .foregroundStyle(state.job?.stage == stage ? Color.accentColor : .secondary)
                    if stage != .verify { Image(systemName: "chevron.right").font(.app(.caption2)).foregroundStyle(.secondary) }
                }
                Spacer()
            }
            if let job = state.job {
                ProgressView(value: Double(min(job.step, job.total)), total: Double(max(1, job.total)))
                HStack {
                    Text("\(job.step) / \(job.total)").font(.app(.body)).monospacedDigit()
                    Spacer()
                    Text("Stage remaining: \(FineTunePresentation.eta(job.etaSeconds))")
                        .font(.app(.callout)).monospacedDigit().foregroundStyle(.secondary)
                }
                if let loss = job.loss {
                    Text("Training loss: \(loss, specifier: "%.3f")").font(.app(.caption)).foregroundStyle(.secondary)
                }
                if !job.preview.isEmpty {
                    Text("Live preview").font(.app(.headline))
                    Text(job.preview).font(.app(.body)).textSelection(.enabled)
                }
            } else {
                ProgressView().controlSize(.small)
                Text("Loading the model and checking available memory…").font(.app(.body))
            }
            HStack {
                if state.error != nil { Button("Reconnect", action: retry).font(.app(.body)) }
                Spacer()
                Button(state.stopping || state.job?.cancelRequested == true ? "Stopping…" : "Stop", action: stop)
                    .font(.app(.body)).disabled(state.job == nil || state.stopping || state.job?.cancelRequested == true)
            }
        }
    }
}
