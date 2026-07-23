import SwiftUI

/// Shared plumbing for offering LAN-discovered models beside a media pane's
/// local presets in ONE Picker. Selection (and the persisted `modelId`) is a
/// string: a preset id for local models, `"lan:<model>@<peer>"` for network
/// ones — the suffix form the local server proxies to the hosting Mac.
enum LanPick {
    static let prefix = "lan:"

    /// The LAN routing id inside a selection/persisted value, or nil for locals.
    static func lanId(_ value: String) -> String? {
        value.hasPrefix(prefix) ? String(value.dropFirst(prefix.count)) : nil
    }

    /// The `modelId` to persist for the current pane state.
    static func persisted(lanModel: String?, presetId: String) -> String {
        lanModel.map { prefix + $0 } ?? presetId
    }

    /// The peer name inside a LAN routing id ("model@peer" → "peer").
    static func peer(of id: String) -> String {
        guard let at = id.lastIndex(of: "@") else { return id }
        return String(id[id.index(after: at)...])
    }

    /// One string binding driving both selections. Local picks flow into
    /// `model` (whose `.onChange` keeps applying preset defaults + persisting,
    /// exactly as before); LAN picks land in `lanModel` and persist directly —
    /// there is no preset change for `.onChange` to observe.
    static func selection<P: Identifiable>(
        model: Binding<P>, lanModel: Binding<String?>,
        resolve: @escaping (String) -> P?, persist: @escaping () -> Void
    ) -> Binding<String> where P.ID == String {
        Binding(
            get: { lanModel.wrappedValue.map { prefix + $0 } ?? model.wrappedValue.id },
            set: { picked in
                if let lan = lanId(picked) {
                    lanModel.wrappedValue = lan
                    persist()
                } else {
                    let hadLan = lanModel.wrappedValue != nil
                    lanModel.wrappedValue = nil
                    if let p = resolve(picked) { model.wrappedValue = p }
                    // Re-picking the SAME preset after a LAN model: `model`
                    // didn't change, so its `.onChange` won't persist — do it.
                    if hadLan { persist() }
                }
            }
        )
    }
}

/// "On Your Network" rows for a media pane's model Picker: one row per
/// LAN-discovered model advertising `capability` ("image", "audio", "music",
/// "video", "3d"). Renders nothing when the server is down, discovery is off,
/// or no peer shares a matching model.
struct LanModelPickerRows: View {
    @EnvironmentObject var server: ServerManager
    let capability: String

    var body: some View {
        let models = server.lanModels(capability: capability)
        if !models.isEmpty {
            Section("On Your Network") {
                ForEach(models, id: \.name) { m in
                    Text(m.lanDisplayName).tag(LanPick.prefix + m.name)
                }
            }
        }
    }
}
