import Foundation

/// Multi-session tab model for the Sandbox window's Terminal tab. Pure —
/// pinned by SandboxSessionTabsTests.
///
/// Several agent sessions run concurrently: each is one more ssh connection
/// into the SAME guest sshd (dropbear multiplexes; one mirror port serves
/// them all). The model owns ordering, selection, stable display names, and
/// the window title. What it deliberately does NOT own: view lifetime — the
/// view keeps every live terminal MOUNTED (ZStack + opacity) regardless of
/// selection or the Terminal/Activity switch, because unmounting an
/// EmbeddedTerminalView terminates its ssh (the live bug this replaced).
struct SandboxSessionTabs: Equatable {

    struct Tab: Identifiable, Equatable {
        enum Phase: Equatable {
            case preparing
            case live
            case exited(Int32?)
        }
        let id: UUID
        let label: String        // registry label: "pi" / "hermes" / "shell"
        let displayName: String  // "pi", "pi 2" — assigned at creation, never renumbered
        var phase: Phase
    }

    private(set) var tabs: [Tab] = []
    private(set) var selectedID: UUID?
    /// Per-label session ordinals. Monotonic — a closed "pi" never frees its
    /// number, so "pi 2" can't silently become "pi" mid-session.
    private var ordinals: [String: Int] = [:]

    var selected: Tab? { tabs.first { $0.id == selectedID } }

    /// "pi 2 — MLX Sandbox" while the selected session is starting or live;
    /// the base title otherwise (a dead session no longer owns the window).
    var windowTitle: String {
        guard let tab = selected else { return "MLX Sandbox" }
        switch tab.phase {
        case .preparing, .live: return "\(tab.displayName) — MLX Sandbox"
        case .exited: return "MLX Sandbox"
        }
    }

    @discardableResult
    mutating func addPreparing(label: String) -> UUID {
        let n = (ordinals[label] ?? 0) + 1
        ordinals[label] = n
        let tab = Tab(id: UUID(), label: label,
                      displayName: n == 1 ? label : "\(label) \(n)",
                      phase: .preparing)
        tabs.append(tab)
        selectedID = tab.id
        return tab.id
    }

    mutating func markLive(_ id: UUID) {
        guard let i = tabs.firstIndex(where: { $0.id == id }) else { return }
        tabs[i].phase = .live
    }

    mutating func markExited(_ id: UUID, exitCode: Int32?) {
        guard let i = tabs.firstIndex(where: { $0.id == id }) else { return }
        tabs[i].phase = .exited(exitCode)
    }

    mutating func select(_ id: UUID) {
        guard tabs.contains(where: { $0.id == id }) else { return }
        selectedID = id
    }

    /// Remove a tab; a selected close moves selection to the nearest
    /// surviving neighbor (previous index, clamped) — never "nothing
    /// selected" while tabs remain.
    mutating func close(_ id: UUID) {
        guard let i = tabs.firstIndex(where: { $0.id == id }) else { return }
        tabs.remove(at: i)
        if selectedID == id {
            selectedID = tabs.isEmpty ? nil : tabs[min(i, tabs.count - 1)].id
        }
    }

    func displayName(_ id: UUID) -> String {
        tabs.first { $0.id == id }?.displayName ?? ""
    }

    /// The newest preparing/live tab for an agent label — the focus target
    /// for the tray's "<agent> in Sandbox" shortcut. Exited tabs never match
    /// (focusing a corpse would read as "the shortcut is broken").
    func mostRecentActive(label: String) -> UUID? {
        tabs.last { tab in
            guard tab.label == label else { return false }
            switch tab.phase {
            case .preparing, .live: return true
            case .exited: return false
            }
        }?.id
    }

    /// True when closing this tab would kill something — a preparing or live
    /// session. Exited/unknown tabs close without a confirm (nothing to lose).
    func closeNeedsConfirmation(_ id: UUID) -> Bool {
        guard let tab = tabs.first(where: { $0.id == id }) else { return false }
        switch tab.phase {
        case .preparing, .live: return true
        case .exited: return false
        }
    }

    /// Honest per-tab exit notice; nil unless the tab actually exited.
    func exitNotice(_ id: UUID) -> String? {
        guard let tab = tabs.first(where: { $0.id == id }),
              case .exited(let code) = tab.phase else { return nil }
        if let code, code != 0 { return "\(tab.displayName) session ended (exit \(code))" }
        return "\(tab.displayName) session ended"
    }
}
