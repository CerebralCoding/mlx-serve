import SwiftUI
import SwiftTerm

/// Gutter balancing (pure — EmbeddedTerminalLayoutTests). SwiftTerm draws its
/// columns from x=0 but stops them short of the right edge by the scroller
/// reservation (its overlay NSScroller is never hidden), so an un-inset
/// terminal reads as "more margin on the right". Mirroring the reservation on
/// the left evens the gutters; the residual difference is the ≤1-cell column
/// quantization every terminal app has.
enum EmbeddedTerminalLayout {
    static func terminalFrame(in bounds: CGRect, scrollerReservation: CGFloat) -> CGRect {
        CGRect(x: bounds.minX + scrollerReservation,
               y: bounds.minY,
               width: max(0, bounds.width - scrollerReservation),
               height: max(0, bounds.height))
    }
}

/// A real terminal emulator embedded in SwiftUI, spawning one command on a
/// PTY and reporting its exit.
///
/// THE SwiftTerm SEAM: this is the only file in the app that imports
/// SwiftTerm. Everything else deals in argv + an exit callback, so a
/// libghostty-backed implementation can replace this single view later.
///
/// The view is created once per session (`.id(sessionUUID)` at the call
/// site) — `updateNSView` never restarts the process.
struct EmbeddedTerminalView: NSViewRepresentable {
    /// Absolute executable path (the sandbox sessions spawn `/usr/bin/ssh`).
    let executable: String
    let args: [String]
    /// Chrome-side handle (End Session button) — SwiftTerm never leaks past
    /// this file, so the button gets a terminate() through this instead.
    var controller: EmbeddedTerminalController? = nil
    /// Called on the main thread when the child exits. nil exit code = the
    /// PTY/IO layer died (e.g. the guest was stopped underneath the session).
    let onExit: (Int32?) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onExit: onExit) }

    func makeNSView(context: Context) -> PaddedTerminalContainer {
        let terminal = LocalProcessTerminalView(frame: .zero)
        terminal.processDelegate = context.coordinator
        context.coordinator.view = terminal
        controller?.coordinator = context.coordinator
        // Default environment (TERM=xterm-256color etc.) — ssh needs nothing
        // from the host env; every path it uses arrives via argv.
        terminal.startProcess(executable: executable, args: args)
        return PaddedTerminalContainer(terminal: terminal)
    }

    func updateNSView(_ view: PaddedTerminalContainer, context: Context) {
        context.coordinator.onExit = onExit
    }

    static func dismantleNSView(_ view: PaddedTerminalContainer, coordinator: Coordinator) {
        // The window can close (or the tab re-render) while ssh is live —
        // never leave an orphaned child on the host.
        view.terminal.terminate()
    }

    /// Hosts the terminal with a left inset mirroring SwiftTerm's right-side
    /// scroller reservation (see EmbeddedTerminalLayout), painted in the
    /// terminal's own background color so the strip reads as margin, not seam.
    final class PaddedTerminalContainer: NSView {
        let terminal: LocalProcessTerminalView

        init(terminal: LocalProcessTerminalView) {
            self.terminal = terminal
            super.init(frame: .zero)
            wantsLayer = true
            addSubview(terminal)
        }

        required init?(coder: NSCoder) { nil }

        override func layout() {
            super.layout()
            // The same width SwiftTerm reserves for its scroller strip
            // (scrollerStyle is public; the reservation itself is not).
            let reservation = NSScroller.scrollerWidth(for: .regular,
                                                       scrollerStyle: terminal.scrollerStyle)
            terminal.frame = EmbeddedTerminalLayout.terminalFrame(in: bounds,
                                                                  scrollerReservation: reservation)
            layer?.backgroundColor = terminal.nativeBackgroundColor.cgColor
        }
    }

    /// Owned by the SwiftUI chrome; bridges its End Session button to the
    /// live coordinator without exposing SwiftTerm types.
    final class EmbeddedTerminalController {
        fileprivate weak var coordinator: Coordinator?
        func terminate() { coordinator?.terminate() }
    }

    final class Coordinator: NSObject, LocalProcessTerminalViewDelegate {
        var onExit: (Int32?) -> Void
        weak var view: LocalProcessTerminalView?
        private var exited = false

        init(onExit: @escaping (Int32?) -> Void) { self.onExit = onExit }

        /// End the session from UI chrome (the End Session button): SIGTERM
        /// to the spawned ssh, which drops the PTY and fires processTerminated.
        func terminate() { view?.terminate() }

        func sizeChanged(source: LocalProcessTerminalView, newCols: Int, newRows: Int) {}
        func setTerminalTitle(source: LocalProcessTerminalView, title: String) {}
        func hostCurrentDirectoryUpdate(source: TerminalView, directory: String?) {}
        func processTerminated(source: TerminalView, exitCode: Int32?) {
            // Dedup: terminate() + the IO teardown can both land here.
            guard !exited else { return }
            exited = true
            let cb = onExit
            DispatchQueue.main.async { cb(exitCode) }
        }
    }
}
