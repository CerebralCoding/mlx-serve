import SwiftUI
import AppKit

/// The app's type ladder: macOS's own text styles, never below `floor`.
///
/// macOS has no dynamic type (`.dynamicTypeSize` does not move a semantic font
/// here), so the numbers come from this table. They are the platform's, so the
/// app sits at the same size as Notes or Finder; only the 10pt steps are raised.
enum AppType {
    /// No text in the app renders smaller than this.
    static let floor: CGFloat = 11

    /// Every step, with the macOS size it was derived from. The `system` column
    /// is what the ladder is anchored to: `SystemTypeTests` fails when macOS
    /// moves one, which is the signal to re-derive this table rather than let
    /// the app drift away from the platform on its own.
    static let table: [(style: Font.TextStyle, system: CGFloat, pointSize: CGFloat)] = [
        (.largeTitle,  26, 26),
        (.title,       22, 22),
        (.title2,      17, 17),
        (.title3,      15, 15),
        (.headline,    13, 13),
        (.body,        13, 13),
        (.callout,     12, 12),
        (.subheadline, 11, 11),
        (.footnote,    10, 11),
        (.caption,     10, 11),
        (.caption2,    10, 11),
    ]

    /// The point size a step renders at.
    static func pointSize(for style: Font.TextStyle) -> CGFloat {
        table.first { $0.style == style }?.pointSize ?? floor
    }

    /// May the app render text at this size? Whole points, never under `floor`.
    static func isLegal(_ size: CGFloat) -> Bool {
        size >= floor && size == size.rounded()
    }

    /// What a piece of text IS, which is what picks its step. Without this the
    /// ladder has a floor and a rule and no way to say "an explainer is not a
    /// caption", and every view picks the smallest step it can — which is how
    /// the settings prose ended up at 10pt in the first place.
    ///
    /// The steps, read as a scale rather than as a menu: page > section > row >
    /// supporting > annotation. `Font.app(_:)` takes the step directly; this is
    /// the sentence that says which one a given piece of copy wants.
    enum Role {
        /// A window's own name. Once per window.
        case pageTitle
        /// A pane or sheet's heading.
        case sectionTitle
        /// The name of one setting, row, model or item.
        case rowTitle
        /// A sentence under a row that says what it does. The one that was too
        /// small: prose the user reads on purpose, not a label.
        case explainer
        /// A value, a status, a cost, a count.
        case value
        /// A badge, a unit, a qualifier next to something bigger.
        case annotation

        /// The step this role takes.
        var step: Font.TextStyle {
            switch self {
            case .pageTitle:   return .largeTitle
            // `.title3`, not `.headline`: seventeen sheet and section headings
            // already sit there, and a pane title cannot be smaller than them.
            case .sectionTitle: return .title3
            case .rowTitle:    return .body
            case .explainer:   return .callout
            case .value:       return .callout
            case .annotation:  return .footnote
            }
        }
    }
}

extension Font {
    /// The ladder, by role: the step says what the text IS, so two views that
    /// mean the same thing cannot pick two different sizes.
    static func app(_ role: AppType.Role, weight: Font.Weight? = nil, design: Font.Design? = nil) -> Font {
        .app(role.step, weight: weight, design: design)
    }
}

extension Font {
    /// The one way a view states a text size. `style` names the step and the
    /// number comes from `AppType`, so a size cannot be typed into a view and
    /// drift off the ladder — `SystemTypeTests` fails the build on a literal.
    ///
    /// The weight and design stay at the call site: they change how a step
    /// looks, never how big it is.
    static func app(
        _ style: Font.TextStyle,
        weight: Font.Weight? = nil,
        design: Font.Design? = nil
    ) -> Font {
        .system(size: AppType.pointSize(for: style), weight: weight, design: design)
    }
}

extension AppType {
    /// The AppKit font for a step, for the text SwiftUI does not draw: the log
    /// view, the status menu, a code block's text view. Same table, so the two
    /// halves of the app cannot drift apart.
    static func system(_ style: Font.TextStyle, weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.systemFont(ofSize: pointSize(for: style), weight: weight)
    }

    static func monospaced(_ style: Font.TextStyle, weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.monospacedSystemFont(ofSize: pointSize(for: style), weight: weight)
    }
}
