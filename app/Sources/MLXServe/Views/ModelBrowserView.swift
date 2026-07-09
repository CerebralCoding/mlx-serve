import SwiftUI

/// Model Browser: a sidebar over four destinations (`ModelBrowserSection`).
///
/// It used to be one pane with a `Toggle("Downloaded")` push-button that swapped
/// the data source underneath you, and it deleted on-disk models from the search
/// results — so the model you just finished downloading vanished at 100%. Now
/// Discover marks what you own instead of hiding it, Downloads is a first-class
/// queue rather than rows stapled to the top of a list, and My Models shows
/// everything the tray picker offers rather than only what we fetched ourselves.
struct ModelBrowserView: View {
    @EnvironmentObject var searchService: HFSearchService
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    @State private var selection: ModelBrowserSection? = .discover
    @State private var localFilter = ""

    private var section: ModelBrowserSection { selection ?? .discover }

    /// Downloading *or* failed — both belong in the queue and both earn a badge.
    private var activeDownloads: [(repoId: String, state: DownloadManager.DownloadState)] {
        downloads.downloads
            .filter { $0.value.status == .downloading || $0.value.status == .failed }
            .sorted { $0.key < $1.key }
            .map { (repoId: $0.key, state: $0.value) }
    }

    private var badges: ModelBrowserBadgeCounts {
        ModelBrowserBadgeCounts(
            myModels: appState.localModels.count,
            activeDownloads: activeDownloads.count,
            draftersReady: GemmaVariant.allCases.filter { downloads.isReady($0.drafterRepoId) }.count
        )
    }

    var body: some View {
        NavigationSplitView {
            List(ModelBrowserSection.allCases, selection: $selection) { item in
                NavigationLink(value: item) {
                    Label {
                        HStack(spacing: 6) {
                            Text(item.title)
                            Spacer(minLength: 4)
                            if item == .downloads, !activeDownloads.isEmpty {
                                ProgressView()
                                    .controlSize(.small)
                                    .scaleEffect(0.6)
                            }
                            if let badge = badges.badge(for: item) {
                                Text(badge)
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 1)
                                    .background(.quaternary, in: Capsule())
                            }
                        }
                    } icon: {
                        Image(systemName: item.systemImage)
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 190, ideal: 205, max: 260)
        } detail: {
            detail
                .frame(minWidth: 720)
        }
        .task {
            if searchService.models.isEmpty {
                await searchService.search()
            }
        }
        .onChange(of: selection) { _, _ in appState.refreshModels() }
        // Live-refresh on-disk sizes while a disk-state pane is showing and a
        // download is in flight, so completion + growing size show up without
        // the user navigating away and back. The task id flips when the section
        // changes or the active-download set changes, which cancels +
        // re-evaluates the guard — so it self-terminates once everything
        // finishes.
        .task(id: "\(section.rawValue)-\(activeDownloads.count)") {
            guard ModelBrowserSection.shouldLivePoll(section: section,
                                                     hasActiveDownloads: !activeDownloads.isEmpty) else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if Task.isCancelled { break }
                appState.refreshModels()
            }
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch section {
        case .discover:  DiscoverPane()
        case .myModels:  MyModelsPane(filter: $localFilter)
        case .downloads: DownloadsPane(items: activeDownloads)
        case .drafters:  DraftersPane()
        }
    }
}

// MARK: - Discover

/// HuggingFace search. On-disk models stay in the list, marked `✓ On disk` with
/// a Use action — never filtered out.
private struct DiscoverPane: View {
    @EnvironmentObject var searchService: HFSearchService
    @EnvironmentObject var downloads: DownloadManager

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search models...", text: $searchService.searchQuery)
                        .textFieldStyle(.plain)
                        .onSubmit { Task { await searchService.search() } }
                }
                .padding(8)
                .background(.quaternary.opacity(0.5))
                .cornerRadius(8)

                // Weight-format filter: MLX (safetensors), GGUF (llama.cpp /
                // ds4), or Both. Re-runs the search on change.
                Picker("Format", selection: $searchService.format) {
                    ForEach(ModelFormat.allCases) { f in
                        Text(f.label).tag(f)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .onChange(of: searchService.format) { _, _ in
                    Task { await searchService.search() }
                }

                Button("Search") {
                    Task { await searchService.search() }
                }
                .controlSize(.regular)
            }
            .padding(12)

            Divider()

            ColumnHeaderRow(searchService: searchService)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.quaternary.opacity(0.3))

            Divider()

            let onDiskCount = searchService.models.filter { downloads.isReady($0.id) }.count

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(searchService.models) { model in
                        ModelBrowserRow(
                            model: model,
                            fitness: searchService.ramFitness(for: model)
                        )
                        Divider().padding(.horizontal, 12)
                    }

                    if searchService.isLoading {
                        ProgressView()
                            .padding(20)
                    } else if let error = searchService.error {
                        Text(error)
                            .foregroundStyle(.red)
                            .font(.caption)
                            .padding(20)
                    } else if searchService.models.isEmpty {
                        Text("No models found")
                            .foregroundStyle(.secondary)
                            .padding(40)
                    }

                    if searchService.hasMore && !searchService.models.isEmpty && !searchService.isLoading {
                        Button("Load More") {
                            Task { await searchService.loadMore() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                        .padding(16)
                    }
                }
            }

            Divider()

            HStack {
                Text("Showing \(searchService.models.count) models")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if onDiskCount > 0 {
                    Text("· \(onDiskCount) on disk")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
                Spacer()
                Text("System RAM: \(MemoryInfo.format(Int64(searchService.systemRAM)))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .navigationTitle("Discover")
    }
}

// MARK: - My Models

/// Everything on this Mac the tray picker can offer, grouped by where it came
/// from. The old "Downloaded" tab listed only `source == .mlxServe`, so it was a
/// strict subset of what you could actually load.
private struct MyModelsPane: View {
    @Binding var filter: String
    @EnvironmentObject var appState: AppState

    private var groups: [LocalModelGroup] {
        ModelBrowserUse.groupedBySource(appState.localModels, filter: filter)
    }

    private var total: Int { groups.reduce(0) { $0 + $1.models.count } }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Filter your models...", text: $filter)
                    .textFieldStyle(.plain)
                if !filter.isEmpty {
                    Button { filter = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(8)
            .background(.quaternary.opacity(0.5))
            .cornerRadius(8)
            .padding(12)

            Divider()

            HStack(spacing: 8) {
                Text("Model")
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text("Size on Disk")
                    .frame(width: 90, alignment: .trailing)
                Text("")
                    .frame(width: 120)
            }
            .font(.callout.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.3))

            Divider()

            ScrollView {
                LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                    ForEach(groups) { group in
                        Section {
                            ForEach(group.models) { model in
                                LocalModelRow(model: model)
                                Divider().padding(.horizontal, 12)
                            }
                        } header: {
                            Text(ModelBrowserUse.groupTitle(group.source))
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 4)
                                .background(.bar)
                        }
                    }

                    if groups.isEmpty {
                        Text(filter.isEmpty ? "No models on this Mac yet" : "No models match “\(filter)”")
                            .foregroundStyle(.secondary)
                            .padding(40)
                    }
                }
            }

            Divider()

            HStack {
                Text("\(total) model\(total == 1 ? "" : "s") on disk")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .navigationTitle("My Models")
    }
}

// MARK: - Downloads

/// The transfer queue. Promoted out of the old Downloaded tab so an in-flight
/// download is reachable (and badged) from anywhere in the browser.
private struct DownloadsPane: View {
    let items: [(repoId: String, state: DownloadManager.DownloadState)]

    var body: some View {
        VStack(spacing: 0) {
            if items.isEmpty {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No downloads in progress")
                        .foregroundStyle(.secondary)
                    Text("Start one from Discover or Drafters.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(items, id: \.repoId) { item in
                            ActiveDownloadRow(repoId: item.repoId, state: item.state)
                            Divider().padding(.horizontal, 12)
                        }
                    }
                }
            }
        }
        .navigationTitle("Downloads")
    }
}

// MARK: - Drafters pane

/// The curated drafter catalog, previously a collapsed disclosure pinned above
/// the search results. As its own destination it's discoverable without
/// competing for vertical space with the model list.
private struct DraftersPane: View {
    @EnvironmentObject var downloads: DownloadManager

    private var rows: [DrafterCatalogRow] {
        GemmaVariant.allCases.map { v in
            DrafterCatalogRow(
                variant: v,
                repoId: v.drafterRepoId,
                pairsWith: "for \(v.label)",
                sizeEstimate: DrafterCatalogRow.sizeEstimate(for: v)
            )
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .foregroundStyle(.purple)
                Text("Pair a drafter with a Gemma 4 base model for +27–40% on code & agents.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(12)
            .background(Color.purple.opacity(0.06))

            Divider()

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(rows) { row in
                        DrafterCatalogRowView(row: row)
                        Divider().padding(.horizontal, 12)
                    }
                }
            }
        }
        .navigationTitle("Drafters")
    }
}

// MARK: - In-use badge

/// Replaces the "Use" button on the model the server is pointed at, so clicking
/// Use produces immediate, visible feedback instead of just greying the button
/// out. Distinguishes "loaded and serving" from "selected, still loading" and
/// "selected, server stopped" — see `ModelUseState`.
private struct ModelUseBadge: View {
    let state: ModelUseState

    private var tint: Color {
        switch state {
        case .inUse:    return .green
        case .loading:  return .orange
        case .selected: return .secondary
        case .idle:     return .clear
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            switch state {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.5)
                    .frame(width: 8, height: 8)
            case .inUse:
                Circle()
                    .fill(tint)
                    .frame(width: 6, height: 6)
            default:
                EmptyView()
            }
            Text(state.label)
                .font(.caption.weight(.medium))
                .foregroundStyle(tint)
                .lineLimit(1)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(tint.opacity(0.12), in: Capsule())
        .help(state.help)
    }
}

// MARK: - Column Headers

private struct ColumnHeaderRow: View {
    @ObservedObject var searchService: HFSearchService

    var body: some View {
        HStack(spacing: 8) {
            Text("Model")
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("Cap.")
                .frame(width: 44, alignment: .center)
            SortableHeader("Quant", field: nil, searchService: searchService)
                .frame(width: 54, alignment: .leading)
            SortableHeader("Size", field: nil, searchService: searchService)
                .frame(width: 54, alignment: .trailing)
            // HuggingFace pull count. Called "Pulls", NOT "Downloads": the
            // sidebar has a Downloads destination meaning "transferring right
            // now", and having both words in one window is what made users read
            // the old "Downloaded" toggle as a filter on this column. 64 wide
            // fits "Pulls" + the sort chevron.
            SortableHeader("Pulls", field: .downloads, searchService: searchService)
                .frame(width: 64, alignment: .trailing)
                .help("How many times this repo has been pulled from HuggingFace")
            SortableHeader("Likes", field: .likes, searchService: searchService)
                .frame(width: 50, alignment: .trailing)
            // RAM Est. column: 120 wide so GGUF range strings produced by
            // `MemoryInfo.formatRange` ("11.1–30.9 GB", "21.2–55.4 GB",
            // up to "999.9–999.9 GB") render on a single line. Single-value
            // strings ("767 MB", "10.2 GB") were comfortable at 80; the
            // wider budget is what GGUF's min–max range needs. Keep the
            // ModelBrowserRow's RAM cell at the same width or alignment
            // drifts across rows.
            SortableHeader("RAM Est.", field: .estimatedSize, searchService: searchService)
                .frame(width: 120, alignment: .trailing)
            SortableHeader("Updated", field: .lastModified, searchService: searchService)
                .frame(width: 64, alignment: .trailing)
            // Action column: 120 wide (was 92) — the widest content is now the
            // on-disk cell, "✓ Use" plus a trash icon. Also fits the
            // "Download ▾" GGUF menu, Resume/Retry, and the progress bar.
            Text("")
                .frame(width: 120)
        }
        .font(.callout.weight(.semibold))
        .foregroundStyle(.secondary)
    }
}

private struct SortableHeader: View {
    let title: String
    let field: HFSortField?
    @ObservedObject var searchService: HFSearchService

    init(_ title: String, field: HFSortField?, searchService: HFSearchService) {
        self.title = title
        self.field = field
        self.searchService = searchService
    }

    private var isActive: Bool {
        guard let field else { return false }
        return searchService.sortField == field
    }

    var body: some View {
        if let field {
            Button {
                searchService.sort(by: field)
            } label: {
                HStack(spacing: 2) {
                    Text(title)
                    if isActive {
                        Image(systemName: searchService.sortDescending ? "chevron.down" : "chevron.up")
                            .font(.system(size: 8))
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(isActive ? .primary : .secondary)
        } else {
            Text(title)
        }
    }
}

// MARK: - Model Row

private struct ModelBrowserRow: View {
    let model: HFModel
    let fitness: RAMFitness
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var isReady: Bool { downloads.isReady(model.id) }
    private var state: DownloadManager.DownloadState? { downloads.downloads[model.id] }
    private var disabled: Bool { !model.isCompatible }

    /// For Gemma 4 dense/MoE base rows, the variant whose drafter pairs with
    /// this checkpoint — drives the inline "+drafter" / "✓ paired" chip. nil
    /// for non-Gemma-4 rows (most everything) and for GGUF repos (the
    /// drafter is an MLX-only kernel). The rule lives in
    /// `DownloadManager.drafterPairingVariant` so it's unit-testable.
    private var pairableVariant: GemmaVariant? {
        DownloadManager.drafterPairingVariant(
            repoId: model.id,
            isDrafter: model.isDrafter,
            isGgufRepo: model.isGgufRepo
        )
    }

    var body: some View {
        HStack(spacing: 8) {
            // Model name — takes all remaining space
            VStack(alignment: .leading, spacing: 1) {
                Text(model.modelName)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    Text(model.author)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                    if let reason = model.incompatibleReason {
                        Text(reason)
                            .font(.system(size: 10))
                            .foregroundStyle(.red.opacity(0.8))
                            .lineLimit(1)
                    }
                    if let v = pairableVariant {
                        DrafterPairChip(variant: v)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Capabilities icons
            HStack(spacing: 3) {
                if model.hasVision {
                    Image(systemName: "eye")
                        .help("Vision (image input)")
                }
                if model.hasToolCalling {
                    Image(systemName: "wrench")
                        .help("Tool calling")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(width: 44, alignment: .center)

            // Quantization badge
            Group {
                if let quant = model.quantization {
                    Text(quant)
                        .font(.system(size: 10).weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.quaternary)
                        .cornerRadius(4)
                } else {
                    Text("\u{2014}")
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(width: 54, alignment: .leading)

            // Size (parsed from model name)
            Text(model.modelSize)
                .font(.callout.monospacedDigit())
                .frame(width: 54, alignment: .trailing)

            // HF pull count — width matched to ColumnHeaderRow's "Pulls" (64).
            Text(formatCount(model.downloads ?? 0))
                .font(.callout.monospacedDigit())
                .frame(width: 64, alignment: .trailing)

            // Likes
            Text(formatCount(model.likes ?? 0))
                .font(.callout.monospacedDigit())
                .frame(width: 50, alignment: .trailing)

            // RAM estimate with color indicator — width matches
            // ColumnHeaderRow (120) so GGUF range strings like
            // "21.2–55.4 GB" stay on one line. `.lineLimit(1)` is the
            // belt-and-suspenders guard against any future format that
            // exceeds the budget — we'd rather truncate than wrap.
            HStack(spacing: 4) {
                Circle()
                    .fill(fitnessColor)
                    .frame(width: 8, height: 8)
                Text(model.ramEstimate)
                    .font(.callout.monospacedDigit())
                    .lineLimit(1)
            }
            .frame(width: 120, alignment: .trailing)

            // Last updated
            Text(formatRelativeDate(model.lastModifiedDate))
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(width: 64, alignment: .trailing)

            // Action cell — width matched to ColumnHeaderRow (120).
            actionCell
                .frame(width: 120, alignment: .center)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .opacity(disabled ? 0.4 : 1.0)
    }

    private var fitnessColor: Color {
        switch fitness {
        case .fits: return .green
        case .tight: return .yellow
        case .wontFit: return .red
        case .unknown: return .gray
        }
    }

    @State private var confirmDelete = false

    /// Resolved by the pure state machine so the branch ladder is unit-tested
    /// (`ModelBrowserSectionTests`). The key change: a ready model resolves to
    /// `.onDisk` and stays in the list, where it used to be filtered out of the
    /// search results entirely — vanishing at the exact moment it finished.
    private var action: ModelRowAction {
        ModelRowAction.resolve(
            isCompatible: model.isCompatible,
            isReady: isReady,
            status: state?.status,
            hasPartial: downloads.hasPartialDownload(model.id),
            progress: state?.fileProgress ?? 0
        )
    }

    /// The on-disk model this row's repo resolves to, when it's loadable as the
    /// server's chat model. nil for drafters, encoders, and media checkpoints —
    /// they're on disk and deletable, but "Use" would load something that can't
    /// serve a completion.
    private var usableModel: LocalModel? {
        ModelBrowserUse.pickableModel(
            atPath: downloads.existingModelDir(for: model.id),
            in: appState.localModels
        )
    }

    @ViewBuilder
    private var actionCell: some View {
        switch action {
        case .unsupported:
            Image(systemName: "nosign")
                .foregroundStyle(.secondary)
                .font(.caption)

        case .onDisk:
            HStack(spacing: 6) {
                if let usable = usableModel {
                    let use = ModelUseState.resolve(
                        selected: appState.selectedModelPath == usable.path,
                        serverStatus: server.status
                    )
                    if use == .idle {
                        Button("Use") { appState.selectedModelPath = usable.path }
                            .controlSize(.small)
                            .help("Load \(usable.name) as the server's model")
                    } else {
                        ModelUseBadge(state: use)
                    }
                } else {
                    Text("✓ On disk")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.green)
                }
                deleteButton
            }

        case .downloading(let progress):
            HStack(spacing: 4) {
                VStack(spacing: 1) {
                    ProgressView(value: progress)
                        .frame(width: 50)
                    Text(state?.percentFormatted ?? "")
                        .font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Button {
                    downloads.cancel(model.id)
                    appState.refreshModels()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Cancel download")
            }

        case .failed(let resumable):
            if model.isGgufRepo {
                GgufDownloadMenu(repoId: model.id, label: "Retry")
            } else {
                Button(resumable ? "Resume" : "Retry") {
                    downloads.start(repoId: model.id) { appState.refreshModels() }
                }
                .font(.callout)
                .controlSize(.small)
            }

        case .notDownloaded(let resumable):
            if model.isGgufRepo {
                // GGUF repos ship many quants — pick one from a menu.
                GgufDownloadMenu(repoId: model.id, label: resumable ? "Resume" : "Download")
            } else {
                Button(resumable ? "Resume" : "Download") {
                    downloads.start(repoId: model.id) { appState.refreshModels() }
                }
                .font(.callout)
                .controlSize(.small)
            }
        }
    }

    private var deleteButton: some View {
        Button {
            confirmDelete = true
        } label: {
            Image(systemName: "trash")
                .foregroundStyle(.red.opacity(0.7))
        }
        .buttonStyle(.plain)
        .font(.callout)
        .help("Delete model")
        .alert("Delete Model", isPresented: $confirmDelete) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                downloads.deleteModel(repoId: model.id)
                appState.refreshModels()
            }
        } message: {
            Text("Delete \(model.modelName)? This will remove all downloaded files.")
        }
    }
}

/// Download button for a GGUF repo: a menu of the repo's quant files. The list
/// is fetched lazily from the HF tree API the first time the menu is shown.
private struct GgufDownloadMenu: View {
    let repoId: String
    let label: String
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var quants: [String] = []
    @State private var loaded = false

    var body: some View {
        Menu {
            if !loaded {
                Text("Loading quants…")
            } else if quants.isEmpty {
                Text("No GGUF files found")
            } else {
                ForEach(quants, id: \.self) { file in
                    Button(DownloadManager.quantLabel(forFilename: file)) {
                        downloads.startGguf(repoId: repoId, ggufFilename: file) {
                            appState.refreshModels()
                        }
                    }
                }
            }
        } label: {
            Text(label)
        }
        .font(.callout)
        .controlSize(.small)
        .fixedSize()
        .task {
            guard !loaded else { return }
            quants = await downloads.listGgufFiles(repoId: repoId)
            loaded = true
        }
    }
}

// MARK: - Local Model Row

private struct LocalModelRow: View {
    let model: LocalModel
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var confirmDelete = false

    private var useState: ModelUseState {
        ModelUseState.resolve(
            selected: appState.selectedModelPath == model.path,
            serverStatus: server.status
        )
    }

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(model.name)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                    // Drafter checkpoints are real, supported models — they
                    // just aren't loadable as a target on their own. Show a
                    // distinct badge instead of the red "unsupported" warning
                    // that the generic check would otherwise render.
                    if model.kind == .drafter {
                        Text("Drafter")
                            .font(.system(size: 10).weight(.medium))
                            .foregroundStyle(.purple)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.purple.opacity(0.15), in: Capsule())
                            .help("Speculative-decoding drafter — pairs with a Gemma 4 base model in Settings, not loadable on its own.")
                    }
                }
                // Metadata caption: params · quant · architecture · engine, so
                // the row actually tells the user what the model is — previously
                // it was just a name and a delete button.
                HStack(spacing: 6) {
                    Text(model.metadataSummary)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    // Capability icons mirror the search rows.
                    if model.hasVision {
                        Image(systemName: "eye")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                            .help("Vision (image input)")
                    }
                    if model.hasToolCalling {
                        Image(systemName: "wrench")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                            .help("Tool calling")
                    }
                    // Only flag genuinely unsupported architectures. Drafters
                    // declare `gemma4_assistant` (not in supportedModelTypes)
                    // intentionally — the badge above already explains them.
                    if model.kind != .drafter, !model.isSupportedArchitecture {
                        Text("Unsupported")
                            .font(.system(size: 10).weight(.medium))
                            .foregroundStyle(.red.opacity(0.8))
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.red.opacity(0.12), in: Capsule())
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text(model.sizeFormatted)
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 90, alignment: .trailing)

            // Use + Delete. "Use" is the missing terminal action the browser
            // never had — before this, the only thing you could do with a model
            // you'd downloaded was throw it away. Once picked, the button is
            // replaced by an In-use badge rather than merely greyed out, so the
            // click produces visible feedback.
            HStack(spacing: 6) {
                if model.isChatPickable {
                    if useState == .idle {
                        Button("Use") { appState.selectedModelPath = model.path }
                            .controlSize(.small)
                            .help("Load \(model.name) as the server's model")
                    } else {
                        ModelUseBadge(state: useState)
                    }
                }
                Button {
                    confirmDelete = true
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(.red.opacity(0.7))
                }
                .buttonStyle(.plain)
                .font(.callout)
                .help("Delete model")
                .alert("Delete Model", isPresented: $confirmDelete) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        downloads.deleteModel(model)
                        appState.refreshModels()
                    }
                } message: {
                    Text("Delete \(model.name)? This will remove all downloaded files.")
                }
            }
            .frame(width: 120, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}

// MARK: - Active Download Row

private struct ActiveDownloadRow: View {
    let repoId: String
    let state: DownloadManager.DownloadState
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var modelName: String {
        repoId.components(separatedBy: "/").last ?? repoId
    }

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                Text(modelName)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)

                if state.status == .downloading, !state.statusText.isEmpty {
                    Text("[\(state.fileIndex)/\(state.fileCount)] \(state.statusText)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else if state.status == .failed, let error = state.error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if state.status == .downloading {
                HStack(spacing: 4) {
                    VStack(alignment: .trailing, spacing: 1) {
                        ProgressView(value: state.fileProgress)
                            .frame(width: 80)
                        Text("\(state.percentFormatted) \(state.speedFormatted)")
                            .font(.system(size: 9).monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Button {
                        downloads.cancel(repoId)
                        appState.refreshModels()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Cancel download")
                }
                .frame(width: 116, alignment: .trailing)
            } else if state.status == .failed {
                Button(downloads.hasPartialDownload(repoId) ? "Resume" : "Retry") {
                    downloads.start(repoId: repoId) { appState.refreshModels() }
                }
                .font(.callout)
                .controlSize(.small)
                .frame(width: 90, alignment: .trailing)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}

// MARK: - Drafter Pair Chip

/// Small inline indicator on Gemma 4 base-model rows that surfaces the
/// matching drafter. Two states:
///   - **On disk**: green "✓ Drafter" chip — silently confirms the pairing
///     is ready, hides the download CTA.
///   - **Not on disk**: clickable "Pair with drafter (+30-40%)" chip that
///     kicks off `DownloadManager.download(repoId:)` for the matching
///     `*-it-assistant-bf16` repo.
private struct DrafterPairChip: View {
    let variant: GemmaVariant
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var repoId: String { variant.drafterRepoId }
    private var isReady: Bool { downloads.isReady(repoId) }
    private var inFlight: Bool { downloads.downloads[repoId]?.status == .downloading }

    var body: some View {
        if isReady {
            Text("✓ Drafter")
                .font(.system(size: 10).weight(.medium))
                .foregroundStyle(.green)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Color.green.opacity(0.10))
                .clipShape(Capsule())
                .help("\(variant.drafterDirName) is downloaded and ready to pair.")
        } else if inFlight {
            Text("Drafter…")
                .font(.system(size: 10).weight(.medium))
                .foregroundStyle(.purple)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Color.purple.opacity(0.10))
                .clipShape(Capsule())
        } else {
            Button {
                Task {
                    await downloads.download(repoId: repoId)
                    appState.refreshModels()
                }
            } label: {
                Text("Pair with drafter +30-40%")
                    .font(.system(size: 10).weight(.medium))
                    .foregroundStyle(.purple)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(Color.purple.opacity(0.10))
                    .clipShape(Capsule())
            }
            .buttonStyle(.plain)
            .help("Download \(variant.drafterDirName) for +30-40% on code & agents (Gemma 4 only).")
        }
    }
}

// MARK: - Curated Drafters catalog

private struct DrafterCatalogRow: Identifiable {
    let variant: GemmaVariant
    let repoId: String
    let pairsWith: String
    let sizeEstimate: String
    var id: String { repoId }

    /// bf16 sizes (the uniform suffix used by `drafterRepoId`).
    static func sizeEstimate(for v: GemmaVariant) -> String {
        switch v {
        case .E2B:        return "~80 MB"
        case .E4B:        return "~120 MB"
        case .gemma12B:   return "~850 MB"
        case .gemma31B:   return "~150 MB"
        case .moe26B:     return "~120 MB"
        }
    }
}

private struct DrafterCatalogRowView: View {
    let row: DrafterCatalogRow
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var confirmDelete = false

    private var isReady: Bool { downloads.isReady(row.repoId) }
    private var state: DownloadManager.DownloadState? { downloads.downloads[row.repoId] }

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                Text(row.variant.drafterDirName)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    Text(row.pairsWith)
                        .font(.caption)
                        .foregroundStyle(.purple)
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text(row.sizeEstimate)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            actionControl
                .frame(width: 110, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private var actionControl: some View {
        if isReady {
            HStack(spacing: 6) {
                Text("✓ Available")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.green)
                Button {
                    confirmDelete = true
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(.red.opacity(0.7))
                }
                .buttonStyle(.plain)
                .font(.callout)
                .help("Delete drafter")
                .alert("Delete Drafter", isPresented: $confirmDelete) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        downloads.deleteModel(repoId: row.repoId)
                        appState.refreshModels()
                    }
                } message: {
                    Text("Delete \(row.variant.drafterDirName)?")
                }
            }
        } else if let s = state, s.status == .downloading {
            VStack(spacing: 1) {
                ProgressView(value: s.fileProgress)
                    .frame(width: 80)
                Text(s.percentFormatted)
                    .font(.system(size: 9).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        } else {
            Button(downloads.hasPartialDownload(row.repoId) ? "Resume" : "Download") {
                Task {
                    await downloads.download(repoId: row.repoId)
                    appState.refreshModels()
                }
            }
            .font(.callout)
            .controlSize(.small)
        }
    }
}
