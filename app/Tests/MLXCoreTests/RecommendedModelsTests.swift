import XCTest
@testable import MLXCore

/// Pins the "Recommended" pane's data: two family catalogs (Gemma 4, Qwen
/// 3.5/3.6), each ascending by size, plus the RAM-requirements math that
/// drives the pane's dim-but-never-hide treatment. The pane exists to answer
/// "which model should I download" for someone with zero AI experience, so
/// every entry's copy and sizing claims are load-bearing — get either wrong
/// and the recommendation itself is wrong, not just a cosmetic bug.
final class RecommendedModelsTests: XCTestCase {

    private let GiB: UInt64 = 1_073_741_824

    // MARK: - Catalog shape

    /// The pane's whole layout assumes exactly three family sections.
    func testExactlyThreeFamiliesArePresent() {
        let families = Set(RecommendedModelPick.gemmaCatalog.map(\.family))
            .union(RecommendedModelPick.qwenCatalog.map(\.family))
            .union(RecommendedModelPick.hunyuanCatalog.map(\.family))
        XCTAssertEqual(families, [.gemma, .qwen, .hunyuan])
    }

    /// A family catalog can't be empty — a section with zero rows would be a
    /// dead header in the UI.
    func testNeitherFamilyCatalogIsEmpty() {
        XCTAssertFalse(RecommendedModelPick.gemmaCatalog.isEmpty)
        XCTAssertFalse(RecommendedModelPick.qwenCatalog.isEmpty)
        XCTAssertFalse(RecommendedModelPick.hunyuanCatalog.isEmpty)
    }

    /// Every entry in `gemmaCatalog` is actually Gemma, and every entry in
    /// `qwenCatalog` is actually Qwen — the section header promises this.
    func testEveryEntryMatchesItsCatalogsFamily() {
        for p in RecommendedModelPick.gemmaCatalog {
            XCTAssertEqual(p.family, .gemma, p.id)
        }
        for p in RecommendedModelPick.qwenCatalog {
            XCTAssertEqual(p.family, .qwen, p.id)
        }
        for p in RecommendedModelPick.hunyuanCatalog {
            XCTAssertEqual(p.family, .hunyuan, p.id)
        }
    }

    /// Each family list renders smallest-to-largest, so a beginner scans it
    /// as "gets more capable as you go".
    func testEachFamilyCatalogIsSortedAscendingBySize() {
        let gemmaSizes = RecommendedModelPick.gemmaCatalog.map(\.sizeGB)
        XCTAssertEqual(gemmaSizes, gemmaSizes.sorted())
        let qwenSizes = RecommendedModelPick.qwenCatalog.map(\.sizeGB)
        XCTAssertEqual(qwenSizes, qwenSizes.sorted())
        let hunyuanSizes = RecommendedModelPick.hunyuanCatalog.map(\.sizeGB)
        XCTAssertEqual(hunyuanSizes, hunyuanSizes.sorted())
    }

    /// No id collisions within or across the two catalogs — ids key the
    /// SwiftUI `ForEach`/download-state lookups.
    func testNoDuplicateIdsAcrossBothCatalogs() {
        let ids = (RecommendedModelPick.gemmaCatalog + RecommendedModelPick.qwenCatalog + RecommendedModelPick.hunyuanCatalog).map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count)
    }

    /// Every repo id must look like a real, resolvable HuggingFace path
    /// (`org/repo`, no whitespace) — a typo here silently 404s the download.
    func testRepoIdsAreWellFormed() {
        for p in RecommendedModelPick.gemmaCatalog + RecommendedModelPick.qwenCatalog + RecommendedModelPick.hunyuanCatalog {
            XCTAssertTrue(p.repoId.contains("/"), p.repoId)
            XCTAssertFalse(p.repoId.contains(" "), p.repoId)
            XCTAssertEqual(p.repoId.split(separator: "/").count, 2, p.repoId)
        }
    }

    /// Every entry needs real, non-empty plain-English copy — an empty blurb
    /// or tagline would silently render a blank description.
    func testEveryPickHasNonEmptyBeginnerCopy() {
        for p in RecommendedModelPick.gemmaCatalog + RecommendedModelPick.qwenCatalog + RecommendedModelPick.hunyuanCatalog {
            XCTAssertFalse(p.name.isEmpty, p.id)
            XCTAssertFalse(p.tagline.isEmpty, p.id)
            XCTAssertGreaterThan(p.blurb.count, 40, "\(p.id) blurb reads as a stub")
            XCTAssertFalse(p.highlights.isEmpty, p.id)
        }
    }

    // MARK: - Meets-system-requirements (reduced opacity, never hidden)

    /// A model well within this Mac's RAM meets requirements.
    func testMeetsSystemRequirementsWhenPlentyOfHeadroom() {
        XCTAssertTrue(RecommendedModelPick.gemmaE4B.meetsSystemRequirements(physicalMemoryBytes: 16 * GiB))
    }

    /// A model bigger than this Mac's total RAM does not meet requirements —
    /// this is the signal that sorts it behind the "Requires more RAM"
    /// disclosure, never a reason to drop it from the list.
    func testDoesNotMeetSystemRequirementsWhenTooBig() {
        XCTAssertFalse(RecommendedModelPick.gemma31B8bit.meetsSystemRequirements(physicalMemoryBytes: 16 * GiB))
    }

    /// The threshold includes the same ~20% overhead the rest of the app
    /// budgets for RAM-vs-disk-weight, so a model whose weights alone are
    /// just under total RAM still correctly reads as "won't fit".
    func testRequirementsThresholdIncludesOverhead() {
        // gemma31B: 17.2 GB weights -> ~20.64 GB needed. 18 GB of RAM covers
        // the raw weights but not the overhead.
        let pick = RecommendedModelPick.gemma31B
        XCTAssertFalse(pick.meetsSystemRequirements(physicalMemoryBytes: 18 * GiB))
        XCTAssertTrue(pick.meetsSystemRequirements(physicalMemoryBytes: 32 * GiB))
    }

    /// Hunyuan 3 is the "over 128 GB" recommendation: on a 128 GB Mac it
    /// sorts behind the "Requires more RAM" disclosure (it RUNS there, with a
    /// minimal context window — the blurb says so — but the recommendation
    /// targets bigger Macs), while a 192 GB+ Mac sees it inline. The override
    /// exists because the generic weights×1.2 formula reads 105 GB as
    /// "fits on 128" — measured live, a 128 GB Mac needs the memory-preflight
    /// override and pins a ~3K context.
    func testHunyuan3IsGatedAbove128GB() {
        let hy3 = RecommendedModelPick.hy3_295b
        XCTAssertFalse(hy3.meetsSystemRequirements(physicalMemoryBytes: 128 * GiB))
        XCTAssertTrue(hy3.meetsSystemRequirements(physicalMemoryBytes: 192 * GiB))
        XCTAssertTrue(hy3.blurb.contains("128 GB"), "blurb must carry the runs-on-128 caveat")
    }

    // MARK: - Partitioning (inline vs "Requires more RAM" disclosure)

    /// On a big enough Mac, everything in a family fits — the disclosure
    /// never appears (nothing goes into `requiresMoreRAM`).
    func testPartitionPutsEverythingInFitsOnAHighRamMac() {
        let split = RecommendedModelPick.gemmaCatalog.partitionedByRequirements(physicalMemoryBytes: 128 * GiB)
        XCTAssertEqual(split.fits.count, RecommendedModelPick.gemmaCatalog.count)
        XCTAssertTrue(split.requiresMoreRAM.isEmpty)
    }

    /// On a small Mac, the biggest Gemma picks land in `requiresMoreRAM`
    /// while the small ones stay in `fits` — and nothing is dropped: the two
    /// buckets together must reconstruct the original catalog.
    func testPartitionSplitsBySizeOnALowRamMac() {
        let split = RecommendedModelPick.gemmaCatalog.partitionedByRequirements(physicalMemoryBytes: 16 * GiB)
        XCTAssertTrue(split.fits.contains(.gemmaE2B))
        XCTAssertTrue(split.requiresMoreRAM.contains(.gemma31B8bit))
        XCTAssertEqual(Set(split.fits + split.requiresMoreRAM), Set(RecommendedModelPick.gemmaCatalog))
    }

    /// Each bucket preserves the catalog's ascending-size order — the
    /// partition must not reshuffle, only split.
    func testPartitionPreservesAscendingOrderWithinEachBucket() {
        let split = RecommendedModelPick.gemmaCatalog.partitionedByRequirements(physicalMemoryBytes: 16 * GiB)
        XCTAssertEqual(split.fits.map(\.sizeGB), split.fits.map(\.sizeGB).sorted())
        XCTAssertEqual(split.requiresMoreRAM.map(\.sizeGB), split.requiresMoreRAM.map(\.sizeGB).sorted())
    }

    // MARK: - Known-good entries (regression pins)

    /// The native-MTP Qwen 3.6 build is strictly better than a plain 4-bit
    /// Qwen 3.6 27B at the same size (built-in speculative-decode speedup,
    /// same weights) — it must be the catalog's Qwen 3.6 27B pick, not the
    /// plain build.
    func testQwen36TwentySevenBPickUsesTheMtpBuild() {
        let repoIds = RecommendedModelPick.qwenCatalog.map(\.repoId)
        XCTAssertTrue(repoIds.contains("ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve"))
    }

    func testGemma4EverydayPicksArePresent() {
        let repoIds = Set(RecommendedModelPick.gemmaCatalog.map(\.repoId))
        XCTAssertTrue(repoIds.contains("mlx-community/gemma-4-e4b-it-4bit"))
        XCTAssertTrue(repoIds.contains("mlx-community/gemma-4-12b-it-4bit"))
    }

    /// The old 0.8B entry-level Qwen pick was replaced with 9B — too small
    /// to be a meaningful comparison against the Gemma lineup.
    func testEntryLevelQwenPickIsNineBNotZeroEightB() {
        let repoIds = RecommendedModelPick.qwenCatalog.map(\.repoId)
        XCTAssertTrue(repoIds.contains("mlx-community/Qwen3.5-9B-MLX-4bit"))
        XCTAssertFalse(repoIds.contains { $0.contains("0.8B") })
    }
}
