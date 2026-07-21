// website_tier_list_logic.mjs — unit-tests the pure logic embedded in
// website/llm-tier-list/index.html: Wilson-score tiering, vote sanitization,
// tally aggregation, and the Hugging Face popularity pipeline (canonical
// top-level model grouping, quant/finetune roll-up rules, size→RAM tiers).
// Invoked by test_website_pages.sh when node is available; exits non-zero on
// the first failed assertion.
//
// It evals the page's module script up to the DOM-dependent rendering section,
// so the code under test is the exact code the browser runs — no copies.
import { readFileSync } from "node:fs";

const html = readFileSync("website/llm-tier-list/index.html", "utf8");
const script = html.split('<script type="module">')[1]?.split("</script>")[0];
if (!script) { console.error("ASSERT FAIL: module script not found in page"); process.exit(1); }
const pure = script.split("// ── rendering")[0];
if (pure.length === script.length) { console.error("ASSERT FAIL: rendering marker missing"); process.exit(1); }

const asserts = `
function assert(c, m) { if (!c) { console.error("ASSERT FAIL: " + m); process.exit(1); } }

// ── Wilson-score tiering: needs MIN_VOTES to rank, consistency to climb ────
assert(wilsonLower(0, 0) === 0, "wilson of no votes is 0");
assert(wilsonLower(8, 0) > wilsonLower(3, 0), "confidence grows with sample size");
assert(tierFor(0, 0) === "U", "no votes stays unranked");
assert(tierFor(1, 0) === "C", "first upvote enters cautiously at C (MIN_VOTES=1)");
assert(tierFor(2, 0) === "B", "2-0 climbs to B");
assert(tierFor(3, 0) === "A", "3-0 ranks A (not yet S)");
assert(tierFor(8, 0) === "S", "8-0 reaches S");
assert(tierFor(0, 5) === "D", "0-5 is D");

// ── HF pipeline: quants roll UP, finetunes drop OUT, one entry per model ───
const NOW = Date.parse("2026-07-20T00:00:00Z");
const FIX_TEXT = [
  // official instruct model (finetune-of-own-Base is the normal official shape)
  { id: "Qwen/Qwen3.6-27B", downloads: 500000,
    tags: ["transformers", "safetensors", "qwen3_5", "text-generation",
           "base_model:finetune:Qwen/Qwen3.6-27B-Base", "license:apache-2.0"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-01T00:00:00.000Z" },
  // its pretrain sibling — must merge into the same top-level entry
  { id: "Qwen/Qwen3.6-27B-Base", downloads: 90000,
    tags: ["transformers", "qwen3_5", "text-generation"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-01T00:00:00.000Z" },
  // pure quantized conversions — downloads roll up into the official entry
  { id: "mlx-community/Qwen3.6-27B-4bit", downloads: 40000,
    tags: ["mlx", "qwen3_5", "text-generation", "base_model:Qwen/Qwen3.6-27B",
           "base_model:quantized:Qwen/Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-10T00:00:00.000Z" },
  { id: "mlx-community/Qwen3.6-27B-8bit", downloads: 10000,
    tags: ["mlx", "qwen3_5", "base_model:quantized:Qwen/Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-10T00:00:00.000Z" },
  // the official org's OWN quant release (live-observed NVFP4 class) — must
  // merge into the base entry, not fork; packed param count must not shrink
  // the size estimate
  { id: "Qwen/Qwen3.6-27B-NVFP4", downloads: 60000,
    tags: ["transformers", "qwen3_5", "text-generation"],
    safetensors: { total: 14000000000 }, createdAt: "2026-04-01T00:00:00.000Z" },
  // community finetune: popular, but NOT a top-level model — rejected
  { id: "coolhacker/Dolphin-Qwen3.6-27B", downloads: 999999,
    tags: ["qwen3_5", "text-generation", "base_model:finetune:Qwen/Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-03-01T00:00:00.000Z" },
  // quant OF a community finetune — also rejected (target org not official)
  { id: "mlx-community/Dolphin-Qwen3.6-27B-4bit", downloads: 888888,
    tags: ["mlx", "qwen3_5", "base_model:quantized:coolhacker/Dolphin-Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-03-02T00:00:00.000Z" },
  // unknown architecture, not a GGUF release — rejected
  { id: "SomeOrg/WeirdArch-70B", downloads: 777777,
    tags: ["superarch", "text-generation"],
    safetensors: { total: 70000000000 }, createdAt: "2026-01-01T00:00:00.000Z" },
  // modern arch beyond mlx-serve's native dispatch — popularity gate keeps it
  { id: "openai/gpt-oss-20b", downloads: 7200000,
    tags: ["transformers", "safetensors", "gpt_oss", "text-generation"],
    safetensors: { total: 21000000000 }, createdAt: "2025-08-05T00:00:00.000Z" },
  // LEGACY arch from an official org — must stay off the board
  { id: "Qwen/Qwen2.5-7B-Instruct", downloads: 11500000,
    tags: ["transformers", "safetensors", "qwen2", "text-generation"],
    safetensors: { total: 7600000000 }, createdAt: "2024-09-16T00:00:00.000Z" },
  // pre-2024 model on a still-listed arch — date floor must reject it
  { id: "meta-llama/Llama-2-7b-hf", downloads: 700000,
    tags: ["transformers", "safetensors", "llama", "text-generation"],
    safetensors: { total: 6700000000 }, createdAt: "2023-07-18T00:00:00.000Z" },
  // new-lab official release (Ornith = deepreinforce-ai, arch qwen3_5_moe)
  { id: "deepreinforce-ai/Ornith-1.0-35B", downloads: 1350838,
    tags: ["transformers", "safetensors", "qwen3_5_moe", "text-generation"],
    safetensors: { total: 35000000000 }, createdAt: "2026-06-25T00:00:00.000Z" },
  // official-org GGUF release with NO arch tag (Bonsai class) — gguf pass-through
  { id: "prism-ml/Bonsai-27B-gguf", downloads: 1262894,
    tags: ["llama.cpp", "gguf", "conversational", "1-bit"],
    createdAt: "2026-07-04T00:00:00.000Z" },
];
const FIX_VISION = [
  // recent official vision model → vision tag + New badge
  { id: "google/gemma-4-e2b-it", downloads: 800000,
    tags: ["transformers", "gemma4", "image-text-to-text",
           "base_model:finetune:google/gemma-4-e2b"],
    safetensors: { total: 5000000000 }, createdAt: "2026-07-05T00:00:00.000Z" },
];

const groups = groupHfModels(FIX_TEXT, FIX_VISION, NOW);
const byId = Object.fromEntries(groups.map((g) => [g.id, g]));

const qwen = byId["qwen3.6-27b"];
assert(qwen, "official model present under its normalized id");
assert(qwen.downloads === 700000, "quant + Base + official-quant downloads rolled up (got " + (qwen && qwen.downloads) + ")");
assert(qwen.params === "27B", "packed-quant param count doesn't shrink the size (got " + (qwen && qwen.params) + ")");
assert(prettyName("gemma-3-270m") === "Gemma 3 270M", "sub-B param suffix uppercased");
assert(prettyName("gpt-oss-20b") === "GPT OSS 20B", "short alpha tokens uppercased");
assert(normalizeBaseName("MiniMax-M3-MXFP8") === "minimax-m3", "mxfp8 quant suffix stripped");
assert(qwen.name === "Qwen3.6 27B", "pretty top-level name (got " + (qwen && qwen.name) + ")");
assert(qwen.minRam === 24, "27B at ~4-bit lands on the 24 GB tier (got " + (qwen && qwen.minRam) + ")");
assert(groups.filter((g) => g.id.includes("qwen3.6-27b")).length === 1, "one entry per top-level model");
assert(!groups.some((g) => g.id.includes("dolphin")), "community finetunes rejected");
assert(!groups.some((g) => g.id.includes("weirdarch")), "unknown non-GGUF architectures rejected");
assert(byId["gpt-oss-20b"], "modern arch beyond native dispatch is listed (gpt_oss)");
assert(!byId["qwen2.5-7b"], "legacy archs stay off the board even from official orgs");
assert(!byId["llama-2-7b"], "pre-2024 releases rejected by the date floor");
assert(byId["ornith-1.0-35b"] && byId["ornith-1.0-35b"].name === "Ornith 1.0 35B",
  "new-lab org (deepreinforce-ai) is listed");
assert(byId["bonsai-27b"] && byId["bonsai-27b"].isNew,
  "official-org GGUF release without an arch tag passes (Bonsai class)");

const gemma = byId["gemma-4-e2b"];
assert(gemma, "vision-pipeline model present (-it stripped from id)");
assert(gemma.name === "Gemma 4 E2B", "vision model pretty name (got " + (gemma && gemma.name) + ")");
assert(gemma.tags.includes("vision"), "image-text-to-text marks vision");
assert(gemma.isNew === true, "recently created model flagged New");
assert(qwen.isNew === false, "old model not flagged New");
assert(gemma.minRam === 8, "5B at ~4-bit fits the 8 GB tier");

// vote ids must be Firestore-map-key-safe and org-free
for (const g of groups) {
  assert(/^[a-z0-9][a-z0-9._+-]*$/.test(g.id), "id is a safe stable key: " + g.id);
}

// seed union: pinned models HF can't surface (GGUF-only) survive, no dupes
const merged = mergeWithSeed(groups, SEED_MODELS);
assert(merged.some((m) => m.id === "deepseek-v4-flash"), "pinned seed survives merge when not found dynamically");
assert(merged.filter((m) => m.id === "qwen3.6-27b").length === 1, "dynamic entry wins over same-id seed");

// ── vote sanitization: at most one exact ±1 per KNOWN model per account ────
setActiveModels(merged);
const s = sanitize({ "qwen3.6-27b": 1, "bogus-model": 1, "gemma-4-e2b": 2, "deepseek-v4-flash": -1 });
assert(JSON.stringify(s) === JSON.stringify({ "qwen3.6-27b": 1, "deepseek-v4-flash": -1 }),
  "sanitize drops unknown ids and non-plus-minus-1 values");

const t = talliesFrom([{ "qwen3.6-27b": 1 }, { "qwen3.6-27b": 1, "deepseek-v4-flash": -1 }]);
assert(t["qwen3.6-27b"].up === 2 && t["qwen3.6-27b"].down === 0 && t["deepseek-v4-flash"].down === 1,
  "tallies aggregate across voter docs");

// ── unranked table: text filter + vote-independent ordering ────────────────
const probe = { id: "qwen3.6-27b", name: "Qwen3.6 27B", params: "27B", tags: ["vision"], downloads: 5 };
assert(matchesQuery(probe, ""), "empty query matches everything");
assert(matchesQuery(probe, "  qWeN "), "name match is case/whitespace-insensitive");
assert(matchesQuery(probe, "vision"), "tags are searchable");
assert(matchesQuery(probe, "27b"), "params are searchable");
assert(!matchesQuery(probe, "llama"), "non-matches are filtered out");
const order = unrankedOrder([
  { id: "a", name: "A", downloads: 5 },
  { id: "b", name: "B", downloads: 50 },
  { id: "c", name: "C", downloads: 0 },
  { id: "d", name: "D", downloads: 5 },
]).map((m) => m.id);
assert(JSON.stringify(order) === JSON.stringify(["b", "a", "d", "c"]),
  "unranked order is downloads-desc + name tiebreak — votes can never reorder it (got " + order + ")");

// ── promotion detection: celebrate real U→tier moves, never a load storm ───
const prev = {};
// first render with real tallies only BASELINES (track=false): models that
// are already ranked on page load must not celebrate
let promos = promotionsSince(prev, [{ id: "m1", tier: "A" }, { id: "m2", tier: "U" }], false);
assert(promos.length === 0, "baseline render never celebrates");
// a later render where m2 earned its votes — that's a real promotion
promos = promotionsSince(prev, [{ id: "m1", tier: "A" }, { id: "m2", tier: "B" }], true);
assert(promos.length === 1 && promos[0].id === "m2", "U-to-ranked transition detected");
// staying ranked or moving between tiers is not a promotion
promos = promotionsSince(prev, [{ id: "m1", tier: "S" }, { id: "m2", tier: "B" }], true);
assert(promos.length === 0, "tier-to-tier moves don't fire the effect");

// ── seed data integrity: unique ids, labeled tags, minRam on a filter tier ─
const ids = SEED_MODELS.map((m) => m.id);
assert(new Set(ids).size === ids.length, "seed ids are unique");
const ramTiers = new Set([8, 16, 24, 32, 48, 64, 96, 128, 192]);
for (const m of SEED_MODELS) {
  assert(ramTiers.has(m.minRam), "seed minRam matches a filter tier: " + m.id);
  for (const tag of m.tags) assert(TAG_LABELS[tag], "seed tag has a label: " + tag);
}
console.log("tier-list logic OK (" + groups.length + " fixture groups, " + SEED_MODELS.length + " seeds)");
`;

eval(pure + asserts);
