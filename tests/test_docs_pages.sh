#!/bin/bash
# test_docs_pages.sh — static checks for the docs/ marketing site (GitHub Pages).
#
# Verifies, without a browser:
#   1. every feature deep-dive page exists at docs/<slug>/index.html
#   2. each page carries the SEO contract: <title>, meta description, canonical
#      URL matching its slug, Open Graph tags, JSON-LD, and the shared stylesheet
#   3. every relative href/src across all docs pages resolves to a real file
#      (screenshots/ refs are reported as PENDING, not failures — they are the
#      screenshot shopping list)
#   4. sitemap.xml lists the homepage + every deep-dive URL; robots.txt points at it
#   5. index.html links to every deep-dive page (internal-link SEO)
#
# Usage: ./tests/test_docs_pages.sh

set -u
cd "$(dirname "$0")/.." || exit 1
DOCS="docs"
BASE_URL="https://ddalcu.github.io/mlx-serve"

SLUGS=(
  claude-code-local
  lm-studio-alternative
  ollama-alternative
  image-generation
  video-generation
  voice-cloning
  tool-calling
  agent-sandbox
  speculative-decoding
  local-ai-assistant
)

PASS=0; FAIL=0; PEND=0
pass() { PASS=$((PASS+1)); }
fail() { FAIL=$((FAIL+1)); echo "FAIL: $1"; }
pend() { PEND=$((PEND+1)); echo "PENDING: $1"; }

check() { # check <file> <grep-pattern> <label>
  if grep -q -- "$2" "$1"; then pass; else fail "$3 (pattern not found: $2)"; fi
}

# ── 1+2: per-page existence + SEO contract ─────────────────────────────────
for slug in "${SLUGS[@]}"; do
  f="$DOCS/$slug/index.html"
  if [ ! -f "$f" ]; then
    fail "$slug: page missing ($f)"
    continue
  fi
  pass
  check "$f" "<title>"                                        "$slug: <title>"
  check "$f" 'name="description"'                             "$slug: meta description"
  check "$f" "rel=\"canonical\" href=\"$BASE_URL/$slug/\""    "$slug: canonical URL"
  check "$f" 'property="og:title"'                            "$slug: og:title"
  check "$f" 'property="og:image"'                            "$slug: og:image"
  check "$f" 'application/ld+json'                            "$slug: JSON-LD"
  check "$f" 'assets/feature.css'                             "$slug: shared stylesheet"
  check "$f" 'assets/feature.js'                              "$slug: shared script"
done

# ── shared assets ───────────────────────────────────────────────────────────
for a in "$DOCS/assets/feature.css" "$DOCS/assets/feature.js"; do
  if [ -f "$a" ]; then pass; else fail "shared asset missing: $a"; fi
done

# ── 3: relative link/image resolution across all docs html ─────────────────
html_files=("$DOCS/index.html")
for slug in "${SLUGS[@]}"; do
  [ -f "$DOCS/$slug/index.html" ] && html_files+=("$DOCS/$slug/index.html")
done

for f in "${html_files[@]}"; do
  dir="$(dirname "$f")"
  refs=$(grep -oE '(href|src)="[^"]+"' "$f" | sed -E 's/^(href|src)="//; s/"$//' | sort -u)
  while IFS= read -r ref; do
    [ -z "$ref" ] && continue
    case "$ref" in
      http://*|https://*|mailto:*|data:*|\#*|javascript:*) continue ;;
    esac
    target="${ref%%#*}"          # strip fragment
    target="${target%%\?*}"     # strip query
    [ -z "$target" ] && continue
    resolved="$dir/$target"
    # a trailing slash means directory-style URL -> index.html
    case "$resolved" in */) resolved="${resolved}index.html" ;; esac
    if [ -e "$resolved" ]; then
      pass
    elif [ -d "${dir}/${target}" ] && [ -f "${dir}/${target}/index.html" ]; then
      pass
    else
      case "$target" in
        *screenshots/*) pend "screenshot not yet added: $target (referenced by $f)" ;;
        *) fail "broken relative ref in $f: $ref" ;;
      esac
    fi
  done <<< "$refs"
done

# ── 4: sitemap + robots ─────────────────────────────────────────────────────
if [ -f "$DOCS/sitemap.xml" ]; then
  pass
  check "$DOCS/sitemap.xml" "$BASE_URL/</loc>" "sitemap: homepage entry"
  for slug in "${SLUGS[@]}"; do
    check "$DOCS/sitemap.xml" "$BASE_URL/$slug/</loc>" "sitemap: $slug entry"
  done
else
  fail "docs/sitemap.xml missing"
fi

if [ -f "$DOCS/robots.txt" ]; then
  pass
  check "$DOCS/robots.txt" "Sitemap: $BASE_URL/sitemap.xml" "robots.txt: sitemap reference"
else
  fail "docs/robots.txt missing"
fi

# ── 5: index.html links to every deep-dive page ─────────────────────────────
for slug in "${SLUGS[@]}"; do
  check "$DOCS/index.html" "href=\"$slug/\"" "index.html links to $slug/"
done

# ── 6: download CTAs point at the direct latest-DMG URL (version-free) ──────
# GitHub redirects releases/latest/download/<asset> to the newest release's
# asset, so this never needs a version bump. The asset name MLXCore.dmg is
# stable across releases (produced by app/build.sh).
DMG_URL="https://github.com/ddalcu/mlx-serve/releases/latest/download/MLXCore.dmg"
for f in "${html_files[@]}"; do
  check "$f" "$DMG_URL" "$f: direct DMG download CTA"
done

echo ""
echo "docs pages: $PASS passed, $FAIL failed, $PEND pending screenshots"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
