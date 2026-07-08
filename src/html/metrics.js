'use strict';
// Live metrics panel — the markup AND the polling logic in one file. The panel
// is injected into the `#mlx-metrics` mount on the index page (only present when
// the server ran with --metrics), then this polls the open /metrics.json feed
// once a second. Everything here is NON-PERSISTED: the sparkline history lives
// only in these JS ring buffers, derived live from the counters/histograms the
// server already exposes (no server-side time series is stored).
//
// Decode & prefill "tok/s" are ACTIVE speeds — Δtokens ÷ Δ(phase time) over a
// trailing window. The phase-time sums (decode_time_seconds / prefill_time_
// seconds) and token counters only advance when a request COMPLETES, so this is
// the true generation/prefill speed of recently-finished requests — NOT the
// idle-averaged counter rate (which reads ~0 between requests).
(function () {
  // Panel markup, injected into the page. A template literal, so the CSS/HTML
  // braces need no escaping — the reason this lives here and not inline in the
  // std.fmt-formatted index.html.
  const PANEL_HTML = `
<style>
.mhead{display:flex;align-items:center;gap:10px;margin:24px 0 10px}
.mhead h2{margin:0}
#m-status{font-size:11px;font-weight:600;letter-spacing:.02em;padding:2px 9px;border-radius:999px;background:#1a1e25;color:#7d8794}
#m-status.live{background:#0f2a17;color:#4ade80}
#m-status.err{background:#2a0f14;color:#ff95a8}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:640px){.mgrid{grid-template-columns:repeat(2,1fr)}}
.mtile{background:#0f1216;border:1px solid #1f242c;border-radius:8px;padding:12px 14px}
.mlbl{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#7d8794;margin-bottom:7px}
.mval{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:26px;font-weight:700;line-height:1;color:#e6e9ee}
.munit{font-size:12px;font-weight:400;color:#7d8794;margin-left:4px}
.msub{font-size:11px;color:#5b6470;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mbar{height:5px;background:#1f242c;border-radius:3px;margin-top:10px;overflow:hidden}
.mfill{height:100%;width:0;border-radius:3px;background:#3b82f6;transition:width .5s}
.mfill.warn{background:#f59e0b}.mfill.crit{background:#ef4444}
.mspark{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:640px){.mspark{grid-template-columns:1fr}}
.msparkbox{background:#0f1216;border:1px solid #1f242c;border-radius:8px;padding:10px 12px}
.msparkhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.msparkval{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;font-weight:700;color:#e6e9ee}
.msparkbox svg{width:100%;height:44px;display:block}
</style>
<div class=mhead><h2 style="margin:0">Live metrics</h2><span id=m-status>connecting…</span></div>
<div class=card style="padding:16px">
<div class=mgrid>
<div class=mtile><div class=mlbl>Decode</div><div class=mval><span id=m-decode-tps>—</span><span class=munit>tok/s</span></div><div class=msub id=m-decode-ms>— ms avg</div></div>
<div class=mtile><div class=mlbl>Prefill</div><div class=mval><span id=m-prefill-tps>—</span><span class=munit>tok/s</span></div><div class=msub id=m-prefill-ms>— ms avg</div></div>
<div class=mtile><div class=mlbl>Requests</div><div class=mval><span id=m-running>0</span><span class=munit>running</span></div><div class=msub id=m-waiting>0 waiting · — req/s</div></div>
<div class=mtile><div class=mlbl>Avg TTFT</div><div class=mval><span id=m-ttft>—</span><span class=munit>ms</span></div><div class=msub id=m-e2e>— ms e2e</div></div>
<div class=mtile><div class=mlbl>Cache hit rate</div><div class=mval><span id=m-cache>—</span><span class=munit>%</span></div><div class=msub id=m-cachedetail>— / — queries</div></div>
<div class=mtile><div class=mlbl>GPU</div><div class=mval><span id=m-gpu>0</span><span class=munit>%</span></div><div class=mbar><div class=mfill id=m-gpubar></div></div></div>
<div class=mtile><div class=mlbl>Memory</div><div class=mval><span id=m-mem>0</span><span class=munit>MB</span></div><div class=msub>physical footprint</div></div>
<div class=mtile><div class=mlbl>Generated</div><div class=mval><span id=m-gen>0</span><span class=munit>tok</span></div><div class=msub id=m-success>0 requests</div></div>
</div>
<div class=mspark>
<div class=msparkbox><div class=msparkhead><span class=mlbl>Decode tok/s · last 60s</span><span class=msparkval id=m-spark-decode-val>—</span></div><svg id=m-spark-decode viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
<div class=msparkbox><div class=msparkhead><span class=mlbl>Prefill tok/s · last 60s</span><span class=msparkval id=m-spark-prefill-val>—</span></div><svg id=m-spark-prefill viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
</div>
</div>`;

  const mount = document.getElementById('mlx-metrics');
  if (mount) mount.innerHTML = PANEL_HTML;

  const $ = (id) => document.getElementById(id);
  const samples = [];              // {t, gen, dsum, prompt, psum, req} ring buffer
  const RETAIN_MS = 120000;        // keep 2 min of samples for the 60s window
  const SPARK_N = 60;              // sparkline points (≈60s at 1 Hz)
  const decodeHist = [], prefillHist = [];
  let lastDecodeTps = null, lastPrefillTps = null;
  const hover = { decode: null, prefill: null };  // hovered point index per chart

  function fmt(v, d) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return v.toFixed(d === undefined ? 1 : d);
  }

  function setStatus(cls, txt) {
    const e = $('m-status');
    if (e) { e.className = cls; e.textContent = txt; }
  }

  // Newest sample that is at least winMs old (tightest window ≥ winMs); the
  // oldest retained sample while still warming up.
  function at(now, winMs) {
    let s = samples[0];
    for (const x of samples) { if (now - x.t >= winMs) s = x; else break; }
    return s;
  }

  // Draw a sparkline (auto-scaled) into an <svg>, set its value label, and — when
  // the mouse is over that chart (hover[key] set) — draw a marker at the hovered
  // point and show that point's value in the label instead of the latest.
  function spark(id, data, color, valId, key, dec) {
    const svg = $(id), label = $(valId);
    if (!svg) return;
    if (data.length < 2) {
      svg.innerHTML = '';
      if (label) label.textContent = data.length ? fmt(data[0], dec) : '—';
      return;
    }
    const max = Math.max.apply(null, data.concat([0.001]));
    const n = data.length, W = 300, H = 44, p = 3;
    const xs = new Array(n), ys = new Array(n);
    let pts = '';
    for (let i = 0; i < n; i++) {
      xs[i] = p + (i / (n - 1)) * (W - 2 * p);
      ys[i] = H - p - (data[i] / max) * (H - 2 * p);
      pts += (i ? ' ' : '') + xs[i].toFixed(1) + ',' + ys[i].toFixed(1);
    }
    let m =
      '<polyline points="' + pts + '" fill="none" stroke="' + color +
      '" stroke-width="1.5" stroke-linejoin="round"/>' +
      '<line x1="' + p + '" y1="' + (H - 1) + '" x2="' + (W - p) + '" y2="' + (H - 1) +
      '" stroke="#1f242c" stroke-width="1"/>';
    const hi = hover[key];
    if (hi !== null && hi >= 0 && hi < n) {
      m += '<line x1="' + xs[hi].toFixed(1) + '" y1="' + p + '" x2="' + xs[hi].toFixed(1) +
        '" y2="' + (H - 1) + '" stroke="' + color + '" stroke-width="1" opacity="0.35"/>' +
        '<circle cx="' + xs[hi].toFixed(1) + '" cy="' + ys[hi].toFixed(1) + '" r="2.5" fill="' + color + '"/>';
      if (label) label.textContent = fmt(data[hi], dec);
    } else if (label) {
      label.textContent = fmt(data[n - 1], dec);
    }
    svg.innerHTML = m;
  }

  // Wire mouse hover on a sparkline: map cursor x → nearest data point, mark it,
  // show that value in the chart's label; mouseleave restores the latest value.
  function attachSparkHover(id, key, getData, color, valId, dec) {
    const svg = $(id);
    if (!svg) return;
    svg.style.cursor = 'crosshair';
    svg.addEventListener('mousemove', function (e) {
      const data = getData();
      if (data.length < 2) return;
      const rect = svg.getBoundingClientRect();
      const frac = rect.width ? (e.clientX - rect.left) / rect.width : 0;
      hover[key] = Math.max(0, Math.min(data.length - 1, Math.round(frac * (data.length - 1))));
      spark(id, data, color, valId, key, dec);
    });
    svg.addEventListener('mouseleave', function () {
      hover[key] = null;
      spark(id, getData(), color, valId, key, dec);
    });
  }

  const histSum = (hist) => (hist && typeof hist.sum === 'number') ? hist.sum : 0;

  async function tick() {
    let d;
    try {
      const r = await fetch('/metrics.json', { cache: 'no-store' });
      if (r.status === 503) { setStatus('err', 'metrics disabled'); return; }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      d = await r.json();
    } catch (e) { setStatus('err', 'error: ' + e.message); return; }

    setStatus('live', '● live');
    const c = d.counters, g = d.gauges, h = d.histograms;
    const now = Date.now();

    const psum = histSum(h.prefill_time_seconds);
    // Live token count = completed generation tokens + tokens generated so far by
    // the in-flight slot(s), sampled server-side every 2s. It MOVES during a
    // running request (tokens only accrue during DECODE, so it's flat through
    // prefill) — unlike the completion histograms, which don't advance until a
    // request finishes.
    const liveTok = (g.generation_tokens_live != null) ? g.generation_tokens_live : c.generation_tokens_total;
    samples.push({ t: now, live: liveTok, prompt: c.prompt_tokens_total, psum: psum, req: c.requests_success_total });
    while (samples.length > 2 && now - samples[0].t > RETAIN_MS) samples.shift();

    // Decode tok/s — LIVE decode speed while a request runs, 0 when idle. Δ(live
    // token gauge) over a short window (tokens only accrue during DECODE). NO
    // carry-forward: when nothing is running it reads 0, so it drops back to 0
    // the moment decoding stops instead of staying elevated at the last speed.
    let decodeTps = 0;
    if (g.requests_running > 0) {
      const wl = at(now, 4000);
      if (wl) {
        const dt = (now - wl.t) / 1000;
        if (dt > 0) decodeTps = Math.max(0, (liveTok - wl.live) / dt);
      }
    }
    lastDecodeTps = decodeTps;
    // Prefill tok/s + req/s — completion-based (Δ over ~60s window); prefill is a
    // one-shot burst that's only recorded when the request finishes, so this
    // populates after the first completion.
    let reqRate = null;
    const wp = at(now, 60000);
    if (wp) {
      const dPre = psum - wp.psum, dPrompt = c.prompt_tokens_total - wp.prompt;
      if (dPre > 1e-6 && dPrompt > 0) lastPrefillTps = dPrompt / dPre;
      const dt2 = (now - wp.t) / 1000;
      if (dt2 > 0) reqRate = Math.max(0, (c.requests_success_total - wp.req) / dt2);
    }

    // Sparkline history (carry-forward once we have a first value).
    if (lastDecodeTps !== null) { decodeHist.push(lastDecodeTps); if (decodeHist.length > SPARK_N) decodeHist.shift(); }
    if (lastPrefillTps !== null) { prefillHist.push(lastPrefillTps); if (prefillHist.length > SPARK_N) prefillHist.shift(); }

    // Average latency from each histogram's sum/count (seconds → ms).
    const avgMs = (hist) => (hist && hist.count > 0) ? (hist.sum / hist.count) * 1000 : null;
    const ttft = avgMs(h.time_to_first_token_seconds);
    const e2e = avgMs(h.e2e_request_latency_seconds);
    const decodeMs = avgMs(h.decode_time_seconds);
    const prefillMs = avgMs(h.prefill_time_seconds);

    const cq = c.prefix_cache_queries_total, ch = c.prefix_cache_hits_total;
    const cachePct = cq > 0 ? Math.round((ch / cq) * 100) : null;

    $('m-decode-tps').textContent = lastDecodeTps !== null ? fmt(lastDecodeTps, 1) : '—';
    $('m-decode-ms').textContent = (decodeMs !== null ? fmt(decodeMs, 0) : '—') + ' ms avg';
    $('m-prefill-tps').textContent = lastPrefillTps !== null ? fmt(lastPrefillTps, 0) : '—';
    $('m-prefill-ms').textContent = (prefillMs !== null ? fmt(prefillMs, 0) : '—') + ' ms avg';
    $('m-running').textContent = g.requests_running;
    $('m-waiting').textContent = g.requests_waiting + ' waiting · ' + (reqRate !== null ? fmt(reqRate, 2) : '—') + ' req/s';
    $('m-ttft').textContent = ttft !== null ? fmt(ttft, 0) : '—';
    $('m-e2e').textContent = (e2e !== null ? fmt(e2e, 0) : '—') + ' ms e2e';
    $('m-cache').textContent = cachePct !== null ? cachePct : '—';
    $('m-cachedetail').textContent = ch + ' / ' + cq + ' queries';

    const gp = g.gpu_utilization_pct;
    $('m-gpu').textContent = gp;
    const bar = $('m-gpubar');
    bar.style.width = gp + '%';
    bar.className = 'mfill' + (gp >= 90 ? ' crit' : gp >= 70 ? ' warn' : '');

    $('m-mem').textContent = g.memory_mb;
    // Live count (completed + in-flight) so it moves during a running request.
    $('m-gen').textContent = fmt(liveTok, 0);
    $('m-success').textContent = fmt(c.requests_success_total, 0) + ' requests';

    spark('m-spark-decode', decodeHist, '#22c55e', 'm-spark-decode-val', 'decode', 1);
    spark('m-spark-prefill', prefillHist, '#3b82f6', 'm-spark-prefill-val', 'prefill', 0);
  }

  attachSparkHover('m-spark-decode', 'decode', function () { return decodeHist; }, '#22c55e', 'm-spark-decode-val', 1);
  attachSparkHover('m-spark-prefill', 'prefill', function () { return prefillHist; }, '#3b82f6', 'm-spark-prefill-val', 0);
  tick();
  setInterval(tick, 1000);
})();
