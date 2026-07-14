// Tab switching
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "decisions") refreshHistory();
    if (btn.dataset.tab === "prompts") refreshPrompts();
  });
});

// WebSocket — use wss:// on HTTPS (required behind Cloudflare)
const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${wsProto}//${location.host}/ws`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.event === "scan_complete" || msg.event === "simulation_reset") refreshAll();
};

// Comparison Chart — guard against CDN failure so data tables still load
let compChart = null;
try {
  const compCtx = document.getElementById("comparison-chart")?.getContext("2d");
  if (compCtx && typeof Chart !== "undefined") {
    compChart = new Chart(compCtx, {
  type: "line",
  data: {
    datasets: [
      { label: "Claude", borderColor: "#7c3aed", backgroundColor: "rgba(124,58,237,0.08)", data: [], cubicInterpolationMode: "monotone", pointRadius: 2 },
      { label: "GPT",    borderColor: "#0ea5e9", backgroundColor: "rgba(14,165,233,0.08)",  data: [], cubicInterpolationMode: "monotone", pointRadius: 2 },
    ],
  },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: "#8b949e" } } },
    scales: {
      x: {
        type: "time",
        ticks: { color: "#8b949e", maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
        grid: { color: "#21262d" },
        time: {
          tooltipFormat: "MMM d, HH:mm",
          displayFormats: { hour: "MMM d HH:mm", day: "MMM d", week: "MMM d" },
        },
      },
      y: { ticks: { color: "#8b949e" }, grid: { color: "#21262d" } },
    },
  },
    });
  }
} catch (err) {
  console.warn("Chart init failed — metrics will still load:", err);
}

const METRIC_LABELS = {
  total_trades: "Total Trades",
  win_rate: "Win Rate (%)",
  avg_rrr: "Avg RRR",
  sharpe_ratio: "Sharpe Ratio",
  max_drawdown_pct: "Max Drawdown (%)",
  total_return_pct: "Total Return (%)",
  avg_trade_duration_days: "Avg Duration (days)",
  optimization_metric: "Expectancy (mean R)",
};

const TRACKS = ["claude", "gpt", "claude-opt", "gpt-opt"];
// Options tracks run on 10x the paper capital, so their SEK curves would dwarf
// the stock curves — the first-page chart stays stocks-only.
const CHART_TRACKS = ["claude", "gpt"];

async function refreshAll() {
  await Promise.all([
    refreshStatus(),
    refreshComparison(),
    refreshDecisions(),
    ...TRACKS.map(refreshTrack),
  ]);
  document.getElementById("last-update").textContent =
    "Updated " + new Date().toLocaleTimeString();
}

async function refreshStatus() {
  const data = await fetchJSON("/api/status");
  if (!data) return;
  const nb = document.getElementById("nordic-status");
  const eb = document.getElementById("eu-status");
  const ub = document.getElementById("us-status");
  nb.textContent = "Nordic: " + (data.nordic_open ? "OPEN" : "CLOSED");
  nb.className = "badge " + (data.nordic_open ? "open" : "closed");
  eb.textContent = "EU: " + (data.eu_open ? "OPEN" : "CLOSED");
  eb.className = "badge " + (data.eu_open ? "open" : "closed");
  ub.textContent = "US: " + (data.us_open ? "OPEN" : "CLOSED");
  ub.className = "badge " + (data.us_open ? "open" : "closed");

  // Show warning banner if a track's API key is missing
  const warn = document.getElementById("gpt-key-warning");
  if (warn) warn.style.display = data.gpt_configured === false ? "block" : "none";
}

async function refreshComparison() {
  const data = await fetchJSON("/api/comparison");
  if (!data) return;

  // Equity curves — dataset order matches CHART_TRACKS (stocks only)
  if (compChart) {
    CHART_TRACKS.forEach((track, i) => {
      compChart.data.datasets[i].data =
        (data[track]?.equity_curve || []).map((p) => ({ x: p.date, y: p.equity }));
    });
    compChart.update();
  }

  // Metrics table
  const tbody = document.getElementById("comparison-tbody");
  tbody.innerHTML = "";
  for (const [key, label] of Object.entries(METRIC_LABELS)) {
    const cells = TRACKS.map(t => `<td>${fmt(data[t]?.metrics?.[key] ?? "—")}</td>`).join("");
    tbody.innerHTML += `<tr><td>${label}</td>${cells}</tr>`;
  }
}

const ACTION_CLASS = { BUY: "act-buy", PASS: "act-pass", HOLD: "act-hold", SELL: "act-sell", BLOCKED: "act-blocked", ERROR: "act-blocked" };

function decisionCard(d, showTime) {
  const cls = ACTION_CLASS[d.action] || "act-hold";
  const conf = (d.confidence !== undefined && d.confidence !== null) ? ` · conf ${d.confidence}` : "";
  const rrr = (d.rrr !== undefined && d.rrr !== null) ? ` · RRR ${d.rrr}` : "";
  const market = (d.market || "").toUpperCase();
  const time = (showTime && d.timestamp) ? `<span class="decision-time">${new Date(d.timestamp).toLocaleString()}</span>` : "";
  const reason = d.reason ? `<div class="decision-blocked">Blocked: ${d.reason}</div>` : "";
  const why = d.reasoning ? `<div class="decision-reason">${d.reasoning}</div>` : "";
  return `<div class="decision-card">
    <div class="decision-head">
      <span class="track-pill track-${d.track}">${d.track}</span>
      <strong>${d.ticker}</strong>
      <span class="action-pill ${cls}">${d.action}</span>
      <span class="decision-meta">${market}${d.regime ? " · " + d.regime : ""}${conf}${rrr}</span>
      ${time}
    </div>
    ${reason}
    ${why}
  </div>`;
}

async function refreshDecisions() {
  const data = await fetchJSON("/api/decisions");
  const list = document.getElementById("decisions-list");
  if (!list) return;
  if (!data) { list.innerHTML = "<p class='neutral'>No decisions yet.</p>"; return; }

  const rows = [];
  let latestTs = null;
  for (const [market, scan] of Object.entries(data)) {
    if (scan?.timestamp && (!latestTs || scan.timestamp > latestTs)) latestTs = scan.timestamp;
    for (const d of (scan?.decisions || [])) rows.push({ ...d, market });
  }

  const meta = document.getElementById("decisions-meta");
  if (meta) meta.textContent = latestTs ? "— scanned " + new Date(latestTs).toLocaleString() : "";

  if (rows.length === 0) {
    list.innerHTML = "<p class='neutral'>No decisions in the latest scan (no candidates passed the screener).</p>";
    return;
  }
  list.innerHTML = rows.map(d => decisionCard(d, false)).join("");
}

async function refreshHistory() {
  const list = document.getElementById("history-list");
  if (!list) return;
  const track = document.getElementById("hist-track")?.value || "";
  const action = document.getElementById("hist-action")?.value || "";
  const ticker = document.getElementById("hist-ticker")?.value.trim() || "";
  const params = new URLSearchParams({ limit: "150" });
  if (track) params.set("track", track);
  if (action) params.set("action", action);
  if (ticker) params.set("ticker", ticker);

  const data = await fetchJSON("/api/decisions/history?" + params.toString());
  const rows = data?.decisions || [];
  list.innerHTML = rows.length
    ? rows.map(d => decisionCard(d, true)).join("")
    : "<p class='neutral'>No decisions recorded yet for this filter.</p>";
}

async function refreshTrack(track) {
  const [pData, tData] = await Promise.all([
    fetchJSON(`/api/portfolio/${track}`),
    fetchJSON(`/api/trades/${track}?limit=30`),
  ]);

  if (pData) {
    const snap = pData.snapshot;
    const metrics = pData.metrics;
    document.getElementById(`${track}-snapshot`).innerHTML = [
      ["Equity", "SEK " + fmtNum(snap.equity)],
      ["Cash", "SEK " + fmtNum(snap.cash)],
      ["Open Positions", snap.open_positions_count],
      ["Win Rate", (metrics.win_rate ?? 0) + "%"],
      ["Avg RRR", metrics.avg_rrr ?? "—"],
      ["Sharpe", metrics.sharpe_ratio ?? "—"],
      ["Drawdown", snap.drawdown_pct + "%"],
      ["Total Trades", snap.total_trades],
    ].map(([label, value]) =>
      `<div class="snapshot-item"><div class="label">${label}</div><div class="value">${value}</div></div>`
    ).join("");

    // Open positions table
    const posBody = document.getElementById(`${track}-positions`);
    posBody.innerHTML = (pData.open_positions || []).map(p => {
      const pnlClass = p.unrealised_pnl_pct >= 0 ? "pos" : "neg";
      const days = Math.round((Date.now() - new Date(p.entry_time)) / 86400000);
      const qty = p.quantity % 1 === 0 ? p.quantity : p.quantity.toFixed(2);
      return `<tr>
        <td><strong>${p.ticker}</strong></td>
        <td>${p.exchange || "—"}</td>
        <td>${fmt(p.entry_price)}</td>
        <td>${fmt(p.current_price)}</td>
        <td>${qty}</td>
        <td>SEK ${fmtNum(p.market_value)}</td>
        <td class="${pnlClass}">${p.unrealised_pnl_pct}%</td>
        <td class="neg">${fmt(p.stop_loss)}</td>
        <td class="pos">${fmt(p.target)}</td>
        <td class="neutral">${days}d</td>
      </tr>`;
    }).join("") || `<tr><td colspan="10" class="neutral">No open positions</td></tr>`;
  }

  if (tData) {
    const tradeBody = document.getElementById(`${track}-trades`);
    tradeBody.innerHTML = (tData.trades || []).map(t => {
      const pnlClass = t.pnl_pct >= 0 ? "pos" : "neg";
      return `<tr>
        <td><strong>${t.ticker}</strong></td>
        <td class="${pnlClass}">${t.pnl_pct}%</td>
        <td>${t.rrr_achieved}</td>
        <td class="neutral">${t.duration_days}d</td>
        <td class="neutral">${t.regime || "—"}</td>
        <td class="neutral">${t.exit_reason}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="6" class="neutral">No closed trades yet</td></tr>`;
  }

  // Heuristics
  const hData = await fetchJSON(`/api/heuristics/${track}?page_size=50`);
  if (hData) {
    document.getElementById(`heuristics-${track}-list`).innerHTML =
      (hData.heuristics || []).map(h =>
        `<div class="heuristic-card">
          <div class="trigger">IF: ${h.trigger}</div>
          <div class="action">→ ${h.action}</div>
          <div class="heuristic-meta">
            <span>Quality: ${h.quality_score?.toFixed(1)}</span>
            <span>Used: ${h.access_count}×</span>
            ${h.outcome_count ? `<span>Trades: ${h.outcome_count} (${((h.cumulative_pnl_pct || 0) * 100).toFixed(1)}% P&L)</span>` : ""}
            <span>${h.market} | ${h.regime}</span>
            ${h.is_core ? '<span class="core-badge">CORE</span>' : ""}
          </div>
        </div>`
      ).join("") || "<p class='neutral'>No heuristics yet — trades needed first.</p>";
  }
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return v;
}

function fmtNum(v) {
  if (typeof v !== "number") return v;
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    return r.ok ? r.json() : null;
  } catch { return null; }
}

// Reset button
const _SCAN_PHASES = [
  [0,  "Fetching market data…"],
  [5,  "Running screener…"],
  [12, "Analyzing news & macro…"],
  [20, "Getting AI decisions…"],
  [35, "Validating risk rules…"],
];

function _showScanToast(market) {
  const toast   = document.getElementById("scan-toast");
  const spinner = toast.querySelector(".scan-spinner");
  const msg     = document.getElementById("scan-toast-msg");
  const timer   = document.getElementById("scan-toast-timer");
  toast.classList.remove("hidden", "scan-toast--done", "scan-toast--error");
  spinner.style.display = "";
  timer.textContent = "0s";

  const start = Date.now();
  let phaseIdx = 0;
  msg.textContent = `${market.toUpperCase()}: ${_SCAN_PHASES[0][1]}`;

  const tick = setInterval(() => {
    const elapsed = Math.round((Date.now() - start) / 1000);
    timer.textContent = `${elapsed}s`;
    while (phaseIdx + 1 < _SCAN_PHASES.length && elapsed >= _SCAN_PHASES[phaseIdx + 1][0]) phaseIdx++;
    msg.textContent = `${market.toUpperCase()}: ${_SCAN_PHASES[phaseIdx][1]}`;
  }, 500);

  return { start, tick };
}

function _completeScanToast(tick, resultText, isError) {
  clearInterval(tick);
  const toast   = document.getElementById("scan-toast");
  const spinner = toast.querySelector(".scan-spinner");
  const msg     = document.getElementById("scan-toast-msg");
  const timer   = document.getElementById("scan-toast-timer");
  spinner.style.display = "none";
  msg.textContent = resultText;
  timer.textContent = "";
  toast.classList.add(isError ? "scan-toast--error" : "scan-toast--done");
  setTimeout(() => toast.classList.add("hidden"), 3500);
}

async function runScan(market) {
  const btn = document.getElementById(`scan-${market}-btn`);
  btn.disabled = true;
  const { tick } = _showScanToast(market);
  try {
    const r    = await fetch(`/api/scan/${market}`, { method: "POST" });
    const data = await r.json();
    const candidates = data.candidates?.length ?? 0;
    const decisions  = data.decisions?.length ?? 0;
    const suffix = data.vix_halt ? " — VIX halt, no entries" : ` — ${candidates} candidate(s), ${decisions} decision(s)`;
    _completeScanToast(tick, `${market.toUpperCase()} scan done${suffix}`, false);
    refreshAll();
  } catch (e) {
    _completeScanToast(tick, `${market.toUpperCase()} scan failed — check server logs`, true);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("scan-nordic-btn").addEventListener("click", () => runScan("nordic"));
document.getElementById("scan-eu-btn").addEventListener("click", () => runScan("eu"));
document.getElementById("scan-us-btn").addEventListener("click", () => runScan("us"));
document.getElementById("scan-options-btn").addEventListener("click", () => runScan("options"));

document.getElementById("reset-btn").addEventListener("click", async () => {
  const pin = prompt("Enter PIN to reset all tracks:");
  if (pin === null) return;  // cancelled
  const r = await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pin }),
  });
  const data = await r.json();
  if (data.reset) {
    const summary = Object.entries(data.cleared)
      .map(([t, c]) => `${t}: ${c.heuristics_deleted} heuristics deleted`)
      .join("\n");
    alert("Simulation reset.\n" + summary);
    refreshAll();
  } else {
    alert(data.error || "Reset failed — check server logs.");
  }
});

async function refreshPrompts() {
  const data = await fetchJSON("/api/prompts");
  for (const track of TRACKS) {
    const el = document.getElementById(`prompts-${track}`);
    if (!el) continue;
    const td = data?.[track];
    if (!td) { el.innerHTML = "<p class='neutral'>Failed to load.</p>"; continue; }

    const current = td.current;
    const history = td.history || [];
    let html = "";

    if (current) {
      const demosNote = current.demos_count > 0
        ? ` · ${current.demos_count} few-shot demo${current.demos_count !== 1 ? "s" : ""}` : "";
      html += `<div class="prompt-version-header">
        <span class="prompt-version-label">Current · ${current.timestamp}${demosNote}</span>
        <span class="prompt-version-meta">${history.length} prior version${history.length !== 1 ? "s" : ""}</span>
      </div>
      <pre class="prompt-text">${esc(current.instructions || "(instructions not found in compiled file)")}</pre>`;
    } else {
      html += `<div class="prompt-version-header">
        <span class="prompt-version-label baseline">Baseline — not yet optimized by MIPRO</span>
        <span class="prompt-version-meta">Needs 30+ closed trades to run</span>
      </div>
      <pre class="prompt-text">${esc(td.baseline || "")}</pre>`;
    }

    if (history.length > 0) {
      html += `<div class="prompt-history-label">Version History</div>`;
      for (const v of history) {
        const demosNote = v.demos_count > 0 ? ` · ${v.demos_count} demo${v.demos_count !== 1 ? "s" : ""}` : "";
        html += `<details class="prompt-history-item">
          <summary>
            <span>${v.timestamp}${demosNote}</span>
            <span class="prompt-demos-badge">${v.filename}</span>
          </summary>
          <pre class="prompt-text prompt-text-history">${esc(v.instructions || "(no instructions)")}</pre>
        </details>`;
      }
    }

    el.innerHTML = html;
  }
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Decision history filters
document.getElementById("hist-apply")?.addEventListener("click", refreshHistory);
document.getElementById("hist-track")?.addEventListener("change", refreshHistory);
document.getElementById("hist-action")?.addEventListener("change", refreshHistory);
document.getElementById("hist-ticker")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") refreshHistory();
});

// Initial load + refresh every 60s
refreshAll();
setInterval(refreshAll, 60000);
