// Reads the aggregate files written by `python -m reviews_pipeline.gold`.
// The names below are checked against the gold output by tests/test_site.py.
// fields: meta.generated_at meta.last_success_at meta.newest_review_at meta.excluded_future_dates
// fields: meta.coverage meta.drift
// fields: row.date row.n row.mean row.median row.low row.mid row.high row.repeated row.complete
// fields: hour.hour hour.n

const STALE_AFTER_HOURS = 3;
const HOURS_SHOWN = 7 * 24;
const PARTIAL_ALPHA = 0.4;
const MIN_REVIEWS_FOR_SHARES = 30; // a share of a handful of reviews is noise, so it is not drawn

const $ = (id) => document.getElementById(id);
let data = null;
let charts = [];

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function withAlpha(color, alpha) {
  const hex = color.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function fmtTime(iso) {
  return iso.replace("T", " ").replace(":00Z", " UTC").replace("Z", " UTC");
}

function ago(iso) {
  const hours = (Date.now() - Date.parse(iso)) / 36e5;
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))} min ago`;
  if (hours < 48) return `${hours.toFixed(1)} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}

// The stale notice is computed here, from the current time, so a stuck pipeline still shows.
function renderFreshness(meta) {
  const status = $("fresh-status");
  const detail = $("fresh-detail");
  if (!meta.last_success_at) {
    status.className = "stale";
    status.textContent = "No data collected yet.";
    detail.textContent = "";
    return;
  }
  const hours = (Date.now() - Date.parse(meta.last_success_at)) / 36e5;
  const stale = hours > STALE_AFTER_HOURS;
  status.className = stale ? "stale" : "ok";
  status.textContent = stale
    ? `Data is stale: the last successful run was ${ago(meta.last_success_at)}.`
    : `Data is fresh: the last successful run was ${ago(meta.last_success_at)}.`;
  const newest = meta.newest_review_at ? ` Newest review: ${fmtTime(meta.newest_review_at)}.` : "";
  detail.textContent = `Last success: ${fmtTime(meta.last_success_at)}.${newest}`;
  $("excluded").textContent = String(meta.excluded_future_dates);
}

function baseOptions(extra = {}) {
  const grid = css("--grid");
  const text = css("--text-2");
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { display: false }, ...extra.plugins },
    scales: {
      x: { grid: { display: false }, ticks: { color: text, maxRotation: 0, autoSkipPadding: 16 } },
      y: { grid: { color: grid }, border: { display: false }, ticks: { color: text },
           beginAtZero: true, ...extra.y },
    },
  };
}

function today(meta) {
  return meta.generated_at.slice(0, 10);
}

function dayLabel(row, meta) {
  if (row.date === today(meta)) return " (day in progress)";
  return row.complete ? "" : " (partial coverage)";
}

function renderDaily(rows, meta) {
  const series = css("--series");
  charts.push(new Chart($("c-daily"), {
    type: "bar",
    data: {
      labels: rows.map((r) => r.date.slice(5)),
      datasets: [{
        label: "Reviews",
        data: rows.map((r) => r.n),
        backgroundColor: rows.map((r) => (r.complete ? series : withAlpha(series, PARTIAL_ALPHA))),
        borderRadius: { topLeft: 4, topRight: 4 }, borderSkipped: "bottom", maxBarThickness: 24,
      }],
    },
    options: baseOptions({
      plugins: { tooltip: { callbacks: {
        title: (items) => rows[items[0].dataIndex].date + dayLabel(rows[items[0].dataIndex], meta),
        afterBody: (items) => {
          const row = rows[items[0].dataIndex];
          return `mean ${row.mean.toFixed(2)}, median ${row.median}`;
        },
      } } },
    }),
  }));
}

function renderBands(rows, meta) {
  const bands = [
    ["low", "Scored 0–1", "--low"], ["mid", "Scored 2–8", "--mid"], ["high", "Scored 9–10", "--high"],
  ];
  charts.push(new Chart($("c-bands"), {
    type: "bar",
    data: {
      labels: rows.map((r) => r.date.slice(5)),
      datasets: bands.map(([key, label, token]) => ({
        label,
        data: rows.map((r) => (r.n >= MIN_REVIEWS_FOR_SHARES ? (100 * r[key]) / r.n : null)),
        backgroundColor: rows.map((r) => (r.complete ? css(token) : withAlpha(css(token), PARTIAL_ALPHA))),
        borderColor: css("--surface"), borderWidth: 1, maxBarThickness: 24,
      })),
    },
    options: baseOptions({
      y: { stacked: true, max: 100, ticks: { color: css("--text-2"), callback: (v) => `${v}%` } },
      plugins: {
        legend: { display: true, labels: {
          color: css("--text-2"), boxWidth: 12, boxHeight: 12,
          generateLabels: () => bands.map(([, label, token], i) => ({
            text: label, fontColor: css("--text-2"), fillStyle: css(token),
            strokeStyle: css(token), datasetIndex: i,
          })),
        } },
        tooltip: { callbacks: {
          title: (items) => rows[items[0].dataIndex].date + dayLabel(rows[items[0].dataIndex], meta),
          label: (item) => (item.parsed.y == null
            ? `${item.dataset.label}: not drawn (fewer than ${MIN_REVIEWS_FOR_SHARES} reviews)`
            : `${item.dataset.label}: ${item.parsed.y.toFixed(1)}%`),
        } },
      },
    }),
  }));
  charts[charts.length - 1].options.scales.x.stacked = true;
  charts[charts.length - 1].update();
}

function renderHourly(hours) {
  const shown = hours.slice(-HOURS_SHOWN);
  const series = css("--series");
  charts.push(new Chart($("c-hourly"), {
    type: "line",
    data: {
      labels: shown.map((h) => h.hour.slice(5, 13).replace("T", " ") + "h"),
      datasets: [{
        label: "Reviews", data: shown.map((h) => h.n), borderColor: series, borderWidth: 2,
        backgroundColor: withAlpha(series, 0.1), fill: true, pointRadius: 0, pointHoverRadius: 4,
        tension: 0.2,
      }],
    },
    options: baseOptions(),
  }));
}

function renderRepeat(rows, meta) {
  const series = css("--series");
  charts.push(new Chart($("c-repeat"), {
    type: "line",
    data: {
      labels: rows.map((r) => r.date.slice(5)),
      datasets: [{
        label: "Repeated texts", data: rows.map((r) => (r.n >= MIN_REVIEWS_FOR_SHARES ? 100 * r.repeated : null)), borderColor: series,
        borderWidth: 2, pointRadius: 4, pointBackgroundColor: rows.map((r) => (r.complete ? series : css("--surface"))),
        pointBorderColor: series, pointBorderWidth: 2,
      }],
    },
    options: baseOptions({
      y: { ticks: { color: css("--text-2"), callback: (v) => `${v}%` } },
      plugins: { tooltip: { callbacks: {
        title: (items) => rows[items[0].dataIndex].date + dayLabel(rows[items[0].dataIndex], meta),
        label: (item) => (item.parsed.y == null
          ? `Not drawn (fewer than ${MIN_REVIEWS_FOR_SHARES} reviews)`
          : `Repeated: ${item.parsed.y.toFixed(2)}%`),
      } } },
    }),
  }));
}

function renderTable(rows) {
  const table = $("table");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const name of ["Date", "Reviews", "Mean", "Median", "0–1", "2–8", "9–10", "Repeated", "Coverage"]) {
    const th = document.createElement("th");
    th.textContent = name;
    head.appendChild(th);
  }
  const body = table.createTBody();
  for (const r of [...rows].reverse()) {
    const cells = [r.date, r.n, r.mean.toFixed(2), r.median, r.low, r.mid, r.high,
      `${(100 * r.repeated).toFixed(2)}%`, r.complete ? "complete" : "partial"];
    const tr = body.insertRow();
    for (const value of cells) tr.insertCell().textContent = String(value);
  }
}

function renderNotes(platform, rows, meta) {
  const drift = meta.drift[platform];
  const driftNote = $("drift");
  driftNote.hidden = !drift;
  if (drift) {
    const sign = drift.delta_pp > 0 ? "+" : "";
    const band = drift.metric === "low" ? "0–1" : "9–10";
    driftNote.textContent =
      `Score mix moved on ${drift.date}: the share of ${band} scores changed by ${sign}${drift.delta_pp} points ` +
      "against the days before. Informational only.";
  }
  const cover = meta.coverage[platform];
  // Only days before `complete_from` are a coverage gap; today is simply unfinished.
  const partial = cover.complete_from && rows.some((r) => r.date < cover.complete_from);
  const note = $("coverage");
  note.hidden = !partial;
  if (!note.hidden) {
    note.textContent =
      `Days before ${cover.complete_from} may be partial for this platform: each query returns at most the ` +
      "10,000 newest reviews, so older days can be undercounted and skewed. Those days are drawn lighter.";
  }
}

function render() {
  const platform = $("platform").value;
  const rows = data.daily.platforms[platform] || [];
  const hours = data.hourly.platforms[platform] || [];
  charts.forEach((c) => c.destroy());
  charts = [];
  const empty = rows.length === 0;
  $("empty").hidden = !empty;
  $("charts").hidden = empty;
  renderNotes(platform, rows, data.meta);
  if (empty) return;
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  renderDaily(rows, data.meta);
  renderBands(rows, data.meta);
  renderHourly(hours);
  renderRepeat(rows, data.meta);
  renderTable(rows);
}

async function load(name) {
  const response = await fetch(`data/${name}.json`, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${name}.json: ${response.status}`);
  return response.json();
}

async function main() {
  try {
    const [meta, daily, hourly] = await Promise.all([load("meta"), load("daily"), load("hourly")]);
    data = { meta, daily, hourly };
  } catch (error) {
    $("fresh-status").className = "stale";
    $("fresh-status").textContent = "Could not load the data files.";
    $("charts").hidden = true;
    console.error(error);
    return;
  }
  renderFreshness(data.meta);
  setInterval(() => renderFreshness(data.meta), 60_000);
  $("platform").addEventListener("change", render);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  render();
}

main();
