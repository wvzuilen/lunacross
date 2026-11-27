"""Built-in lightweight HTTP server for Live Sky View and Recorded Pass Replay."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from lunacross.config import Config
from lunacross.detector import AircraftSighting, PassPoint, PassRecord
from lunacross.moon import MoonPosition
from lunacross.notifier import load_passes
from lunacross.visualize import render_live_snapshot_svg, render_pass_svg

_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LunaCross — Live Sky &amp; Pass Replay</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 20px; background: #020617; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; }
  .container { max-width: 1200px; margin: 0 auto; }
  header { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  h1 { margin: 0; font-size: 20px; }
  .tabs { display: flex; gap: 8px; }
  .tab { background: #0f172a; color: #94a3b8; border: 1px solid #334155; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 600; }
  .tab.active { background: #1e293b; color: #f8fafc; border-color: #38bdf8; }
  .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; margin-bottom: 16px; }
  .meta { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: #94a3b8; margin-bottom: 10px; font-family: ui-monospace, monospace; }
  .meta strong { color: #f8fafc; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1e293b; }
  th { color: #94a3b8; font-weight: 600; }
  tr.pass-row { cursor: pointer; transition: background-color 0.12s ease; }
  tr.pass-row:hover { background: #1e293b; }
  tr.pass-row.selected { background: #172554; }
  tr.pass-row.selected:hover { background: #1e3a8a; }
  tr.pass-row.selected td:first-child { box-shadow: inset 3px 0 0 #38bdf8; }
  tr.pass-row.selected td { color: #f8fafc; }
  .passes-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
  .filter-group { display: flex; gap: 6px; flex-wrap: wrap; }
  .filter-btn { background: #0f172a; color: #94a3b8; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; }
  .filter-btn:hover { background: #1e293b; color: #e2e8f0; }
  .filter-btn.active { background: #1e293b; color: #f8fafc; border-color: #38bdf8; }
  .pagination { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 12px; font-size: 12px; color: #94a3b8; }
  .pagination-controls { display: flex; align-items: center; gap: 6px; }
  .page-btn { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 5px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; }
  .page-btn:hover:not(:disabled) { background: #334155; }
  .page-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .page-select { background: #0f172a; color: #e2e8f0; border: 1px solid #334155; padding: 4px 8px; border-radius: 6px; font-size: 12px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
  .badge-TRANSIT { background: #7f1d1d; color: #fecaca; }
  .badge-NEAR_MISS { background: #78350f; color: #fde68a; }
  .badge-CLOSE_PASS { background: #0c4a6e; color: #bae6fd; }
  .actions { display: flex; gap: 8px; margin-top: 8px; }
  a.btn, button.btn { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 12px; cursor: pointer; }
  a.btn:hover, button.btn:hover { background: #334155; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>LunaCross — Observer Sky Monitor</h1>
    <div class="tabs">
      <button id="tab-live" class="tab active" onclick="showTab('live')">Live Sky View</button>
      <button id="tab-passes" class="tab" onclick="showTab('passes')">Recorded Passes</button>
    </div>
  </header>

  <section id="panel-live">
    <div class="card">
      <div class="meta" id="live-meta">Loading live sky state...</div>
      <div id="live-svg"></div>
      <div class="actions">
        <a class="btn" href="/api/live.svg" download="lunacross_live.svg">Download Live SVG</a>
        <a class="btn" href="/api/live.json" target="_blank">Live JSON</a>
      </div>
    </div>
  </section>

  <section id="panel-passes" class="hidden">
    <div class="card">
      <div class="passes-toolbar">
        <div class="meta" style="margin-bottom:0;">
          <span>Recorded close passes from <strong>passes.jsonl</strong> (newest first)</span>
        </div>
        <div class="filter-group" id="pass-filters">
          <button class="filter-btn active" data-filter="ALL" onclick="setPassFilter('ALL')">All</button>
          <button class="filter-btn" data-filter="TRANSIT" onclick="setPassFilter('TRANSIT')">Transit</button>
          <button class="filter-btn" data-filter="NEAR_MISS" onclick="setPassFilter('NEAR_MISS')">Near miss</button>
          <button class="filter-btn" data-filter="CLOSE_PASS" onclick="setPassFilter('CLOSE_PASS')">Close pass</button>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Closest (Local)</th><th>Flight</th><th>Class</th><th>Min Sep</th><th>Alt (ft)</th><th>Track pts</th></tr>
        </thead>
        <tbody id="passes-body"><tr><td colspan="6">Loading...</td></tr></tbody>
      </table>
      <div class="pagination" id="passes-pagination">
        <span id="passes-summary"></span>
        <div class="pagination-controls">
          <label for="page-size">Per page:</label>
          <select id="page-size" class="page-select" onchange="setPageSize(this.value)">
            <option value="15" selected>15</option>
            <option value="25">25</option>
            <option value="50">50</option>
          </select>
          <button id="btn-first" class="page-btn" onclick="goToPage(1)">First</button>
          <button id="btn-prev" class="page-btn" onclick="goToPage(currentPage - 1)">Prev</button>
          <span id="page-indicator" style="min-width:84px;text-align:center;">Page 1 / 1</span>
          <button id="btn-next" class="page-btn" onclick="goToPage(currentPage + 1)">Next</button>
          <button id="btn-last" class="page-btn" onclick="goToPage(totalPages())">Last</button>
        </div>
      </div>
    </div>
    <div id="pass-viewer" class="card hidden">
      <div id="pass-svg"></div>
      <div class="actions">
        <a id="pass-dl-svg" class="btn" href="#" download>Download Pass SVG</a>
        <a id="pass-dl-json" class="btn" href="#" target="_blank">Pass JSON</a>
      </div>
    </div>
  </section>
</div>

<script>
let currentTab = 'live';
function showTab(name) {
  currentTab = name;
  document.getElementById('panel-live').classList.toggle('hidden', name !== 'live');
  document.getElementById('panel-passes').classList.toggle('hidden', name !== 'passes');
  document.getElementById('tab-live').classList.toggle('active', name === 'live');
  document.getElementById('tab-passes').classList.toggle('active', name === 'passes');
  if (name === 'passes') loadPasses();
}

async function refreshLive() {
  if (currentTab !== 'live') return;
  try {
    const [jRes, sRes] = await Promise.all([
      fetch('/api/live.json?_=' + Date.now()),
      fetch('/api/live.svg?_=' + Date.now()),
    ]);
    if (jRes.ok) {
      const data = await jRes.json();
      const m = data.moon;
      const meta = document.getElementById('live-meta');
      if (m) {
        meta.innerHTML =
          `<span>Moon: <strong>${m.altitude_deg.toFixed(1)}° alt / ${m.azimuth_deg.toFixed(1)}° az</strong> (${m.phase_name}, ${m.illumination_pct.toFixed(1)}%)</span>` +
          `<span>Feed: <strong>${data.feed_ok ? 'Live connected' : 'Disconnected'}</strong></span>` +
          `<span>Airborne: <strong>${data.sightings.length}/${data.total_raw}</strong></span>`;
      }
    }
    if (sRes.ok) {
      document.getElementById('live-svg').innerHTML = await sRes.text();
    }
  } catch (e) {}
}

let allPasses = [];
let passFilter = 'ALL';
let pageSize = 15;
let currentPage = 1;
let selectedPassId = null;

function filteredPasses() {
  if (passFilter === 'ALL') return allPasses;
  return allPasses.filter(p => p.classification === passFilter);
}

function totalPages() {
  return Math.max(1, Math.ceil(filteredPasses().length / pageSize));
}

function setPassFilter(filter) {
  passFilter = filter;
  currentPage = 1;
  document.querySelectorAll('#pass-filters .filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === filter);
  });
  renderPassesPage();
}

function setPageSize(value) {
  pageSize = Math.max(1, parseInt(value, 10) || 15);
  currentPage = 1;
  renderPassesPage();
}

function goToPage(page) {
  const maxP = totalPages();
  currentPage = Math.min(maxP, Math.max(1, page));
  renderPassesPage();
}

function renderPassesPage() {
  const items = filteredPasses();
  const tbody = document.getElementById('passes-body');
  const summary = document.getElementById('passes-summary');
  const maxP = totalPages();
  if (currentPage > maxP) currentPage = maxP;

  document.getElementById('page-indicator').textContent = `Page ${currentPage} / ${maxP}`;
  document.getElementById('btn-first').disabled = currentPage <= 1;
  document.getElementById('btn-prev').disabled = currentPage <= 1;
  document.getElementById('btn-next').disabled = currentPage >= maxP;
  document.getElementById('btn-last').disabled = currentPage >= maxP;

  if (!items.length) {
    summary.textContent = 'Showing 0 passes';
    tbody.innerHTML = '<tr><td colspan="6">No matching passes found.</td></tr>';
    return;
  }

  const startIdx = (currentPage - 1) * pageSize;
  const pageItems = items.slice(startIdx, startIdx + pageSize);
  summary.textContent = `Showing ${startIdx + 1}–${startIdx + pageItems.length} of ${items.length} passes`;

  tbody.innerHTML = pageItems.map(p => {
    const dt = new Date(p.closest_utc).toLocaleString();
    const selClass = p.pass_id === selectedPassId ? ' selected' : '';
    return `<tr class="pass-row${selClass}" data-pass-id="${p.pass_id}" onclick="viewPass('${p.pass_id}')">` +
      `<td>${dt}</td>` +
      `<td><strong>${p.flight}</strong></td>` +
      `<td><span class="badge badge-${p.classification}">${p.classification}</span></td>` +
      `<td>${p.min_angular_sep_deg.toFixed(3)}° (${p.min_moon_radii.toFixed(2)}R)</td>` +
      `<td>${Math.round(p.closest_altitude_ft).toLocaleString()}</td>` +
      `<td>${p.track_points}</td>` +
      `</tr>`;
  }).join('');
}

async function loadPasses() {
  const res = await fetch('/api/passes?_=' + Date.now());
  allPasses = await res.json();
  renderPassesPage();
}

async function viewPass(passId) {
  selectedPassId = passId;
  document.querySelectorAll('#passes-body tr.pass-row').forEach(tr => {
    tr.classList.toggle('selected', tr.dataset.passId === passId);
  });
  const res = await fetch('/api/passes/' + encodeURIComponent(passId) + '.svg');
  if (!res.ok) return;
  document.getElementById('pass-svg').innerHTML = await res.text();
  document.getElementById('pass-dl-svg').href = '/api/passes/' + encodeURIComponent(passId) + '.svg';
  document.getElementById('pass-dl-svg').download = passId + '.svg';
  document.getElementById('pass-dl-json').href = '/api/passes/' + encodeURIComponent(passId) + '.json';
  const viewer = document.getElementById('pass-viewer');
  viewer.classList.remove('hidden');
  viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

refreshLive();
setInterval(refreshLive, 2000);
</script>
</body>
</html>
"""


class LiveState:
    """Thread-safe container for the latest monitor poll state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.moon: MoonPosition | None = None
        self.sightings: list[AircraftSighting] = []
        self.tracks: dict[str, list[PassPoint]] = {}
        self.total_raw: int = 0
        self.feed_ok: bool = False
        self.updated_utc: datetime | None = None

    def update(
        self,
        *,
        moon: MoonPosition,
        sightings: list[AircraftSighting],
        tracks: dict[str, list[PassPoint]],
        total_raw: int,
        feed_ok: bool,
    ) -> None:
        with self._lock:
            self.moon = moon
            self.sightings = list(sightings)
            self.tracks = {k: list(v) for k, v in tracks.items()}
            self.total_raw = total_raw
            self.feed_ok = feed_ok
            self.updated_utc = datetime.now(timezone.utc)

    def snapshot(
        self,
    ) -> tuple[MoonPosition | None, list[AircraftSighting], dict[str, list[PassPoint]], int, bool, datetime | None]:
        with self._lock:
            return (
                self.moon,
                list(self.sightings),
                {k: list(v) for k, v in self.tracks.items()},
                self.total_raw,
                self.feed_ok,
                self.updated_utc,
            )


class LunaCrossWebServer:
    """Embedded HTTP server serving the Live Sky View and Recorded Pass Replay UI."""

    def __init__(self, host: str, port: int, cfg: Config, live_state: LiveState) -> None:
        self.host = host
        self.cfg = cfg
        self.live_state = live_state

        def handler_factory(*args: Any, **kwargs: Any) -> _Handler:
            return _Handler(*args, cfg=cfg, live_state=live_state, **kwargs)

        self._httpd = ThreadingHTTPServer((host, port), handler_factory)
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, cfg: Config, live_state: LiveState, **kwargs: Any) -> None:
        self.cfg = cfg
        self.live_state = live_state
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _find_pass(self, pass_id: str) -> PassRecord | None:
        for rec in load_passes(self.cfg.passes_file):
            if rec.pass_id == pass_id:
                return rec
        return None

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path

        if path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", _INDEX_HTML.encode("utf-8"))
            return

        if path == "/api/live.json":
            moon, sightings, _, total_raw, feed_ok, updated = self.live_state.snapshot()
            payload: dict[str, Any] = {
                "updated_utc": updated.isoformat() if updated else None,
                "feed_ok": feed_ok,
                "total_raw": total_raw,
                "moon": (
                    {
                        "altitude_deg": round(moon.altitude_deg, 3),
                        "azimuth_deg": round(moon.azimuth_deg, 3),
                        "angular_diameter_deg": round(moon.angular_diameter_deg, 4),
                        "illumination_pct": round(moon.illumination_pct, 1),
                        "phase_name": moon.phase_name,
                        "is_above_horizon": moon.is_above_horizon,
                    }
                    if moon
                    else None
                ),
                "sightings": [
                    {
                        "flight": s.aircraft.flight,
                        "hex": s.aircraft.hex,
                        "altitude_ft": round(s.aircraft.altitude_ft),
                        "slant_range_km": round(s.sky_pos.slant_range_km, 2),
                        "aircraft_alt_deg": round(s.sky_pos.altitude_deg, 2),
                        "aircraft_az_deg": round(s.sky_pos.azimuth_deg, 2),
                        "angular_sep_deg": round(s.angular_sep_deg, 3),
                        "moon_radii": round(s.moon_radii, 2),
                        "status": s.status,
                    }
                    for s in sightings
                ],
            }
            self._send(HTTPStatus.OK, "application/json", json.dumps(payload).encode("utf-8"))
            return

        if path == "/api/live.svg":
            moon, sightings, tracks, _, _, _ = self.live_state.snapshot()
            if moon is None:
                svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 120"><rect width="600" height="120" fill="#020617"/><text x="20" y="65" fill="#94a3b8" font-family="sans-serif">Waiting for first monitor sample...</text></svg>'
            else:
                svg = render_live_snapshot_svg(moon, sightings, tracks, self.cfg)
            self._send(HTTPStatus.OK, "image/svg+xml; charset=utf-8", svg.encode("utf-8"))
            return

        if path == "/api/passes":
            passes = list(reversed(load_passes(self.cfg.passes_file)))
            summaries = [
                {
                    "pass_id": p.pass_id,
                    "flight": p.flight,
                    "hex": p.hex,
                    "classification": p.classification,
                    "crossed_lunar_disk": p.crossed_lunar_disk,
                    "closest_utc": p.closest_utc,
                    "min_angular_sep_deg": p.min_angular_sep_deg,
                    "min_moon_radii": p.min_moon_radii,
                    "closest_altitude_ft": p.closest_altitude_ft,
                    "track_points": len(p.track),
                }
                for p in passes
            ]
            self._send(HTTPStatus.OK, "application/json", json.dumps(summaries).encode("utf-8"))
            return

        if path.startswith("/api/passes/") and path.endswith(".svg"):
            pass_id = path[len("/api/passes/") : -len(".svg")]
            rec = self._find_pass(pass_id)
            if rec is None:
                self._send(HTTPStatus.NOT_FOUND, "application/json", b'{"error":"Pass not found"}')
                return
            self._send(HTTPStatus.OK, "image/svg+xml; charset=utf-8", render_pass_svg(rec).encode("utf-8"))
            return

        if path.startswith("/api/passes/") and path.endswith(".json"):
            pass_id = path[len("/api/passes/") : -len(".json")]
            rec = self._find_pass(pass_id)
            if rec is None:
                self._send(HTTPStatus.NOT_FOUND, "application/json", b'{"error":"Pass not found"}')
                return
            self._send(HTTPStatus.OK, "application/json", json.dumps(rec.to_dict()).encode("utf-8"))
            return

        self._send(HTTPStatus.NOT_FOUND, "application/json", b'{"error":"Not found"}')
