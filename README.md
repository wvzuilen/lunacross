# LunaCross

A Python tool that computes the exact apparent position of the Moon from your location (WGS-84) and monitors live ADS-B aircraft from a local receiver (`readsb` / `tar1090` / `ultrafeeder`) to detect and predict **lunar transits** (aircraft crossing in front of the lunar disk).

## Features

- **High-precision lunar position (`skyfield` + JPL `de421.bsp`)**: topocentric altitude (including atmospheric refraction), azimuth + compass direction, distance in km, exact apparent angular diameter, illumination percentage, lunar phase, and next moonrise/moonset.
- **Local ADS-B receiver integration**: reads live `/data/aircraft.json` from a local `tar1090`/`readsb` feed and can optionally sync coordinates from `/data/receiver.json` (`--sync-receiver`), plus an optional **Hybrid ADS-B mode** combining your local receiver with a free community radius API (`adsb.lol`) for extended coverage and aircraft type/registration enrichment.
- **3D WGS-84 line-of-sight geometry**: transforms aircraft positions (`lat`, `lon`, `alt_geom` HAE / QNH- & geoid-corrected `alt_baro`) via ECEF → local ENU into exact apparent viewing angles (azimuth & altitude) with live temperature/pressure-aware atmospheric refraction and `seen_pos` dead-reckoning.
- **Lookahead trajectory prediction & sub-second detection**: projects flight paths ahead on the WGS-84 ellipsoid (including turn rate and real-time Moon drift velocity) to warn of upcoming crossings (`INCOMING`) while interpolating between 1-second ADS-B polls so fast aircraft never slip across the lunar disk undetected.
- **Live weather & cloud-cover awareness (`METAR` ceilometers + `Open-Meteo ICON`)**: periodically fetches real cloud layers, visibility, temperature, and QNH from the nearest airport laser ceilometer (`aviationweather.gov` METAR, falling back to `Open-Meteo` `icon_seamless` with no API key required) to feed exact atmospheric refraction and fallback QNH, display live sky clarity in the UI, and automatically suppress Pushover alerts during overcast skies or configured `pushover_quiet_hours`.
- **Complete close-pass track recording**: every aircraft entering `log_margin_deg` (default `2.0°`) is recorded with its full topocentric sky track (`passes.jsonl`), true lunar phase angle, and automatic retention pruning for older `CLOSE_PASS` entries.
- **Observer-perspective visualization & Web UI**: generates dark-mode SVG/HTML charts showing your horizon, compass bearings, Moon phase, and aircraft trajectory (both real-time and post-hoc replay with pagination, filters, and mtime-cached pass lookup), served via an optional built-in Web UI (`web_port = 8091`) or `lunacross render`.
- **Alerts & logging**: live terminal dashboard (`rich`), JSONL logs (`transits.jsonl` & `passes.jsonl`), Pushover push notifications (including a direct link to `tar1090`), macOS Notification Center alerts, and optional webhook support.

## UI Preview

### Live Sky View
*Example using demo coordinates (`52.3731° N, 4.8926° E`): 4-card live status bar (Moon, METAR sky/weather, ADS-B feed, closest aircraft), real-time observer horizon with realistic topocentric Moon phase & maria preview, `+120s` dashed flight path predictions with direction arrows, and the live Closest Aircraft table.*

![LunaCross Live Sky View](docs/ui-live-sky.png)

### Pass Replay
*Recorded Passes tab with pagination, `Transit + Near miss` default filter, highlighted row selection, and post-hoc Pass Replay of a simulated lunar transit (`KLM999`).*

![LunaCross Pass Replay](docs/ui-pass-replay.png)

## Installation

```bash
git clone https://github.com/wvzuilen/lunacross.git
cd lunacross
uv sync
```

## Configuration

Copy `config.example.toml` to `config.toml` (gitignored) and set your own observer location and ADS-B feed URL:

```bash
cp config.example.toml config.toml
```

Settings are resolved in the following order (lowest to highest priority):
1. Built-in defaults,
2. `config.toml` (auto-loaded from the current working directory, or specified via `--config`),
3. Environment variables (`LUNACROSS_LATITUDE`, `LUNACROSS_LONGITUDE`, `LUNACROSS_ADSB_URL`, `LUNACROSS_WEBHOOK_URL`, ...),
4. CLI flags (`--lat`, `--lon`, `--elevation`, `--url`, `--margin`, `--horizon`, `--sync-receiver`).

### Hybrid ADS-B Mode (Optional)
If your indoor antenna only picks up nearby aircraft, you can combine your local `readsb`/`tar1090` feed with the free community `adsb.lol` radius API (`supplement_url`) to extend coverage and enrich local aircraft with registration and aircraft type metadata:

```toml
supplement_url = "https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{dist_nm}"
supplement_radius_nm = 45.0
supplement_interval_s = 5.0
```

## Usage

```bash
# 1. Show the current Moon position for your location
uv run lunacross moon

# 2. One-shot check: Moon position + live aircraft
uv run lunacross check

# 3. Start the live monitoring dashboard (polls every second)
uv run lunacross monitor

# 4. Run a simulation to verify prediction & transit detection
uv run lunacross simulate
```

Useful global options (placed before the subcommand):

```bash
uv run lunacross --sync-receiver check
uv run lunacross --margin 0.8 --horizon 180 monitor
uv run lunacross --config config.toml monitor
```

## Pushover Notifications

Receive instant Pushover alerts when an aircraft is on course to cross the Moon (`INCOMING`, configurable lead time via `pushover_max_lead_s`) and/or while it is crossing (`TRANSIT`), complete with estimated visual visibility (`Naked eye`, `Binoculars / zoom`, or `Telescope / telephoto only`), live sky conditions, and a clickable link to that aircraft on your local `tar1090` radar.

- **Cloud-cover & quiet-hours filtering**: set `pushover_max_cloud_cover_pct` (default `85.0`) to suppress Pushover alerts under overcast skies or fog, and optionally `pushover_quiet_hours = "23:30-07:00"` for overnight quiet hours (all events are still recorded in `transits.jsonl` and `passes.jsonl`).
- **Optional size filter**: set `pushover_min_angular_arcmin` (e.g. `2.0`) to only alert on larger or closer aircraft.

1. Set `pushover_app_token` and `pushover_user_key` in `config.toml` (or via `LUNACROSS_PUSHOVER_APP_TOKEN` and `LUNACROSS_PUSHOVER_USER_KEY`).
2. Test delivery:
   ```bash
   uv run lunacross test-pushover
   ```
3. Or send simulated transit alerts to your device:
   ```bash
   uv run lunacross simulate --pushover
   ```

## Sky Visualization & Web UI

Generate an observer-perspective sky chart (horizon, compass ticks, topocentric Moon phase with lunar maria, aircraft path, and zoomed lunar inset) directly from the CLI:

```bash
# List all recorded close passes
uv run lunacross render --list

# Render the latest recorded pass (or pass --pass-id <id>) to an interactive HTML or SVG file
uv run lunacross render -o pass.html

# Render the current live sky + predicted flight paths
uv run lunacross render --live -o live_sky.html

# Simulate a lunar transit and render its sky chart
uv run lunacross simulate --render sim_transit.html
```

To enable the built-in real-time Web UI (with auto-refreshing **Live Sky View**, live **Closest Aircraft** table, and paginated/filterable **Recorded Passes** replay), set `web_port = 8091` in `config.toml` (or pass `--web-port 8091`), then open `http://<server-ip>:8091`. Older `CLOSE_PASS` records are pruned automatically according to `close_pass_retention_days` (default `7` days), while `TRANSIT` and `NEAR_MISS` records are kept permanently.

## Docker (24/7 Server Deployment)

1. Create `config.toml` from `config.example.toml` and set the container paths:
   ```toml
   log_file = "/data/transits.jsonl"
   passes_file = "/data/passes.jsonl"
   ephemeris_dir = "/data/skyfield"
   notify_macos = false
   web_port = 8091
   ```
2. Build and start the container (`restart: unless-stopped`):
   ```bash
   docker compose up -d --build
   ```
3. Follow the live logs:
   ```bash
   docker logs -f lunacross
   ```

## Running Tests

```bash
uv run pytest -v
```
