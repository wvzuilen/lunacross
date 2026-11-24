# LunaCross

A Python tool that computes the exact apparent position of the Moon from your location (WGS-84) and monitors live ADS-B aircraft from a local receiver (`readsb` / `tar1090` / `ultrafeeder`) to detect and predict **lunar transits** (aircraft crossing in front of the lunar disk).

## Features

- **High-precision lunar position (`skyfield` + JPL `de421.bsp`)**: topocentric altitude (including atmospheric refraction), azimuth + compass direction, distance in km, exact apparent angular diameter, illumination percentage, lunar phase, and next moonrise/moonset.
- **Local ADS-B receiver integration**: reads live `/data/aircraft.json` from a local `tar1090`/`readsb` feed and can optionally sync coordinates from `/data/receiver.json` (`--sync-receiver`).
- **3D WGS-84 line-of-sight geometry**: transforms aircraft positions (`lat`, `lon`, `alt_geom` / `alt_baro`) via ECEF → local ENU into exact apparent viewing angles (azimuth & altitude) with altitude-scaled refraction and `seen_pos` dead-reckoning.
- **Lookahead trajectory prediction**: projects flight paths up to 120 seconds ahead along a great circle to warn of upcoming crossings (`INCOMING`) before the aircraft crosses the Moon.
- **Alerts & logging**: live terminal dashboard (`rich`), JSONL event log (`transits.jsonl`), Pushover push notifications (including a direct link to the aircraft in `tar1090`), macOS Notification Center alerts, and optional webhook support (e.g. for `n8n` or Home Assistant).

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

Receive instant Pushover alerts when an aircraft is on course to cross the Moon (`INCOMING`, up to 120s ahead) and/or while it is crossing (`TRANSIT`), complete with a clickable link to that aircraft on your local `tar1090` radar.

1. Set `pushover_app_token` and `pushover_user_key` in `config.toml` (or via `LUNACROSS_PUSHOVER_APP_TOKEN` and `LUNACROSS_PUSHOVER_USER_KEY`).
2. Test delivery:
   ```bash
   uv run lunacross test-pushover
   ```
3. Or send simulated transit alerts to your device:
   ```bash
   uv run lunacross simulate --pushover
   ```

## Docker (24/7 Server Deployment)

1. Create `config.toml` from `config.example.toml` and set the container paths:
   ```toml
   log_file = "/data/transits.jsonl"
   ephemeris_dir = "/data/skyfield"
   notify_macos = false
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
