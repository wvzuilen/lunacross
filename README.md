# LunaCross

Een Python-tool die vanaf jouw exacte locatie (WGS-84) de schijnbare positie van de maan berekent en live vliegtuigen uit een lokale ADS-B ontvanger (`readsb` / `tar1090` / `ultrafeeder`) volgt om **maanovergangen** (vliegtuigen die voor de maanschijf langs vliegen) te detecteren en vooraf te voorspellen.

## Functies

- **Nauwkeurige maanpositie (`skyfield` + JPL `de421.bsp`)**: topocentrische hoogte (incl. atmosferische refractie), azimut + windstreek, afstand in km, exacte schijnbare hoekdiameter, verlichtingspercentage, maanfase en eerstvolgende opkomst/ondergang.
- **Koppeling met je eigen ADS-B ontvanger**: leest live `/data/aircraft.json` van een lokale `tar1090`/`readsb` feed en kan desgewenst coördinaten synchroniseren via `/data/receiver.json` (`--sync-receiver`).
- **3D WGS-84 zichtlijngeometrie**: zet vliegtuigposities (`lat`, `lon`, `alt_geom` / `alt_baro`) om via ECEF → lokaal ENU naar exacte kijkhoeken (azimut & hoogte) met hoogte-afhankelijke refractie en dead-reckoning voor `seen_pos`.
- **Vooruitkijkende trajectvoorspelling**: projecteert vliegroutes tot 120 seconden vooruit langs een grootcirkel om naderende overgangen (`INCOMING`) aan te kondigen vóór het vliegtuig voor de maan schuift.
- **Meldingen & logging**: live terminal-dashboard (`rich`), JSONL-log (`transits.jsonl`), Pushover-pushmeldingen (incl. directe link naar `tar1090`), macOS Notification Center meldingen en optionele webhook (bijv. voor `n8n` of Home Assistant).

## Installatie

```bash
git clone https://github.com/wvzuilen/lunacross.git
cd lunacross
uv sync
```

## Configuratie

Kopieer `config.example.toml` naar `config.toml` (genegeerd door Git) en stel je eigen locatie en ADS-B URL in:

```bash
cp config.example.toml config.toml
```

Je kunt instellingen aanpassen via:
1. `config.toml` (wordt automatisch ingelezen uit de huidige map, of via `--config`),
2. Omgevingsvariabelen (`LUNACROSS_LATITUDE`, `LUNACROSS_LONGITUDE`, `LUNACROSS_ADSB_URL`, `LUNACROSS_WEBHOOK_URL`, ...),
3. CLI-flags (`--lat`, `--lon`, `--elevation`, `--url`, `--margin`, `--horizon`, `--sync-receiver`).

## Gebruik

```bash
# 1. Alleen de actuele maanpositie bekijken
uv run lunacross moon

# 2. Eenmalige check: maanpositie + live vliegtuigen
uv run lunacross check

# 3. Live monitor-dashboard starten (pollt elke seconde)
uv run lunacross monitor

# 4. Simulatie draaien om voorspelling & transitdetectie te testen
uv run lunacross simulate
```

Handige opties (geldig vóór het subcommando):

```bash
uv run lunacross --sync-receiver check
uv run lunacross --margin 0.8 --horizon 180 monitor
uv run lunacross --config config.toml monitor
```

## Pushover notificaties

Je kunt direct een Pushover-melding ontvangen zodra een vliegtuig op koers ligt om voor de maan langs te vliegen (`INCOMING`, tot 120s vooraf) en/of tijdens de daadwerkelijke overgang (`TRANSIT`), inclusief een directe link naar dat toestel in `tar1090`.

1. Zet je `pushover_app_token` en `pushover_user_key` in `config.toml` (of gebruik `LUNACROSS_PUSHOVER_APP_TOKEN` en `LUNACROSS_PUSHOVER_USER_KEY`).
2. Test de koppeling:
   ```bash
   uv run lunacross test-pushover
   ```
3. Of stuur een gesimuleerde transitmelding naar je telefoon:
   ```bash
   uv run lunacross simulate --pushover
   ```

## Tests draaien

```bash
uv run pytest -v
```
