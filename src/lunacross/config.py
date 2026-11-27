"""Configuration management for LunaCross."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Config:
    """Settings for observer location, ADS-B feed, transit detection, and notifications."""

    latitude: float = 52.3731
    longitude: float = 4.8926
    elevation_m: float = 2.0
    geoid_undulation_m: float = 0.0
    adsb_url: str = "http://localhost:8090/data/aircraft.json"
    supplement_url: str | None = None
    supplement_radius_nm: float = 45.0
    supplement_interval_s: float = 5.0
    weather_enabled: bool = True
    weather_interval_s: float = 600.0
    poll_interval_s: float = 1.0
    alert_margin_deg: float = 0.5
    log_margin_deg: float = 5.0
    predict_horizon_s: float = 120.0
    min_moon_alt_deg: float = 0.0
    max_seen_pos_s: float = 30.0
    log_file: str = "transits.jsonl"
    passes_file: str = "passes.jsonl"
    close_pass_retention_days: int = 7
    notify_macos: bool = True
    webhook_url: str | None = None
    pushover_app_token: str | None = None
    pushover_user_key: str | None = None
    pushover_device: str | None = None
    pushover_sound: str | None = None
    pushover_priority: int = 1
    pushover_events: str = "INCOMING,TRANSIT"
    pushover_max_lead_s: float = 60.0
    pushover_margin_deg: float = 0.25
    pushover_min_angular_arcmin: float = 0.0
    pushover_max_cloud_cover_pct: float = 85.0
    pushover_quiet_hours: str | None = None
    ephemeris_dir: str = ".cache/skyfield"
    top_n: int = 12
    web_port: int = 0


def _parse_env_value(raw: str, current: Any) -> Any:
    """Convert an environment variable string to the target field type."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:
        stripped = raw.strip()
        return stripped if stripped else None
    return raw


def load_config(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration with precedence: defaults < config.toml < env < CLI."""
    cfg = Config()
    valid_fields = {f.name: f for f in fields(Config)}

    # 1. TOML file (explicit path or ./config.toml if present)
    resolved_path = config_path
    if resolved_path is None:
        default_toml = Path("config.toml")
        if default_toml.is_file():
            resolved_path = default_toml

    if resolved_path is not None:
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
        with resolved_path.open("rb") as fh:
            raw_toml = tomllib.load(fh)
        for key, value in raw_toml.items():
            if key in valid_fields:
                current = getattr(cfg, key)
                if isinstance(current, bool):
                    setattr(cfg, key, bool(value))
                elif isinstance(current, float):
                    setattr(cfg, key, float(value))
                elif isinstance(current, int):
                    setattr(cfg, key, int(value))
                else:
                    setattr(cfg, key, value)

    # 2. Environment variables (LUNACROSS_<FIELD_UPPER>)
    for name in valid_fields:
        env_key = f"LUNACROSS_{name.upper()}"
        if env_key in os.environ:
            current = getattr(cfg, name)
            setattr(cfg, name, _parse_env_value(os.environ[env_key], current))

    # 3. CLI overrides (ignore None)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and key in valid_fields:
                setattr(cfg, key, value)

    return cfg
