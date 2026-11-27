"""Local weather, cloud-cover obstruction, and atmospheric state via Open-Meteo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, value))


def compute_sky_obstruction_pct(
    cloud_cover_pct: float,
    cloud_low_pct: float | None = None,
    cloud_mid_pct: float | None = None,
    cloud_high_pct: float | None = None,
    visibility_km: float | None = None,
    weather_code: int | None = None,
) -> float:
    """Estimate optical sky obstruction (0-100%) for lunar viewing.

    Low and mid-level clouds are optically dense, whereas high-altitude cirrus
    is typically translucent enough for the Moon to shine through. Dense fog
    also obstructs the sky even when model cloud layers are low.
    """
    total = _clamp_pct(cloud_cover_pct)
    if cloud_low_pct is not None and cloud_mid_pct is not None and cloud_high_pct is not None:
        low = _clamp_pct(cloud_low_pct)
        mid = _clamp_pct(cloud_mid_pct)
        high = _clamp_pct(cloud_high_pct)
        layered = min(total, max(low, mid) + 0.5 * high)
    else:
        layered = total

    if weather_code in {45, 48} or (visibility_km is not None and visibility_km < 1.5):
        layered = max(layered, 95.0)
    return round(_clamp_pct(layered), 1)


def format_sky_condition(
    cloud_cover_pct: float,
    sky_obstruction_pct: float,
    cloud_high_pct: float | None = None,
    weather_code: int | None = None,
) -> str:
    """Build a concise English sky condition label."""
    if weather_code in {45, 48}:
        return f"Fog ({sky_obstruction_pct:.0f}% obstructed)"
    if (
        cloud_high_pct is not None
        and cloud_high_pct >= 50.0
        and sky_obstruction_pct <= 55.0
        and cloud_cover_pct >= 60.0
    ):
        return f"Thin high clouds ({cloud_high_pct:.0f}% cirrus)"
    if sky_obstruction_pct < 15.0:
        return f"Clear ({cloud_cover_pct:.0f}% clouds)"
    if sky_obstruction_pct < 45.0:
        return f"Partly cloudy ({cloud_cover_pct:.0f}% clouds)"
    if sky_obstruction_pct < 80.0:
        return f"Mostly cloudy ({cloud_cover_pct:.0f}% clouds)"
    base = _WMO_DESCRIPTIONS.get(weather_code or -1, "Overcast")
    if base in {"Clear sky", "Mainly clear", "Partly cloudy"}:
        base = "Overcast"
    return f"{base} ({cloud_cover_pct:.0f}% clouds)"


@dataclass(frozen=True, slots=True)
class WeatherSnapshot:
    """Local atmospheric conditions relevant to refraction, QNH, and lunar visibility."""

    fetched_utc: datetime
    temperature_c: float
    pressure_msl_hpa: float
    surface_pressure_hpa: float
    cloud_cover_pct: float
    cloud_cover_low_pct: float
    cloud_cover_mid_pct: float
    cloud_cover_high_pct: float
    visibility_km: float | None
    weather_code: int | None
    sky_obstruction_pct: float
    condition_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched_utc": self.fetched_utc.isoformat(),
            "temperature_c": round(self.temperature_c, 1),
            "pressure_msl_hpa": round(self.pressure_msl_hpa, 1),
            "surface_pressure_hpa": round(self.surface_pressure_hpa, 1),
            "cloud_cover_pct": round(self.cloud_cover_pct, 1),
            "cloud_cover_low_pct": round(self.cloud_cover_low_pct, 1),
            "cloud_cover_mid_pct": round(self.cloud_cover_mid_pct, 1),
            "cloud_cover_high_pct": round(self.cloud_cover_high_pct, 1),
            "visibility_km": round(self.visibility_km, 1) if self.visibility_km is not None else None,
            "weather_code": self.weather_code,
            "sky_obstruction_pct": round(self.sky_obstruction_pct, 1),
            "condition_label": self.condition_label,
        }


def parse_open_meteo_payload(
    data: dict[str, Any],
    *,
    fetched_utc: datetime | None = None,
) -> WeatherSnapshot:
    """Parse an Open-Meteo /v1/forecast JSON response into a WeatherSnapshot."""
    cur = data.get("current")
    if not isinstance(cur, dict):
        raise ValueError("Missing 'current' object in Open-Meteo response")

    ts = fetched_utc or datetime.now(timezone.utc)
    temp_c = _as_float(cur.get("temperature_2m"), 10.0)
    pressure_msl = _as_float(cur.get("pressure_msl"), 1013.25)
    surface_pressure = _as_float(cur.get("surface_pressure"), pressure_msl)

    cloud_total = _clamp_pct(_as_float(cur.get("cloud_cover"), 0.0))
    has_layers = all(
        isinstance(cur.get(k), (int, float)) and not isinstance(cur.get(k), bool)
        for k in ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high")
    )
    cloud_low = _clamp_pct(_as_float(cur.get("cloud_cover_low"), cloud_total))
    cloud_mid = _clamp_pct(_as_float(cur.get("cloud_cover_mid"), 0.0))
    cloud_high = _clamp_pct(_as_float(cur.get("cloud_cover_high"), 0.0))

    raw_vis = cur.get("visibility")
    vis_km = (
        float(raw_vis) / 1000.0
        if isinstance(raw_vis, (int, float)) and not isinstance(raw_vis, bool)
        else None
    )
    raw_code = cur.get("weather_code")
    wmo_code = (
        int(raw_code)
        if isinstance(raw_code, (int, float)) and not isinstance(raw_code, bool)
        else None
    )

    obstruction = compute_sky_obstruction_pct(
        cloud_total,
        cloud_low if has_layers else None,
        cloud_mid if has_layers else None,
        cloud_high if has_layers else None,
        visibility_km=vis_km,
        weather_code=wmo_code,
    )
    label = format_sky_condition(
        cloud_total,
        obstruction,
        cloud_high if has_layers else None,
        weather_code=wmo_code,
    )
    return WeatherSnapshot(
        fetched_utc=ts,
        temperature_c=temp_c,
        pressure_msl_hpa=pressure_msl,
        surface_pressure_hpa=surface_pressure,
        cloud_cover_pct=cloud_total,
        cloud_cover_low_pct=cloud_low,
        cloud_cover_mid_pct=cloud_mid,
        cloud_cover_high_pct=cloud_high,
        visibility_km=vis_km,
        weather_code=wmo_code,
        sky_obstruction_pct=obstruction,
        condition_label=label,
    )


class WeatherClient:
    """Cached HTTP client for Open-Meteo local atmospheric & cloud-cover conditions."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        enabled: bool = True,
        refresh_interval_s: float = 600.0,
        max_stale_s: float = 3600.0,
        timeout_s: float = 4.0,
        base_url: str = OPEN_METEO_FORECAST_URL,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.enabled = enabled
        self.refresh_interval_s = max(60.0, float(refresh_interval_s))
        self.max_stale_s = max(self.refresh_interval_s, float(max_stale_s))
        self.base_url = base_url
        self._client = httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": "LunaCross/0.1 (+https://github.com/wvzuilen/lunacross)"},
        )
        self._cached: WeatherSnapshot | None = None
        self._cached_mono: float | None = None
        self._last_attempt_mono: float | None = None

    def __enter__(self) -> WeatherClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_weather(self) -> WeatherSnapshot | None:
        """Return current or cached WeatherSnapshot, refreshing periodically without raising."""
        if not self.enabled:
            return None

        now_mono = time.monotonic()
        if (
            self._cached is not None
            and self._cached_mono is not None
            and (now_mono - self._cached_mono) < self.refresh_interval_s
        ):
            return self._cached

        # Back off for 60s after any failed attempt so network hiccups never stall polling
        if (
            self._last_attempt_mono is not None
            and (now_mono - self._last_attempt_mono) < 60.0
        ):
            if (
                self._cached is not None
                and self._cached_mono is not None
                and (now_mono - self._cached_mono) <= self.max_stale_s
            ):
                return self._cached
            return None

        self._last_attempt_mono = now_mono
        params = {
            "latitude": f"{self.latitude:.4f}",
            "longitude": f"{self.longitude:.4f}",
            "current": (
                "temperature_2m,pressure_msl,surface_pressure,"
                "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
                "visibility,weather_code"
            ),
            "timezone": "UTC",
        }
        try:
            resp = self._client.get(self.base_url, params=params)
            resp.raise_for_status()
            self._cached = parse_open_meteo_payload(resp.json())
            self._cached_mono = now_mono
            return self._cached
        except Exception:
            if (
                self._cached is not None
                and self._cached_mono is not None
                and (now_mono - self._cached_mono) <= self.max_stale_s
            ):
                return self._cached
            return None
