"""Local weather, cloud-cover obstruction, and atmospheric state via METAR ceilometers & Open-Meteo."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

AVIATION_WEATHER_METAR_URL = "https://aviationweather.gov/api/data/metar"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_EARTH_RADIUS_KM = 6371.0

_METAR_COVER_PCT: dict[str, float] = {
    "CLR": 0.0,
    "SKC": 0.0,
    "NSC": 0.0,
    "NCD": 0.0,
    "CAVOK": 0.0,
    "FEW": 18.0,
    "SCT": 44.0,
    "BKN": 75.0,
    "OVC": 100.0,
    "OVX": 100.0,
}

_CLEAR_COVER_CODES = {"CLR", "SKC", "NSC", "NCD", "CAVOK"}

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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _surface_pressure_from_msl(pressure_msl_hpa: float, elevation_m: float, temp_c: float) -> float:
    if abs(elevation_m) < 0.5:
        return pressure_msl_hpa
    t_kelvin = max(200.0, temp_c + 273.15)
    factor = (1.0 - (0.0065 * elevation_m) / (t_kelvin + 0.0065 * elevation_m)) ** 5.257
    return pressure_msl_hpa * factor


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
    source_tag: str | None = None,
) -> str:
    """Build a concise English sky condition label with optional station/source tag."""
    suffix = f" · {source_tag}" if source_tag else ""
    if weather_code in {45, 48}:
        return f"Fog ({sky_obstruction_pct:.0f}% obstructed){suffix}"
    if (
        cloud_high_pct is not None
        and cloud_high_pct >= 40.0
        and sky_obstruction_pct <= 55.0
        and cloud_cover_pct >= 40.0
    ):
        return f"Thin high clouds ({cloud_high_pct:.0f}% cirrus){suffix}"
    if sky_obstruction_pct < 15.0:
        return f"Clear ({cloud_cover_pct:.0f}% clouds){suffix}"
    if sky_obstruction_pct < 45.0:
        return f"Partly cloudy ({cloud_cover_pct:.0f}% clouds){suffix}"
    if sky_obstruction_pct < 80.0:
        return f"Mostly cloudy ({cloud_cover_pct:.0f}% clouds){suffix}"
    base = _WMO_DESCRIPTIONS.get(weather_code or -1, "Overcast")
    if base in {"Clear sky", "Mainly clear", "Partly cloudy"}:
        base = "Overcast"
    return f"{base} ({cloud_cover_pct:.0f}% clouds){suffix}"


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
    source: str = "open-meteo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched_utc": self.fetched_utc.isoformat(),
            "source": self.source,
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


def _parse_metar_visibility_km(visib: Any, raw_ob: str) -> float | None:
    if " 9999 " in f" {raw_ob} " or "CAVOK" in raw_ob:
        return 10.0
    if isinstance(visib, (int, float)) and not isinstance(visib, bool):
        return round(float(visib) * 1.60934, 2)
    if isinstance(visib, str):
        cleaned = visib.strip().rstrip("+").strip()
        if not cleaned:
            return None
        try:
            if "/" in cleaned:
                parts = cleaned.split()
                total_sm = 0.0
                for part in parts:
                    if "/" in part:
                        n_s, d_s = part.split("/", 1)
                        total_sm += float(n_s) / float(d_s)
                    else:
                        total_sm += float(part)
                return round(total_sm * 1.60934, 2)
            return round(float(cleaned) * 1.60934, 2)
        except ValueError:
            return None
    return None


def parse_metar_station(
    station: dict[str, Any],
    *,
    observer_elevation_m: float = 0.0,
    fetched_utc: datetime | None = None,
) -> WeatherSnapshot | None:
    """Convert a single AviationWeather METAR JSON record into a WeatherSnapshot.

    Returns None if the station lacks usable ceilometer/cloud coverage information.
    """
    icao = str(station.get("icaoId") or "").strip().upper()
    raw_ob = str(station.get("rawOb") or "")
    top_cover = str(station.get("cover") or "").strip().upper()
    clouds_raw = station.get("clouds")
    clouds_list = clouds_raw if isinstance(clouds_raw, list) else []

    # Reject automated stations whose ceilometer data is missing (e.g. '///' with no layers)
    if not clouds_list and top_cover not in _METAR_COVER_PCT:
        if not any(tok in f" {raw_ob} " for tok in (" CAVOK ", " SKC ", " CLR ", " NCD ", " NSC ")):
            return None

    low_pct = 0.0
    mid_pct = 0.0
    high_pct = 0.0

    for layer in clouds_list:
        if not isinstance(layer, dict):
            continue
        cov_code = str(layer.get("cover") or "").strip().upper()
        pct = _METAR_COVER_PCT.get(cov_code)
        if pct is None:
            continue
        base_ft = layer.get("base")
        if cov_code == "OVX" or not (isinstance(base_ft, (int, float)) and not isinstance(base_ft, bool)):
            low_pct = max(low_pct, pct)
        elif float(base_ft) < 6500.0:
            low_pct = max(low_pct, pct)
        elif float(base_ft) < 20000.0:
            mid_pct = max(mid_pct, pct)
        else:
            high_pct = max(high_pct, pct)

    if not clouds_list:
        if top_cover == "OVX":
            low_pct = 100.0
        elif top_cover in _METAR_COVER_PCT and top_cover not in _CLEAR_COVER_CODES:
            low_pct = _METAR_COVER_PCT[top_cover]

    cloud_total = max(low_pct, mid_pct, high_pct)

    temp_c = _as_float(station.get("temp"), 10.0)
    raw_altim = station.get("altim")
    if isinstance(raw_altim, (int, float)) and not isinstance(raw_altim, bool):
        val = float(raw_altim)
        pressure_msl = val * 33.8639 if val < 100.0 else val
    else:
        pressure_msl = 1013.25
    surface_pressure = _surface_pressure_from_msl(pressure_msl, observer_elevation_m, temp_c)

    vis_km = _parse_metar_visibility_km(station.get("visib"), raw_ob)
    wx_str = str(station.get("wxString") or "").upper()
    is_fog = "FG" in wx_str or top_cover == "OVX"
    wmo_code: int | None = 45 if is_fog else (0 if cloud_total == 0.0 else None)

    obstruction = compute_sky_obstruction_pct(
        cloud_total,
        low_pct,
        mid_pct,
        high_pct,
        visibility_km=vis_km,
        weather_code=wmo_code,
    )
    label = format_sky_condition(
        cloud_total,
        obstruction,
        high_pct,
        weather_code=wmo_code,
        source_tag=icao or None,
    )
    ts = fetched_utc or datetime.now(timezone.utc)
    return WeatherSnapshot(
        fetched_utc=ts,
        temperature_c=temp_c,
        pressure_msl_hpa=pressure_msl,
        surface_pressure_hpa=surface_pressure,
        cloud_cover_pct=cloud_total,
        cloud_cover_low_pct=low_pct,
        cloud_cover_mid_pct=mid_pct,
        cloud_cover_high_pct=high_pct,
        visibility_km=vis_km,
        weather_code=wmo_code,
        sky_obstruction_pct=obstruction,
        condition_label=label,
        source=f"METAR:{icao}" if icao else "METAR",
    )


def select_nearest_metar_snapshot(
    stations: list[dict[str, Any]],
    observer_lat: float,
    observer_lon: float,
    *,
    observer_elevation_m: float = 0.0,
    max_distance_km: float = 60.0,
    max_obs_age_s: float = 5400.0,
    now_utc: datetime | None = None,
) -> WeatherSnapshot | None:
    """Pick the closest fresh METAR ceilometer station to the observer."""
    cur_utc = now_utc or datetime.now(timezone.utc)
    cur_unix = cur_utc.timestamp()

    candidates: list[tuple[float, WeatherSnapshot]] = []
    for st in stations:
        if not isinstance(st, dict):
            continue
        slat = st.get("lat")
        slon = st.get("lon")
        if not (
            isinstance(slat, (int, float))
            and not isinstance(slat, bool)
            and isinstance(slon, (int, float))
            and not isinstance(slon, bool)
        ):
            continue
        dist_km = _haversine_km(observer_lat, observer_lon, float(slat), float(slon))
        if dist_km > max_distance_km:
            continue

        obs_time = st.get("obsTime")
        if isinstance(obs_time, (int, float)) and not isinstance(obs_time, bool):
            # Reject stale METARs (> max_obs_age_s old relative to report clock)
            age_s = cur_unix - float(obs_time)
            if age_s > max_obs_age_s:
                continue

        snap = parse_metar_station(
            st,
            observer_elevation_m=observer_elevation_m,
            fetched_utc=cur_utc,
        )
        if snap is not None:
            candidates.append((dist_km, snap))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def parse_open_meteo_payload(
    data: dict[str, Any],
    *,
    fetched_utc: datetime | None = None,
    source_tag: str | None = None,
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
        source_tag=source_tag,
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
        source="open-meteo:icon",
    )


class WeatherClient:
    """Cached HTTP client using nearby METAR ceilometers first and Open-Meteo ICON as fallback."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        elevation_m: float = 0.0,
        enabled: bool = True,
        refresh_interval_s: float = 600.0,
        max_stale_s: float = 3600.0,
        timeout_s: float = 4.0,
        metar_url: str = AVIATION_WEATHER_METAR_URL,
        base_url: str = OPEN_METEO_FORECAST_URL,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.elevation_m = elevation_m
        self.enabled = enabled
        self.refresh_interval_s = max(60.0, float(refresh_interval_s))
        self.max_stale_s = max(self.refresh_interval_s, float(max_stale_s))
        self.metar_url = metar_url
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

    def _fetch_metar(self) -> WeatherSnapshot | None:
        dlat = 0.60
        dlon = max(0.60, 0.60 / max(0.25, abs(math.cos(math.radians(self.latitude)))))
        bbox = (
            f"{self.latitude - dlat:.3f},{self.longitude - dlon:.3f},"
            f"{self.latitude + dlat:.3f},{self.longitude + dlon:.3f}"
        )
        resp = self._client.get(self.metar_url, params={"bbox": bbox, "format": "json"})
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return None
        return select_nearest_metar_snapshot(
            payload,
            self.latitude,
            self.longitude,
            observer_elevation_m=self.elevation_m,
        )

    def _fetch_open_meteo_icon(self) -> WeatherSnapshot:
        params = {
            "latitude": f"{self.latitude:.4f}",
            "longitude": f"{self.longitude:.4f}",
            "current": (
                "temperature_2m,pressure_msl,surface_pressure,"
                "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
                "visibility,weather_code"
            ),
            "models": "icon_seamless",
            "timezone": "UTC",
        }
        resp = self._client.get(self.base_url, params=params)
        resp.raise_for_status()
        return parse_open_meteo_payload(resp.json(), source_tag="ICON")

    def get_weather(self) -> WeatherSnapshot | None:
        """Return current or cached WeatherSnapshot, preferring real METAR ceilometers."""
        if not self.enabled:
            return None

        now_mono = time.monotonic()
        if (
            self._cached is not None
            and self._cached_mono is not None
            and (now_mono - self._cached_mono) < self.refresh_interval_s
        ):
            return self._cached

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

        # 1. Try real airport METAR ceilometer within 60 km
        try:
            metar_snap = self._fetch_metar()
            if metar_snap is not None:
                self._cached = metar_snap
                self._cached_mono = now_mono
                return self._cached
        except Exception:
            pass

        # 2. Fall back to DWD ICON seamless model via Open-Meteo
        try:
            icon_snap = self._fetch_open_meteo_icon()
            self._cached = icon_snap
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
