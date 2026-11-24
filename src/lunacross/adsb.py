"""HTTP client and parser for tar1090 / readsb ADS-B JSON data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from lunacross.geometry import FEET_TO_METERS


@dataclass(frozen=True, slots=True)
class Aircraft:
    """Current ADS-B state vector of an airborne aircraft."""

    hex: str
    flight: str
    registration: str | None
    aircraft_type: str | None
    category: str | None
    lat: float
    lon: float
    altitude_m: float
    altitude_ft: float
    altitude_source: str
    ground_speed_kt: float | None
    track_deg: float | None
    vertical_rate_fpm: float
    seen_pos_s: float
    squawk: str | None
    rssi: float | None


@dataclass(frozen=True, slots=True)
class AdsbSnapshot:
    """Snapshot of all usable airborne aircraft from a single ADS-B poll."""

    timestamp_unix: float
    aircraft: list[Aircraft]
    total_raw: int


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_aircraft_payload(data: dict[str, Any], max_seen_pos_s: float = 30.0) -> AdsbSnapshot:
    """Parse a tar1090/readsb aircraft.json dictionary into an AdsbSnapshot."""
    raw_list = data.get("aircraft") or []
    timestamp_unix = float(data.get("now") or 0.0)
    parsed: list[Aircraft] = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue

        lat = item.get("lat")
        lon = item.get("lon")
        if not (_is_number(lat) and _is_number(lon)):
            continue

        seen_pos = item.get("seen_pos", item.get("seen", 0.0))
        seen_pos_s = float(seen_pos) if _is_number(seen_pos) else 0.0
        if seen_pos_s > max_seen_pos_s:
            continue

        alt_geom = item.get("alt_geom")
        alt_baro = item.get("alt_baro")
        if _is_number(alt_geom):
            altitude_ft = float(alt_geom)
            altitude_source = "geom"
        elif _is_number(alt_baro):
            altitude_ft = float(alt_baro)
            altitude_source = "baro"
        else:
            # Skip ground traffic ("ground") and aircraft without altitude
            continue

        geom_rate = item.get("geom_rate")
        baro_rate = item.get("baro_rate")
        if _is_number(geom_rate):
            vertical_rate_fpm = float(geom_rate)
        elif _is_number(baro_rate):
            vertical_rate_fpm = float(baro_rate)
        else:
            vertical_rate_fpm = 0.0

        hex_code = str(item.get("hex") or "").strip().lower()
        raw_flight = str(item.get("flight") or "").strip()
        flight = raw_flight if raw_flight else hex_code.upper()

        gs = item.get("gs")
        track = item.get("track")
        rssi = item.get("rssi")

        parsed.append(
            Aircraft(
                hex=hex_code,
                flight=flight,
                registration=item.get("r"),
                aircraft_type=item.get("t"),
                category=item.get("category"),
                lat=float(lat),
                lon=float(lon),
                altitude_m=altitude_ft * FEET_TO_METERS,
                altitude_ft=altitude_ft,
                altitude_source=altitude_source,
                ground_speed_kt=float(gs) if _is_number(gs) else None,
                track_deg=float(track) % 360.0 if _is_number(track) else None,
                vertical_rate_fpm=vertical_rate_fpm,
                seen_pos_s=seen_pos_s,
                squawk=item.get("squawk"),
                rssi=float(rssi) if _is_number(rssi) else None,
            )
        )

    return AdsbSnapshot(
        timestamp_unix=timestamp_unix,
        aircraft=parsed,
        total_raw=len(raw_list),
    )


def _derive_receiver_url(adsb_url: str) -> str:
    parts = urlsplit(adsb_url)
    path = parts.path
    if path.endswith("aircraft.json"):
        new_path = path[: -len("aircraft.json")] + "receiver.json"
    else:
        new_path = path.rstrip("/") + "/receiver.json"
    return urlunsplit((parts.scheme, parts.netloc, new_path, "", ""))


class AdsbClient:
    """Synchronous HTTP client for tar1090/readsb JSON endpoints."""

    def __init__(self, adsb_url: str, timeout_s: float = 3.0) -> None:
        self.adsb_url = adsb_url
        self.receiver_url = _derive_receiver_url(adsb_url)
        self._client = httpx.Client(timeout=timeout_s)

    def __enter__(self) -> AdsbClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch_snapshot(self, max_seen_pos_s: float = 30.0) -> AdsbSnapshot:
        """Fetch live aircraft from /data/aircraft.json."""
        resp = self._client.get(self.adsb_url)
        resp.raise_for_status()
        return parse_aircraft_payload(resp.json(), max_seen_pos_s=max_seen_pos_s)

    def fetch_receiver_location(self) -> tuple[float, float] | None:
        """Fetch configured receiver coordinates from /data/receiver.json."""
        resp = self._client.get(self.receiver_url)
        resp.raise_for_status()
        data = resp.json()
        lat = data.get("lat")
        lon = data.get("lon")
        if _is_number(lat) and _is_number(lon):
            return float(lat), float(lon)
        return None
