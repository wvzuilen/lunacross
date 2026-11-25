"""HTTP client and parser for tar1090 / readsb and community ADS-B JSON feeds."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
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
    source: str = "local"
    track_rate_deg_s: float = 0.0


@dataclass(frozen=True, slots=True)
class AdsbSnapshot:
    """Snapshot of all usable airborne aircraft from an ADS-B poll."""

    timestamp_unix: float
    aircraft: list[Aircraft]
    total_raw: int


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_aircraft_payload(
    data: dict[str, Any],
    max_seen_pos_s: float = 30.0,
    *,
    source: str = "local",
    geoid_undulation_m: float = 0.0,
) -> AdsbSnapshot:
    """Parse a tar1090/readsb aircraft.json or v2 community API dictionary into an AdsbSnapshot."""
    raw_list = data.get("aircraft")
    if raw_list is None:
        raw_list = data.get("ac") or []

    raw_now = data.get("now") or 0.0
    timestamp_unix = float(raw_now)
    if timestamp_unix > 1e12:
        timestamp_unix /= 1000.0

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
            # RTCA DO-260B alt_geom is GNSS Height Above WGS-84 Ellipsoid (HAE)
            altitude_m = altitude_ft * FEET_TO_METERS
        elif _is_number(alt_baro):
            altitude_ft = float(alt_baro)
            altitude_source = "baro"
            nav_qnh = item.get("nav_qnh")
            qnh_corr_ft = (
                (float(nav_qnh) - 1013.25) * 27.3
                if _is_number(nav_qnh) and 940.0 <= float(nav_qnh) <= 1060.0
                else 0.0
            )
            altitude_m = (altitude_ft + qnh_corr_ft) * FEET_TO_METERS + geoid_undulation_m
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

        raw_track_rate = item.get("track_rate")
        track_rate_deg_s = (
            float(raw_track_rate)
            if _is_number(raw_track_rate) and abs(float(raw_track_rate)) <= 10.0
            else 0.0
        )

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
                altitude_m=altitude_m,
                altitude_ft=altitude_ft,
                altitude_source=altitude_source,
                ground_speed_kt=float(gs) if _is_number(gs) else None,
                track_deg=float(track) % 360.0 if _is_number(track) else None,
                vertical_rate_fpm=vertical_rate_fpm,
                seen_pos_s=seen_pos_s,
                squawk=item.get("squawk"),
                rssi=float(rssi) if _is_number(rssi) else None,
                source=source,
                track_rate_deg_s=track_rate_deg_s,
            )
        )

    return AdsbSnapshot(
        timestamp_unix=timestamp_unix,
        aircraft=parsed,
        total_raw=len(raw_list),
    )


def merge_snapshots(
    primary: AdsbSnapshot | None,
    supplement: AdsbSnapshot | None,
    *,
    supplement_age_s: float = 0.0,
    max_seen_pos_s: float = 30.0,
) -> AdsbSnapshot:
    """Merge primary local ADS-B aircraft with an optional community supplement snapshot."""
    if primary is None and supplement is None:
        return AdsbSnapshot(timestamp_unix=0.0, aircraft=[], total_raw=0)

    supp_by_hex: dict[str, Aircraft] = {}
    if supplement is not None:
        for ac in supplement.aircraft:
            supp_by_hex[ac.hex] = ac

    merged: list[Aircraft] = []
    seen_hexes: set[str] = set()

    if primary is not None:
        for ac in primary.aircraft:
            seen_hexes.add(ac.hex)
            match = supp_by_hex.get(ac.hex)
            if match is not None:
                merged.append(
                    replace(
                        ac,
                        registration=ac.registration or match.registration,
                        aircraft_type=ac.aircraft_type or match.aircraft_type,
                        category=ac.category or match.category,
                    )
                )
            else:
                merged.append(ac)

    age_offset = max(0.0, supplement_age_s)
    added_supplement = 0
    if supplement is not None:
        for ac in supplement.aircraft:
            if ac.hex in seen_hexes:
                continue
            adjusted_seen = ac.seen_pos_s + age_offset
            if adjusted_seen > max_seen_pos_s:
                continue
            seen_hexes.add(ac.hex)
            added_supplement += 1
            merged.append(replace(ac, seen_pos_s=adjusted_seen))

    timestamp_unix = (
        primary.timestamp_unix
        if primary is not None
        else (supplement.timestamp_unix if supplement is not None else 0.0)
    )
    primary_raw = primary.total_raw if primary is not None else 0
    total_raw = max(len(merged), primary_raw + added_supplement)
    return AdsbSnapshot(
        timestamp_unix=timestamp_unix,
        aircraft=merged,
        total_raw=total_raw,
    )


def _derive_receiver_url(adsb_url: str) -> str:
    parts = urlsplit(adsb_url)
    path = parts.path
    if path.endswith("aircraft.json"):
        new_path = path[: -len("aircraft.json")] + "receiver.json"
    else:
        new_path = path.rstrip("/") + "/receiver.json"
    return urlunsplit((parts.scheme, parts.netloc, new_path, "", ""))


def _format_supplement_url(
    template: str | None,
    lat: float | None,
    lon: float | None,
    radius_nm: float,
) -> str | None:
    if not template or lat is None or lon is None:
        return None
    return template.format(
        lat=f"{lat:.4f}",
        lon=f"{lon:.4f}",
        dist_nm=max(1, int(round(radius_nm))),
    )


class AdsbClient:
    """Synchronous HTTP client for local tar1090/readsb + optional community supplement feed."""

    def __init__(
        self,
        adsb_url: str,
        timeout_s: float = 3.0,
        *,
        supplement_url_template: str | None = None,
        observer_lat: float | None = None,
        observer_lon: float | None = None,
        supplement_radius_nm: float = 45.0,
        supplement_interval_s: float = 5.0,
        geoid_undulation_m: float = 0.0,
    ) -> None:
        self.adsb_url = adsb_url
        self.receiver_url = _derive_receiver_url(adsb_url)
        self.supplement_url = _format_supplement_url(
            supplement_url_template, observer_lat, observer_lon, supplement_radius_nm
        )
        self.supplement_interval_s = max(1.0, supplement_interval_s)
        self.geoid_undulation_m = geoid_undulation_m
        self._client = httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": "LunaCross/0.1 (+https://github.com/wvzuilen/lunacross)"},
        )
        self._cached_supplement: AdsbSnapshot | None = None
        self._cached_supplement_at: float | None = None

    def __enter__(self) -> AdsbClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch_snapshot(self, max_seen_pos_s: float = 30.0) -> AdsbSnapshot:
        """Fetch live aircraft from local /data/aircraft.json and merge optional supplement."""
        primary: AdsbSnapshot | None = None
        primary_err: Exception | None = None
        try:
            resp = self._client.get(self.adsb_url)
            resp.raise_for_status()
            primary = parse_aircraft_payload(
                resp.json(),
                max_seen_pos_s=max_seen_pos_s,
                source="local",
                geoid_undulation_m=self.geoid_undulation_m,
            )
        except httpx.HTTPError as exc:
            primary_err = exc

        if self.supplement_url is not None:
            now_mono = time.monotonic()
            should_refresh = (
                self._cached_supplement is None
                or self._cached_supplement_at is None
                or (now_mono - self._cached_supplement_at) >= self.supplement_interval_s
            )
            if should_refresh:
                try:
                    s_resp = self._client.get(self.supplement_url)
                    s_resp.raise_for_status()
                    self._cached_supplement = parse_aircraft_payload(
                        s_resp.json(),
                        max_seen_pos_s=max_seen_pos_s,
                        source="supplement",
                        geoid_undulation_m=self.geoid_undulation_m,
                    )
                    self._cached_supplement_at = now_mono
                except httpx.HTTPError:
                    pass

        if primary is None:
            if self._cached_supplement is not None and self._cached_supplement_at is not None:
                age_s = max(0.0, time.monotonic() - self._cached_supplement_at)
                if age_s <= max_seen_pos_s:
                    return merge_snapshots(
                        None,
                        self._cached_supplement,
                        supplement_age_s=age_s,
                        max_seen_pos_s=max_seen_pos_s,
                    )
            assert primary_err is not None
            raise primary_err

        age_s = (
            max(0.0, time.monotonic() - self._cached_supplement_at)
            if self._cached_supplement_at is not None
            else 0.0
        )
        return merge_snapshots(
            primary,
            self._cached_supplement,
            supplement_age_s=age_s,
            max_seen_pos_s=max_seen_pos_s,
        )

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
