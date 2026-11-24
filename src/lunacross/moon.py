"""Topocentrische berekening van de maanpositie met Skyfield."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from skyfield import almanac
from skyfield.api import Loader, wgs84

MOON_RADIUS_KM = 1737.4

_COMPASS_POINTS = (
    "N",
    "NNO",
    "NO",
    "ONO",
    "O",
    "OZO",
    "ZO",
    "ZZO",
    "Z",
    "ZZW",
    "ZW",
    "WZW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


def compass_direction(azimuth_deg: float) -> str:
    """Geef de 16-punts Nederlandse windstreek voor een azimut in graden."""
    idx = int(((azimuth_deg % 360.0) + 11.25) // 22.5) % 16
    return _COMPASS_POINTS[idx]


def moon_phase_name(phase_deg: float) -> str:
    """Geef de Nederlandse naam van de maanfase voor een fasehoek (0°-360°)."""
    angle = phase_deg % 360.0
    if angle < 22.5 or angle >= 337.5:
        return "Nieuwe maan"
    if angle < 67.5:
        return "Jonge maansikkel"
    if angle < 112.5:
        return "Eerste kwartier"
    if angle < 157.5:
        return "Wassende maan"
    if angle < 202.5:
        return "Volle maan"
    if angle < 247.5:
        return "Afnemende maan"
    if angle < 292.5:
        return "Laatste kwartier"
    return "Asgrauwe maansikkel"


@dataclass(frozen=True, slots=True)
class MoonPosition:
    """Topocentrische positie en eigenschappen van de maan op een tijdstip."""

    timestamp_utc: datetime
    altitude_deg: float
    azimuth_deg: float
    distance_km: float
    angular_radius_deg: float
    angular_diameter_deg: float
    illumination_pct: float
    phase_deg: float
    phase_name: str
    is_above_horizon: bool


@dataclass(frozen=True, slots=True)
class MoonRiseSet:
    """Eerstvolgende maansopkomst en maansondergang in UTC."""

    next_rise_utc: datetime | None
    next_set_utc: datetime | None


def _ensure_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MoonTracker:
    """Berekent de actuele positie, grootte, verlichting en opkomst/ondergang van de maan."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float = 0.0,
        ephemeris_dir: str | Path = ".cache/skyfield",
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.elevation_m = elevation_m

        loader = Loader(str(ephemeris_dir))
        self.ts = loader.timescale()
        self.eph = loader("de421.bsp")
        self.earth = self.eph["earth"]
        self.moon = self.eph["moon"]
        self.sun = self.eph["sun"]
        self.topos = wgs84.latlon(latitude, longitude, elevation_m=elevation_m)
        self.observer = self.earth + self.topos

    def get_position(self, dt: datetime | None = None) -> MoonPosition:
        """Bereken schijnbare topocentrische hoogte, azimut en hoekdiameter van de maan."""
        dt_utc = _ensure_utc(dt)
        t = self.ts.from_datetime(dt_utc)

        apparent = self.observer.at(t).observe(self.moon).apparent()
        alt, az, distance = apparent.altaz("standard")

        dist_km = float(distance.km)
        angular_radius_deg = math.degrees(math.asin(min(1.0, MOON_RADIUS_KM / dist_km)))
        illumination_pct = float(apparent.fraction_illuminated(self.sun) * 100.0)
        phase_deg = float(almanac.moon_phase(self.eph, t).degrees) % 360.0
        alt_deg = float(alt.degrees)
        az_deg = float(az.degrees) % 360.0

        return MoonPosition(
            timestamp_utc=dt_utc,
            altitude_deg=alt_deg,
            azimuth_deg=az_deg,
            distance_km=dist_km,
            angular_radius_deg=angular_radius_deg,
            angular_diameter_deg=2.0 * angular_radius_deg,
            illumination_pct=illumination_pct,
            phase_deg=phase_deg,
            phase_name=moon_phase_name(phase_deg),
            is_above_horizon=alt_deg > 0.0,
        )

    def next_rise_set(self, dt: datetime | None = None) -> MoonRiseSet:
        """Zoek eerstvolgende maansopkomst en maansondergang binnen 36 uur."""
        dt_utc = _ensure_utc(dt)
        t0 = self.ts.from_datetime(dt_utc)
        t1 = self.ts.from_datetime(dt_utc + timedelta(hours=36))

        rise_times, rise_crossed = almanac.find_risings(self.observer, self.moon, t0, t1)
        set_times, set_crossed = almanac.find_settings(self.observer, self.moon, t0, t1)

        next_rise = next(
            (ti.utc_datetime() for ti, crossed in zip(rise_times, rise_crossed) if crossed),
            None,
        )
        next_set = next(
            (ti.utc_datetime() for ti, crossed in zip(set_times, set_crossed) if crossed),
            None,
        )
        return MoonRiseSet(next_rise_utc=next_rise, next_set_utc=next_set)
