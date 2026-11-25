"""Topocentric Moon position and instantaneous angular velocity using Skyfield."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from skyfield import almanac
from skyfield.api import Loader, wgs84

MOON_RADIUS_KM = 1737.4
_VELOCITY_STEP_S = 10.0

_COMPASS_POINTS = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


def compass_direction(azimuth_deg: float) -> str:
    """Return the 16-point compass direction for an azimuth in degrees."""
    idx = int(((azimuth_deg % 360.0) + 11.25) // 22.5) % 16
    return _COMPASS_POINTS[idx]


def moon_phase_name(phase_deg: float) -> str:
    """Return the English Moon phase name for a phase angle (0°-360°)."""
    angle = phase_deg % 360.0
    if angle < 22.5 or angle >= 337.5:
        return "New Moon"
    if angle < 67.5:
        return "Waxing Crescent"
    if angle < 112.5:
        return "First Quarter"
    if angle < 157.5:
        return "Waxing Gibbous"
    if angle < 202.5:
        return "Full Moon"
    if angle < 247.5:
        return "Waning Gibbous"
    if angle < 292.5:
        return "Last Quarter"
    return "Waning Crescent"


@dataclass(frozen=True, slots=True)
class MoonPosition:
    """Topocentric position, angular size, and sky drift rate of the Moon at an instant."""

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
    alt_rate_deg_s: float = 0.0
    az_rate_deg_s: float = 0.0

    def at_offset(self, dt_s: float) -> tuple[float, float]:
        """Return predicted (altitude_deg, azimuth_deg) of the Moon dt_s seconds later."""
        alt = max(-90.0, min(90.0, self.altitude_deg + self.alt_rate_deg_s * dt_s))
        az = (self.azimuth_deg + self.az_rate_deg_s * dt_s) % 360.0
        return alt, az


@dataclass(frozen=True, slots=True)
class MoonRiseSet:
    """Next moonrise and moonset timestamps in UTC."""

    next_rise_utc: datetime | None
    next_set_utc: datetime | None


def _ensure_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MoonTracker:
    """Computes apparent Moon position, angular size, drift velocity, illumination, and rise/set."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float = 0.0,
        ephemeris_dir: str | Path = ".cache/skyfield",
        *,
        geoid_undulation_m: float = 0.0,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.elevation_m = elevation_m
        self.geoid_undulation_m = geoid_undulation_m

        loader = Loader(str(ephemeris_dir))
        self.ts = loader.timescale()
        self.eph = loader("de421.bsp")
        self.earth = self.eph["earth"]
        self.moon = self.eph["moon"]
        self.sun = self.eph["sun"]
        self.topos = wgs84.latlon(
            latitude,
            longitude,
            elevation_m=elevation_m + geoid_undulation_m,
        )
        self.observer = self.earth + self.topos

    def get_position(self, dt: datetime | None = None) -> MoonPosition:
        """Compute apparent topocentric altitude, azimuth, drift rate, and diameter of the Moon."""
        dt_utc = _ensure_utc(dt)
        t = self.ts.from_datetime(dt_utc)
        t_next = self.ts.from_datetime(dt_utc + timedelta(seconds=_VELOCITY_STEP_S))

        apparent = self.observer.at(t).observe(self.moon).apparent()
        alt, az, distance = apparent.altaz("standard")

        apparent_next = self.observer.at(t_next).observe(self.moon).apparent()
        alt_next, az_next, _ = apparent_next.altaz("standard")

        dist_km = float(distance.km)
        angular_radius_deg = math.degrees(math.asin(min(1.0, MOON_RADIUS_KM / dist_km)))
        illumination_pct = float(apparent.fraction_illuminated(self.sun) * 100.0)
        phase_deg = float(almanac.moon_phase(self.eph, t).degrees) % 360.0
        alt_deg = float(alt.degrees)
        az_deg = float(az.degrees) % 360.0

        alt_rate = (float(alt_next.degrees) - alt_deg) / _VELOCITY_STEP_S
        d_az = ((float(az_next.degrees) - az_deg + 180.0) % 360.0) - 180.0
        az_rate = d_az / _VELOCITY_STEP_S

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
            alt_rate_deg_s=alt_rate,
            az_rate_deg_s=az_rate,
        )

    def next_rise_set(self, dt: datetime | None = None) -> MoonRiseSet:
        """Find the next moonrise and moonset within the next 36 hours."""
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
