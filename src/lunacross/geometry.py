"""3D WGS-84 geometrie, refractie en baanextrapolatie voor vliegtuigen."""

from __future__ import annotations

import math
from dataclasses import dataclass

from skyfield.earthlib import refraction

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

FEET_TO_METERS = 0.3048
KNOTS_TO_MPS = 0.5144444444
FPM_TO_MPS = 0.00508
EARTH_MEAN_RADIUS_M = 6371008.8
ATMOSPHERE_SCALE_HEIGHT_M = 8500.0

# Geschatte karakteristieke vliegtuigafmeting (spanwijdte/romp) per ADS-B categorie
_CATEGORY_SIZE_M: dict[str, float] = {
    "A1": 12.0,   # Licht (< 15.500 lbs)
    "A2": 25.0,   # Klein (15.500 - 75.000 lbs)
    "A3": 38.0,   # Groot (75.000 - 300.000 lbs, bijv. A320 / 737)
    "A4": 45.0,   # High-vortex groot (bijv. B757)
    "A5": 65.0,   # Zwaar (> 300.000 lbs, bijv. B777 / A350 / A380)
    "A6": 20.0,   # High performance
    "A7": 15.0,   # Helikopter
}
_DEFAULT_AIRCRAFT_SIZE_M = 35.0


@dataclass(frozen=True, slots=True)
class SkyPosition:
    """Positie van een vliegtuig aan de hemel vanuit het standpunt van de waarnemer."""

    altitude_deg: float
    azimuth_deg: float
    slant_range_km: float
    ground_distance_km: float


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    """Zet WGS-84 geodetische coördinaten om naar ECEF (x, y, z) in meters."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(
    dx: float,
    dy: float,
    dz: float,
    obs_lat_deg: float,
    obs_lon_deg: float,
) -> tuple[float, float, float]:
    """Zet een ECEF-verschilvector om naar lokaal East-North-Up (ENU) in meters."""
    lat = math.radians(obs_lat_deg)
    lon = math.radians(obs_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def _haversine_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    dlat = lat2 - lat1
    dlon = math.radians(lon2_deg - lon1_deg)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return (EARTH_MEAN_RADIUS_M * c) / 1000.0


def aircraft_altaz(
    obs_lat: float,
    obs_lon: float,
    obs_alt_m: float,
    ac_lat: float,
    ac_lon: float,
    ac_alt_m: float,
    apply_refraction: bool = True,
) -> SkyPosition:
    """Bereken schijnbare hoogte, azimut, schuine afstand en grondafstand tot een vliegtuig."""
    ox, oy, oz = geodetic_to_ecef(obs_lat, obs_lon, obs_alt_m)
    ax, ay, az = geodetic_to_ecef(ac_lat, ac_lon, ac_alt_m)
    east, north, up = ecef_to_enu(ax - ox, ay - oy, az - oz, obs_lat, obs_lon)

    horiz = math.hypot(east, north)
    slant_m = math.hypot(horiz, up)
    geom_alt_deg = math.degrees(math.atan2(up, horiz))
    az_deg = math.degrees(math.atan2(east, north)) % 360.0

    apparent_alt_deg = geom_alt_deg
    if apply_refraction and geom_alt_deg > -1.0:
        # Schaal atmosferische refractie met de hoogte van het vliegtuig boven de waarnemer
        delta_alt = max(0.0, ac_alt_m - obs_alt_m)
        scale = 1.0 - math.exp(-delta_alt / ATMOSPHERE_SCALE_HEIGHT_M)
        apparent_alt_deg += float(refraction(geom_alt_deg, temperature_C=10.0, pressure_mbar=1010.0)) * scale

    return SkyPosition(
        altitude_deg=apparent_alt_deg,
        azimuth_deg=az_deg,
        slant_range_km=slant_m / 1000.0,
        ground_distance_km=_haversine_km(obs_lat, obs_lon, ac_lat, ac_lon),
    )


def angular_separation_deg(
    alt1_deg: float,
    az1_deg: float,
    alt2_deg: float,
    az2_deg: float,
) -> float:
    """Numeriek stabiele hoekafstand (Vincenty-formule) tussen twee punten aan de hemel."""
    phi1 = math.radians(alt1_deg)
    phi2 = math.radians(alt2_deg)
    d_lam = math.radians(az2_deg - az1_deg)

    cos_phi1 = math.cos(phi1)
    sin_phi1 = math.sin(phi1)
    cos_phi2 = math.cos(phi2)
    sin_phi2 = math.sin(phi2)
    cos_dlam = math.cos(d_lam)
    sin_dlam = math.sin(d_lam)

    num = math.hypot(
        cos_phi2 * sin_dlam,
        cos_phi1 * sin_phi2 - sin_phi1 * cos_phi2 * cos_dlam,
    )
    den = sin_phi1 * sin_phi2 + cos_phi1 * cos_phi2 * cos_dlam
    return math.degrees(math.atan2(num, den))


def extrapolate_position(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    ground_speed_kt: float,
    track_deg: float,
    vertical_rate_fpm: float,
    dt_s: float,
) -> tuple[float, float, float]:
    """Extrapoleer vliegtuigpositie (lat, lon, alt_m) over dt_s seconden langs grootcirkel."""
    new_alt_m = max(0.0, alt_m + vertical_rate_fpm * FPM_TO_MPS * dt_s)
    dist_m = ground_speed_kt * KNOTS_TO_MPS * dt_s
    if abs(dist_m) < 1e-6:
        return lat_deg, lon_deg, new_alt_m

    angular_dist = dist_m / EARTH_MEAN_RADIUS_M
    bearing = math.radians(track_deg)
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_ad = math.sin(angular_dist)
    cos_ad = math.cos(angular_dist)

    sin_lat2 = max(-1.0, min(1.0, sin_lat1 * cos_ad + cos_lat1 * sin_ad * math.cos(bearing)))
    lat2 = math.asin(sin_lat2)
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * sin_ad * cos_lat1,
        cos_ad - sin_lat1 * sin_lat2,
    )
    lon2_deg = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    return math.degrees(lat2), lon2_deg, new_alt_m


def estimate_aircraft_angular_size_deg(slant_range_km: float, category: str | None) -> float:
    """Schat de schijnbare hoekgrootte van het vliegtuig aan de hemel in graden."""
    if slant_range_km <= 0.0:
        return 0.0
    size_m = _CATEGORY_SIZE_M.get((category or "").strip().upper(), _DEFAULT_AIRCRAFT_SIZE_M)
    return math.degrees(2.0 * math.atan(size_m / (2.0 * slant_range_km * 1000.0)))
