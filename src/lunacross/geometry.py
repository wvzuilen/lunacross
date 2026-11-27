"""3D WGS-84 geometry, atmospheric refraction, and flight trajectory extrapolation."""

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
TURN_DAMPING_TAU_S = 15.0

# Characteristic wingspan/length (meters) for common ICAO aircraft type designators
_ICAO_TYPE_WINGSPAN_M: dict[str, float] = {
    "A388": 79.8,
    "B748": 68.4,
    "B744": 64.4,
    "A35K": 64.8,
    "A359": 64.8,
    "B77W": 64.8,
    "B77L": 64.8,
    "B772": 60.9,
    "B773": 60.9,
    "B788": 60.1,
    "B789": 60.1,
    "B78X": 60.1,
    "A332": 60.3,
    "A333": 60.3,
    "A338": 64.0,
    "A339": 64.0,
    "A343": 60.3,
    "A346": 63.5,
    "MD11": 51.7,
    "B763": 47.6,
    "B764": 51.9,
    "A306": 44.8,
    "A310": 43.9,
    "C17": 51.8,
    "A400": 42.4,
    "B752": 38.1,
    "B753": 38.1,
    "A321": 35.8,
    "A21N": 35.8,
    "A320": 35.8,
    "A20N": 35.8,
    "A319": 35.8,
    "A19N": 35.8,
    "A318": 34.1,
    "B737": 34.3,
    "B738": 35.8,
    "B739": 35.8,
    "B38M": 35.9,
    "B39M": 35.9,
    "BCS1": 35.1,
    "BCS3": 35.1,
    "A221": 35.1,
    "A223": 35.1,
    "E190": 28.7,
    "E195": 28.7,
    "E290": 33.7,
    "E295": 35.1,
    "E170": 26.0,
    "E175": 26.0,
    "CRJ9": 24.9,
    "CRJX": 26.2,
    "DH8D": 28.4,
    "AT76": 27.1,
    "AT75": 27.1,
    "GL7T": 31.7,
    "GLEX": 28.7,
    "GL6T": 28.7,
    "GLF6": 30.4,
    "GLF5": 28.5,
    "FA8X": 26.2,
    "FA7X": 26.2,
    "F2TH": 21.4,
    "CL60": 19.6,
    "CL35": 21.0,
    "CL30": 21.0,
    "C68A": 19.4,
    "C680": 19.4,
    "C56X": 18.0,
    "C25C": 15.5,
    "C25B": 15.9,
    "C25A": 14.3,
    "E55P": 16.2,
    "E50P": 12.3,
    "PC24": 17.0,
    "PC12": 16.3,
    "BE20": 16.6,
    "TBM9": 12.8,
    "SR22": 11.7,
    "SR20": 11.7,
    "DA40": 11.9,
    "DA42": 13.4,
    "C172": 11.0,
    "A210": 9.1,
    "EC45": 11.0,
    "EC35": 10.2,
    "A139": 13.8,
}

# Estimated characteristic aircraft size (wingspan/length) per ADS-B emitter category
_CATEGORY_SIZE_M: dict[str, float] = {
    "A1": 12.0,   # Light (< 15,500 lbs)
    "A2": 25.0,   # Small (15,500 - 75,000 lbs)
    "A3": 38.0,   # Large (75,000 - 300,000 lbs, e.g. A320 / 737)
    "A4": 45.0,   # High-vortex large (e.g. B757)
    "A5": 65.0,   # Heavy (> 300,000 lbs, e.g. B777 / A350 / A380)
    "A6": 20.0,   # High performance
    "A7": 15.0,   # Rotorcraft
}
_DEFAULT_AIRCRAFT_SIZE_M = 35.0


@dataclass(frozen=True, slots=True)
class SkyPosition:
    """Apparent sky coordinates of an aircraft from the observer's point of view."""

    altitude_deg: float
    azimuth_deg: float
    slant_range_km: float
    ground_distance_km: float


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    """Convert WGS-84 geodetic coordinates to ECEF (x, y, z) in meters."""
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
    """Convert an ECEF difference vector to local East-North-Up (ENU) in meters."""
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


def _wgs84_radii(lat_rad: float) -> tuple[float, float]:
    """Return WGS-84 meridional (M) and prime-vertical (N) radii of curvature in meters."""
    sin_lat = math.sin(lat_rad)
    w2 = 1.0 - WGS84_E2 * sin_lat * sin_lat
    w = math.sqrt(w2)
    m_rad = WGS84_A * (1.0 - WGS84_E2) / (w2 * w)
    n_rad = WGS84_A / w
    return m_rad, n_rad


def aircraft_altaz(
    obs_lat: float,
    obs_lon: float,
    obs_alt_m: float,
    ac_lat: float,
    ac_lon: float,
    ac_alt_m: float,
    apply_refraction: bool = True,
    *,
    temperature_c: float = 10.0,
    pressure_hpa: float = 1010.0,
) -> SkyPosition:
    """Compute apparent altitude, azimuth, slant range, and ground distance to an aircraft."""
    ox, oy, oz = geodetic_to_ecef(obs_lat, obs_lon, obs_alt_m)
    ax, ay, az = geodetic_to_ecef(ac_lat, ac_lon, ac_alt_m)
    east, north, up = ecef_to_enu(ax - ox, ay - oy, az - oz, obs_lat, obs_lon)

    horiz = math.hypot(east, north)
    slant_m = math.hypot(horiz, up)
    geom_alt_deg = math.degrees(math.atan2(up, horiz))
    az_deg = math.degrees(math.atan2(east, north)) % 360.0

    apparent_alt_deg = geom_alt_deg
    if apply_refraction and geom_alt_deg > -1.0:
        # Scale atmospheric refraction by aircraft height above the observer
        delta_alt = max(0.0, ac_alt_m - obs_alt_m)
        scale = 1.0 - math.exp(-delta_alt / ATMOSPHERE_SCALE_HEIGHT_M)
        apparent_alt_deg += (
            float(refraction(geom_alt_deg, temperature_C=temperature_c, pressure_mbar=pressure_hpa))
            * scale
        )

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
    """Numerically stable great-circle angular separation (Vincenty formula) in degrees."""
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


def _damped_turn_offset_deg(track_rate_deg_s: float, dt_s: float) -> float:
    if abs(track_rate_deg_s) <= 1e-4 or abs(dt_s) <= 1e-6:
        return 0.0
    sign = 1.0 if dt_s >= 0.0 else -1.0
    return sign * track_rate_deg_s * TURN_DAMPING_TAU_S * (1.0 - math.exp(-abs(dt_s) / TURN_DAMPING_TAU_S))


def extrapolate_position(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    ground_speed_kt: float,
    track_deg: float,
    vertical_rate_fpm: float,
    dt_s: float,
    track_rate_deg_s: float = 0.0,
) -> tuple[float, float, float]:
    """Project aircraft position (lat, lon, alt_m) forward by dt_s seconds on WGS-84."""
    new_alt_m = max(0.0, alt_m + vertical_rate_fpm * FPM_TO_MPS * dt_s)
    dist_m = ground_speed_kt * KNOTS_TO_MPS * dt_s
    if abs(dist_m) < 1e-6:
        return lat_deg, lon_deg, new_alt_m

    if abs(lat_deg) > 85.0:
        # Polar fallback using spherical trigonometry
        angular_dist = dist_m / (EARTH_MEAN_RADIUS_M + 0.5 * (alt_m + new_alt_m))
        bearing = math.radians(track_deg + _damped_turn_offset_deg(track_rate_deg_s, 0.5 * dt_s))
        lat1 = math.radians(lat_deg)
        lon1 = math.radians(lon_deg)
        sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
        sin_ad, cos_ad = math.sin(angular_dist), math.cos(angular_dist)
        sin_lat2 = max(-1.0, min(1.0, sin_lat1 * cos_ad + cos_lat1 * sin_ad * math.cos(bearing)))
        lat2 = math.asin(sin_lat2)
        lon2 = lon1 + math.atan2(math.sin(bearing) * sin_ad * cos_lat1, cos_ad - sin_lat1 * sin_lat2)
        lon2_deg = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
        return math.degrees(lat2), lon2_deg, new_alt_m

    # Second-order midpoint integration on the WGS-84 ellipsoid at flight altitude
    lat1_rad = math.radians(lat_deg)
    lon1_rad = math.radians(lon_deg)
    h_mid = 0.5 * (alt_m + new_alt_m)

    m1, n1 = _wgs84_radii(lat1_rad)
    psi_quarter = math.radians(track_deg + _damped_turn_offset_deg(track_rate_deg_s, 0.25 * dt_s))
    dlat_half = (0.5 * dist_m * math.cos(psi_quarter)) / (m1 + alt_m)
    lat_mid = max(math.radians(-89.9), min(math.radians(89.9), lat1_rad + dlat_half))

    m_mid, n_mid = _wgs84_radii(lat_mid)
    psi_mid = math.radians(track_deg + _damped_turn_offset_deg(track_rate_deg_s, 0.5 * dt_s))
    dlat = (dist_m * math.cos(psi_mid)) / (m_mid + h_mid)
    dlon = (dist_m * math.sin(psi_mid)) / ((n_mid + h_mid) * math.cos(lat_mid))

    lat2_deg = math.degrees(max(math.radians(-90.0), min(math.radians(90.0), lat1_rad + dlat)))
    lon2_deg = (math.degrees(lon1_rad + dlon) + 540.0) % 360.0 - 180.0
    return lat2_deg, lon2_deg, new_alt_m


def estimate_aircraft_angular_size_deg(
    slant_range_km: float,
    category: str | None,
    aircraft_type: str | None = None,
) -> float:
    """Estimate the apparent angular span of an aircraft in the sky in degrees."""
    if slant_range_km <= 0.0:
        return 0.0
    type_key = (aircraft_type or "").strip().upper()
    if type_key in _ICAO_TYPE_WINGSPAN_M:
        size_m = _ICAO_TYPE_WINGSPAN_M[type_key]
    else:
        size_m = _CATEGORY_SIZE_M.get((category or "").strip().upper(), _DEFAULT_AIRCRAFT_SIZE_M)
    return math.degrees(2.0 * math.atan(size_m / (2.0 * slant_range_km * 1000.0)))


def classify_visual_visibility(
    plane_angular_size_deg: float,
    moon_illumination_pct: float = 50.0,
) -> tuple[str, str]:
    """Classify whether a transit is visible to the naked eye or requires optics."""
    arcmin = max(0.0, plane_angular_size_deg * 60.0)
    naked_thresh = 2.8 if moon_illumination_pct >= 80.0 else 2.2
    if arcmin >= naked_thresh:
        return "NAKED_EYE", f"Naked eye ({arcmin:.1f}')"
    if arcmin >= 1.2:
        return "BINOCULARS", f"Binoculars / zoom ({arcmin:.1f}')"
    return "TELESCOPE", f"Telescope / telephoto only ({arcmin:.1f}')"


def _altaz_to_unit(alt_deg: float, az_deg: float) -> tuple[float, float, float]:
    alt = math.radians(alt_deg)
    az = math.radians(az_deg)
    cos_alt = math.cos(alt)
    return cos_alt * math.sin(az), cos_alt * math.cos(az), math.sin(alt)


def segment_closest_approach(
    alt1_deg: float,
    az1_deg: float,
    alt2_deg: float,
    az2_deg: float,
    target_alt_deg: float,
    target_az_deg: float,
) -> tuple[float, float, float, float]:
    """Find closest approach along the chord between two sky positions to a target.

    Returns (min_sep_deg, fraction_u, interp_alt_deg, interp_az_deg).
    """
    v1 = _altaz_to_unit(alt1_deg, az1_deg)
    v2 = _altaz_to_unit(alt2_deg, az2_deg)
    vt = _altaz_to_unit(target_alt_deg, target_az_deg)

    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    dz = v2[2] - v1[2]
    denom = dx * dx + dy * dy + dz * dz
    if denom < 1e-15:
        sep = angular_separation_deg(alt1_deg, az1_deg, target_alt_deg, target_az_deg)
        return sep, 0.0, alt1_deg, az1_deg % 360.0

    num = (vt[0] - v1[0]) * dx + (vt[1] - v1[1]) * dy + (vt[2] - v1[2]) * dz
    u = max(0.0, min(1.0, num / denom))

    px = v1[0] + u * dx
    py = v1[1] + u * dy
    pz = v1[2] + u * dz
    norm = math.sqrt(px * px + py * py + pz * pz)
    if norm > 0.0:
        px /= norm
        py /= norm
        pz /= norm

    interp_alt_deg = math.degrees(math.asin(max(-1.0, min(1.0, pz))))
    interp_az_deg = math.degrees(math.atan2(px, py)) % 360.0
    min_sep = angular_separation_deg(interp_alt_deg, interp_az_deg, target_alt_deg, target_az_deg)
    return min_sep, u, interp_alt_deg, interp_az_deg


def sky_tangent_offset_deg(
    alt_deg: float,
    az_deg: float,
    ref_alt_deg: float,
    ref_az_deg: float,
) -> tuple[float, float]:
    """Return local tangent-plane offsets (dx_east_deg, dy_up_deg) around a reference sky point."""
    d_az = ((az_deg - ref_az_deg + 180.0) % 360.0) - 180.0
    dx_deg = d_az * math.cos(math.radians(ref_alt_deg))
    dy_deg = alt_deg - ref_alt_deg
    return dx_deg, dy_deg
