"""Unit tests for 3D WGS-84 geometry, refraction, and trajectory extrapolation."""

from __future__ import annotations

import math

import pytest

from lunacross.geometry import (
    aircraft_altaz,
    angular_separation_deg,
    classify_visual_visibility,
    estimate_aircraft_angular_size_deg,
    extrapolate_position,
    geodetic_to_ecef,
    ecef_to_enu,
    segment_closest_approach,
    sky_tangent_offset_deg,
)

OBS_LAT = 52.3731
OBS_LON = 4.8926
OBS_ALT_M = 2.0


def test_overhead_aircraft() -> None:
    sky = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT, OBS_LON, OBS_ALT_M + 10000.0)
    assert sky.altitude_deg == pytest.approx(90.0, abs=1e-4)
    assert sky.slant_range_km == pytest.approx(10.0, abs=1e-3)
    assert sky.ground_distance_km == pytest.approx(0.0, abs=1e-5)


def test_cardinal_azimuths() -> None:
    north = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT + 0.1, OBS_LON, 10000.0)
    east = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT, OBS_LON + 0.1, 10000.0)
    south = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT - 0.1, OBS_LON, 10000.0)
    west = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT, OBS_LON - 0.1, 10000.0)

    assert north.azimuth_deg == pytest.approx(0.0, abs=0.05) or north.azimuth_deg == pytest.approx(
        360.0, abs=0.05
    )
    assert east.azimuth_deg == pytest.approx(90.0, abs=0.1)
    assert south.azimuth_deg == pytest.approx(180.0, abs=0.05)
    assert west.azimuth_deg == pytest.approx(270.0, abs=0.1)


def test_angular_separation_precision_and_wraparound() -> None:
    assert angular_separation_deg(45.0, 120.0, 45.0, 120.0) == pytest.approx(0.0, abs=1e-9)
    assert angular_separation_deg(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0, abs=1e-6)
    assert angular_separation_deg(0.0, 0.0, 0.0, 90.0) == pytest.approx(90.0, abs=1e-6)

    # Wraparound across 0° / 360°
    wrap = angular_separation_deg(10.0, 359.5, 10.0, 0.5)
    expected = math.degrees(2.0 * math.asin(math.cos(math.radians(10.0)) * math.sin(math.radians(0.5))))
    assert wrap == pytest.approx(expected, abs=1e-6)

    # Tiny separation of 0.01° at 45° altitude
    tiny = angular_separation_deg(45.0, 180.0, 45.01, 180.0)
    assert tiny == pytest.approx(0.01, abs=1e-7)


def test_refraction_increases_low_elevation_and_scales_with_altitude() -> None:
    low_ac = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT + 0.5, OBS_LON, 1000.0, apply_refraction=True)
    low_ac_geom = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT + 0.5, OBS_LON, 1000.0, apply_refraction=False)
    high_ac = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT + 1.5, OBS_LON, 12000.0, apply_refraction=True)
    high_ac_geom = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, OBS_LAT + 1.5, OBS_LON, 12000.0, apply_refraction=False)

    assert low_ac.altitude_deg > low_ac_geom.altitude_deg
    assert high_ac.altitude_deg > high_ac_geom.altitude_deg
    assert (high_ac.altitude_deg - high_ac_geom.altitude_deg) > (
        low_ac.altitude_deg - low_ac_geom.altitude_deg
    ) or high_ac.altitude_deg > 1.0


def test_extrapolate_position_north_and_climb() -> None:
    lat2, lon2, alt2 = extrapolate_position(
        OBS_LAT,
        OBS_LON,
        1000.0,
        ground_speed_kt=360.0,
        track_deg=0.0,
        vertical_rate_fpm=1000.0,
        dt_s=60.0,
    )
    assert lat2 > OBS_LAT
    assert lon2 == pytest.approx(OBS_LON, abs=1e-6)
    assert alt2 == pytest.approx(1000.0 + 1000.0 * 0.3048, rel=1e-4)


def test_aircraft_angular_size() -> None:
    ang = estimate_aircraft_angular_size_deg(slant_range_km=20.0, category="A5")
    # 65m at 20km ~ 0.186°
    assert 0.15 < ang < 0.22


def test_segment_closest_approach_midpoint_crossing() -> None:
    sep, u, interp_alt, interp_az = segment_closest_approach(
        alt1_deg=40.0,
        az1_deg=179.0,
        alt2_deg=40.0,
        az2_deg=181.0,
        target_alt_deg=40.0,
        target_az_deg=180.0,
    )
    assert u == pytest.approx(0.5, abs=1e-3)
    assert sep < 0.02
    assert interp_alt == pytest.approx(40.0, abs=0.05)
    assert interp_az == pytest.approx(180.0, abs=0.05)


def test_sky_tangent_offset_wraparound() -> None:
    dx, dy = sky_tangent_offset_deg(alt_deg=41.5, az_deg=1.0, ref_alt_deg=40.0, ref_az_deg=359.0)
    assert dx == pytest.approx(2.0 * math.cos(math.radians(40.0)), abs=1e-4)
    assert dy == pytest.approx(1.5, abs=1e-6)


def test_wgs84_ellipsoidal_east_extrapolation_and_turn_rate() -> None:
    # 100 m/s due East for 100s = 10,000 m at 10,000 m altitude
    gs_kt = 100.0 / 0.5144444444
    lat2, lon2, alt2 = extrapolate_position(
        OBS_LAT, OBS_LON, 10000.0, ground_speed_kt=gs_kt, track_deg=90.0, vertical_rate_fpm=0.0, dt_s=100.0
    )
    ox, oy, oz = geodetic_to_ecef(OBS_LAT, OBS_LON, 10000.0)
    ax, ay, az = geodetic_to_ecef(lat2, lon2, alt2)
    east, north, _ = ecef_to_enu(ax - ox, ay - oy, az - oz, OBS_LAT, OBS_LON)
    # True 3D chord/arc East distance at 10 km altitude should match 10,000 m within < 2 m
    assert math.hypot(east, north) == pytest.approx(10000.0, abs=2.0)

    # Positive turn rate from North (0°) deflects aircraft Eastward
    _, lon_turn, _ = extrapolate_position(
        OBS_LAT, OBS_LON, 10000.0, ground_speed_kt=gs_kt, track_deg=0.0, vertical_rate_fpm=0.0, dt_s=10.0, track_rate_deg_s=1.5
    )
    assert lon_turn > OBS_LON


def test_icao_type_angular_size_and_visibility_tiers() -> None:
    # CL35 (21m) at 41.4 km against 97.8% Full Moon -> BINOCULARS (~1.74')
    cl35_deg = estimate_aircraft_angular_size_deg(41.4, category="A2", aircraft_type="CL35")
    tier_cl, label_cl = classify_visual_visibility(cl35_deg, moon_illumination_pct=97.8)
    assert tier_cl == "BINOCULARS"
    assert "Binoculars" in label_cl

    # B77W (64.8m) at 18.0 km against Full Moon -> NAKED_EYE (~12.4')
    b77w_deg = estimate_aircraft_angular_size_deg(18.0, category="A5", aircraft_type="B77W")
    tier_77, label_77 = classify_visual_visibility(b77w_deg, moon_illumination_pct=99.0)
    assert tier_77 == "NAKED_EYE"
    assert "Naked eye" in label_77
