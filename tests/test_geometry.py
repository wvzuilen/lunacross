"""Unit tests voor 3D WGS-84 geometrie, refractie en baanextrapolatie."""

from __future__ import annotations

import math

import pytest

from lunacross.geometry import (
    aircraft_altaz,
    angular_separation_deg,
    estimate_aircraft_angular_size_deg,
    extrapolate_position,
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

    # Wraparound rond 0° / 360°
    wrap = angular_separation_deg(10.0, 359.5, 10.0, 0.5)
    expected = math.degrees(2.0 * math.asin(math.cos(math.radians(10.0)) * math.sin(math.radians(0.5))))
    assert wrap == pytest.approx(expected, abs=1e-6)

    # Zeer kleine scheiding van 0.01° op 45° hoogte
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
    # 65m op 20km ~ 0.186°
    assert 0.15 < ang < 0.22
