"""Unit tests for lunar transit detection and trajectory prediction."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from lunacross.adsb import Aircraft
from lunacross.detector import TransitDetector
from lunacross.geometry import KNOTS_TO_MPS, aircraft_altaz, extrapolate_position
from lunacross.moon import MoonPosition

OBS_LAT = 52.3731
OBS_LON = 4.8926
OBS_ALT_M = 2.0


def _test_moon(alt_deg: float = 35.0, az_deg: float = 180.0) -> MoonPosition:
    return MoonPosition(
        timestamp_utc=datetime(2026, 11, 24, 22, 0, 0, tzinfo=timezone.utc),
        altitude_deg=alt_deg,
        azimuth_deg=az_deg,
        distance_km=384400.0,
        angular_radius_deg=0.259,
        angular_diameter_deg=0.518,
        illumination_pct=85.0,
        phase_deg=140.0,
        phase_name="Waxing Gibbous",
        is_above_horizon=alt_deg > 0.0,
    )


def _aircraft_on_moon_line(moon: MoonPosition, alt_ft: float = 35000.0) -> tuple[float, float, float]:
    alt_m = alt_ft * 0.3048
    ground_dist_m = (alt_m - OBS_ALT_M) / math.tan(math.radians(moon.altitude_deg))
    lat, lon, _ = extrapolate_position(
        OBS_LAT,
        OBS_LON,
        alt_m,
        ground_speed_kt=ground_dist_m / KNOTS_TO_MPS,
        track_deg=moon.azimuth_deg,
        vertical_rate_fpm=0.0,
        dt_s=1.0,
    )
    return lat, lon, alt_m


def test_direct_transit_and_deduplication() -> None:
    moon = _test_moon()
    lat, lon, alt_m = _aircraft_on_moon_line(moon)
    ac = Aircraft(
        hex="484161",
        flight="KLM123",
        registration="PH-BVA",
        aircraft_type="B77W",
        category="A5",
        lat=lat,
        lon=lon,
        altitude_m=alt_m,
        altitude_ft=35000.0,
        altitude_source="geom",
        ground_speed_kt=450.0,
        track_deg=90.0,
        vertical_rate_fpm=0.0,
        seen_pos_s=0.0,
        squawk="1000",
        rssi=-10.0,
    )

    detector = TransitDetector(OBS_LAT, OBS_LON, OBS_ALT_M, alert_margin_deg=0.5, cooldown_s=180.0)
    t0 = datetime(2026, 11, 24, 22, 0, 0, tzinfo=timezone.utc)
    sightings, events = detector.evaluate(moon, [ac], now_utc=t0)

    assert len(sightings) == 1
    assert sightings[0].status == "TRANSIT"
    assert sightings[0].is_disk_crossing is True
    assert sightings[0].angular_sep_deg < moon.angular_radius_deg
    assert len(events) == 1
    assert events[0].event_type == "TRANSIT"

    # A second poll 2s later must not re-emit within the cooldown window
    _, events2 = detector.evaluate(moon, [ac], now_utc=t0 + timedelta(seconds=2))
    assert events2 == []


def test_incoming_prediction_and_upgrade_to_transit() -> None:
    moon = _test_moon()
    cross_lat, cross_lon, alt_m = _aircraft_on_moon_line(moon)
    gs_kt = 450.0
    track_deg = 90.0
    # Place aircraft 30s before the crossing point on heading 90° (west of the crossing point)
    start_lat, start_lon, _ = extrapolate_position(
        cross_lat,
        cross_lon,
        alt_m,
        ground_speed_kt=gs_kt,
        track_deg=270.0,
        vertical_rate_fpm=0.0,
        dt_s=30.0,
    )

    ac_incoming = Aircraft(
        hex="484999",
        flight="TRA456",
        registration="PH-HSI",
        aircraft_type="B738",
        category="A3",
        lat=start_lat,
        lon=start_lon,
        altitude_m=alt_m,
        altitude_ft=35000.0,
        altitude_source="geom",
        ground_speed_kt=gs_kt,
        track_deg=track_deg,
        vertical_rate_fpm=0.0,
        seen_pos_s=0.0,
        squawk="2000",
        rssi=-18.0,
    )

    detector = TransitDetector(
        OBS_LAT,
        OBS_LON,
        OBS_ALT_M,
        alert_margin_deg=0.5,
        predict_horizon_s=120.0,
        cooldown_s=180.0,
    )
    t0 = datetime(2026, 11, 24, 22, 0, 0, tzinfo=timezone.utc)
    sightings, events = detector.evaluate(moon, [ac_incoming], now_utc=t0)

    assert sightings[0].status == "INCOMING"
    assert sightings[0].time_to_closest_s is not None
    assert 27.0 <= sightings[0].time_to_closest_s <= 33.0
    assert sightings[0].min_predicted_sep_deg is not None
    assert sightings[0].min_predicted_sep_deg < moon.angular_radius_deg
    assert len(events) == 1
    assert events[0].event_type == "INCOMING"

    # Upgrading to TRANSIT 30s later must emit immediately despite cooldown
    ac_transit = Aircraft(
        hex=ac_incoming.hex,
        flight=ac_incoming.flight,
        registration=ac_incoming.registration,
        aircraft_type=ac_incoming.aircraft_type,
        category=ac_incoming.category,
        lat=cross_lat,
        lon=cross_lon,
        altitude_m=alt_m,
        altitude_ft=35000.0,
        altitude_source="geom",
        ground_speed_kt=gs_kt,
        track_deg=track_deg,
        vertical_rate_fpm=0.0,
        seen_pos_s=0.0,
        squawk="2000",
        rssi=-15.0,
    )
    sightings2, events2 = detector.evaluate(moon, [ac_transit], now_utc=t0 + timedelta(seconds=30))
    assert sightings2[0].status == "TRANSIT"
    assert len(events2) == 1
    assert events2[0].event_type == "TRANSIT"


def test_near_miss_status() -> None:
    moon = _test_moon(alt_deg=35.0, az_deg=180.0)
    cross_lat, cross_lon, alt_m = _aircraft_on_moon_line(moon)
    # Offset ~0.45° east of the lunar center (~146 m at 18.6 km slant range)
    near_lat, near_lon, _ = extrapolate_position(
        cross_lat,
        cross_lon,
        alt_m,
        ground_speed_kt=146.0 / KNOTS_TO_MPS,
        track_deg=90.0,
        vertical_rate_fpm=0.0,
        dt_s=1.0,
    )
    sky = aircraft_altaz(OBS_LAT, OBS_LON, OBS_ALT_M, near_lat, near_lon, alt_m)
    assert sky.slant_range_km > 0.0

    ac = Aircraft(
        hex="484777",
        flight="KLM777",
        registration=None,
        aircraft_type="E190",
        category="A3",
        lat=near_lat,
        lon=near_lon,
        altitude_m=alt_m,
        altitude_ft=35000.0,
        altitude_source="geom",
        ground_speed_kt=400.0,
        track_deg=90.0,
        vertical_rate_fpm=0.0,
        seen_pos_s=0.0,
        squawk=None,
        rssi=None,
    )
    detector = TransitDetector(OBS_LAT, OBS_LON, OBS_ALT_M, alert_margin_deg=0.5)
    sightings, events = detector.evaluate(moon, [ac])
    assert sightings[0].status == "NEAR_MISS"
    assert len(events) == 1
    assert events[0].event_type == "NEAR_MISS"
