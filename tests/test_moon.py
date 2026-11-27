"""Unit tests for topocentric Moon position using Skyfield."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lunacross.moon import MoonTracker, compass_direction, moon_phase_name


def test_moon_tracker_fixed_timestamp(tmp_path) -> None:
    tracker = MoonTracker(
        latitude=52.3731,
        longitude=4.8926,
        elevation_m=2.0,
        ephemeris_dir=".cache/skyfield",
    )
    dt = datetime(2026, 11, 24, 21, 45, 0, tzinfo=timezone.utc)
    pos = tracker.get_position(dt)

    assert -90.0 <= pos.altitude_deg <= 90.0
    assert 0.0 <= pos.azimuth_deg < 360.0
    assert 350000.0 <= pos.distance_km <= 410000.0
    assert 0.45 <= pos.angular_diameter_deg <= 0.60
    assert 0.0 <= pos.illumination_pct <= 100.0
    assert pos.is_above_horizon == (pos.altitude_deg > 0.0)

    rs = tracker.next_rise_set(dt)
    assert rs.next_rise_utc is not None
    assert rs.next_set_utc is not None
    assert rs.next_rise_utc > dt
    assert rs.next_set_utc > dt

    # Verify instantaneous Moon drift velocity accurately predicts Moon position 60s later
    pos_60 = tracker.get_position(dt + timedelta(seconds=60))
    pred_alt, pred_az = pos.at_offset(60.0)
    assert pred_alt == pytest.approx(pos_60.altitude_deg, abs=0.002)
    assert pred_az == pytest.approx(pos_60.azimuth_deg, abs=0.002)
    assert pos.bright_limb_angle_deg is not None
    assert 0.0 <= pos.effective_bright_limb_deg < 360.0
    assert 0.0 <= pos.north_angle_deg < 360.0


def test_compass_and_phase_helpers() -> None:
    assert compass_direction(0.0) == "N"
    assert compass_direction(90.0) == "E"
    assert compass_direction(180.0) == "S"
    assert compass_direction(270.0) == "W"
    assert compass_direction(359.0) == "N"

    assert moon_phase_name(0.0) == "New Moon"
    assert moon_phase_name(90.0) == "First Quarter"
    assert moon_phase_name(180.0) == "Full Moon"
    assert moon_phase_name(270.0) == "Last Quarter"
