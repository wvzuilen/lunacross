"""Unit tests for pass persistence, sky visualization (SVG/HTML), and Web UI."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from lunacross.adsb import Aircraft
from lunacross.config import Config
from lunacross.detector import AircraftSighting, PassPoint, PassRecord
from lunacross.geometry import SkyPosition
from lunacross.moon import MoonPosition
from lunacross.notifier import TransitNotifier, load_passes
from lunacross.visualize import (
    render_live_snapshot_svg,
    render_pass_html,
    render_pass_svg,
)
from lunacross.web import LiveState, LunaCrossWebServer


def _sample_pass_record() -> PassRecord:
    return PassRecord(
        pass_id="20261125T003435Z_484161",
        flight="KLM999",
        hex="484161",
        registration="PH-BVA",
        aircraft_type="B77W",
        category="A5",
        observer_lat=52.3731,
        observer_lon=4.8926,
        observer_elevation_m=2.0,
        start_utc="2026-11-25T00:34:25+00:00",
        end_utc="2026-11-25T00:34:45+00:00",
        closest_utc="2026-11-25T00:34:35+00:00",
        classification="TRANSIT",
        crossed_lunar_disk=True,
        min_angular_sep_deg=0.142,
        min_moon_radii=0.506,
        closest_aircraft_alt_deg=45.0,
        closest_aircraft_az_deg=180.05,
        closest_altitude_ft=35000.0,
        closest_slant_range_km=15.1,
        moon_alt_deg=45.0,
        moon_az_deg=180.0,
        moon_angular_diameter_deg=0.561,
        moon_illumination_pct=99.6,
        moon_phase_name="Full Moon",
        track=[
            PassPoint(
                timestamp_utc="2026-11-25T00:34:25+00:00",
                aircraft_alt_deg=44.2,
                aircraft_az_deg=176.0,
                moon_alt_deg=45.0,
                moon_az_deg=180.0,
                moon_angular_radius_deg=0.2805,
                angular_sep_deg=2.94,
                altitude_ft=35000.0,
                slant_range_km=15.4,
                ground_distance_km=10.9,
            ),
            PassPoint(
                timestamp_utc="2026-11-25T00:34:35+00:00",
                aircraft_alt_deg=45.0,
                aircraft_az_deg=180.05,
                moon_alt_deg=45.0,
                moon_az_deg=180.0,
                moon_angular_radius_deg=0.2805,
                angular_sep_deg=0.142,
                altitude_ft=35000.0,
                slant_range_km=15.1,
                ground_distance_km=10.7,
                interpolated=True,
            ),
            PassPoint(
                timestamp_utc="2026-11-25T00:34:45+00:00",
                aircraft_alt_deg=45.8,
                aircraft_az_deg=184.0,
                moon_alt_deg=45.0,
                moon_az_deg=180.0,
                moon_angular_radius_deg=0.2805,
                angular_sep_deg=2.92,
                altitude_ft=35000.0,
                slant_range_km=15.3,
                ground_distance_km=10.8,
            ),
        ],
    )


def test_pass_record_jsonl_round_trip(tmp_path) -> None:
    passes_path = tmp_path / "passes.jsonl"
    notifier = TransitNotifier(log_file=None, passes_file=passes_path, notify_macos=False)
    rec = _sample_pass_record()
    notifier.write_pass(rec)

    loaded = load_passes(passes_path)
    assert len(loaded) == 1
    assert loaded[0].pass_id == rec.pass_id
    assert loaded[0].flight == "KLM999"
    assert loaded[0].crossed_lunar_disk is True
    assert len(loaded[0].track) == 3
    assert loaded[0].track[1].interpolated is True


def test_render_pass_svg_and_html() -> None:
    rec = _sample_pass_record()
    svg = render_pass_svg(rec)
    assert "<svg" in svg
    assert "HORIZON (0°)" in svg
    assert "<polyline" in svg
    assert "KLM999" in svg
    assert "Zoomed Lunar View" in svg

    html_doc = render_pass_html(rec)
    assert "<!DOCTYPE html>" in html_doc
    assert "Download SVG" in html_doc
    assert "KLM999" in html_doc


def test_render_live_snapshot_svg() -> None:
    now = datetime(2026, 11, 25, 0, 35, 0, tzinfo=timezone.utc)
    moon = MoonPosition(
        timestamp_utc=now,
        altitude_deg=45.0,
        azimuth_deg=180.0,
        distance_km=355000.0,
        angular_radius_deg=0.28,
        angular_diameter_deg=0.56,
        illumination_pct=99.6,
        phase_deg=180.0,
        phase_name="Full Moon",
        is_above_horizon=True,
    )
    ac = Aircraft(
        hex="484161",
        flight="KLM999",
        registration=None,
        aircraft_type="B77W",
        category="A5",
        lat=52.27,
        lon=4.89,
        altitude_m=10668.0,
        altitude_ft=35000.0,
        altitude_source="geom",
        ground_speed_kt=450.0,
        track_deg=90.0,
        vertical_rate_fpm=0.0,
        seen_pos_s=0.0,
        squawk=None,
        rssi=None,
    )
    sighting = AircraftSighting(
        aircraft=ac,
        sky_pos=SkyPosition(45.0, 179.0, 15.1, 10.7),
        angular_sep_deg=0.71,
        moon_radii=2.5,
        plane_angular_size_deg=0.25,
        status="INCOMING",
        min_predicted_sep_deg=0.12,
        time_to_closest_s=10.0,
        predicted_transit_duration_s=0.6,
        is_disk_crossing=False,
    )
    cfg = Config()
    svg = render_live_snapshot_svg(moon, [sighting], {"484161": _sample_pass_record().track}, cfg)
    assert "<svg" in svg
    assert "KLM999" in svg
    assert "HORIZON (0°)" in svg
    assert "+120s forecast" in svg

    svg_90 = render_live_snapshot_svg(
        moon, [sighting], {"484161": _sample_pass_record().track}, Config(predict_horizon_s=90.0)
    )
    assert "+90s forecast" in svg_90


def test_web_server_endpoints(tmp_path) -> None:
    passes_path = tmp_path / "passes.jsonl"
    notifier = TransitNotifier(log_file=None, passes_file=passes_path, notify_macos=False)
    rec = _sample_pass_record()
    notifier.write_pass(rec)

    cfg = Config(passes_file=str(passes_path))
    live_state = LiveState()
    now = datetime(2026, 11, 25, 0, 35, 0, tzinfo=timezone.utc)
    moon = MoonPosition(
        timestamp_utc=now,
        altitude_deg=45.0,
        azimuth_deg=180.0,
        distance_km=355000.0,
        angular_radius_deg=0.28,
        angular_diameter_deg=0.56,
        illumination_pct=99.6,
        phase_deg=180.0,
        phase_name="Full Moon",
        is_above_horizon=True,
    )
    live_state.update(moon=moon, sightings=[], tracks={}, total_raw=1, feed_ok=True)

    server = LunaCrossWebServer("127.0.0.1", 0, cfg, live_state)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        r_idx = httpx.get(f"{base}/")
        assert r_idx.status_code == 200
        assert "LunaCross" in r_idx.text
        assert "passes-pagination" in r_idx.text
        assert "tr.pass-row.selected" in r_idx.text

        r_live = httpx.get(f"{base}/api/live.json")
        assert r_live.status_code == 200
        assert r_live.json()["moon"]["phase_name"] == "Full Moon"

        r_lsvg = httpx.get(f"{base}/api/live.svg")
        assert r_lsvg.status_code == 200
        assert "<svg" in r_lsvg.text

        r_list = httpx.get(f"{base}/api/passes")
        assert r_list.status_code == 200
        assert r_list.json()[0]["pass_id"] == rec.pass_id

        r_psvg = httpx.get(f"{base}/api/passes/{rec.pass_id}.svg")
        assert r_psvg.status_code == 200
        assert "KLM999" in r_psvg.text
    finally:
        server.stop()
