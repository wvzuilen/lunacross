"""Unit tests for Open-Meteo weather integration, live refraction, QNH fallback, quiet hours, and caching."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from lunacross.adsb import Aircraft, parse_aircraft_payload
from lunacross.config import Config
from lunacross.detector import AircraftSighting, PassPoint, PassRecord, TransitEvent
from lunacross.geometry import SkyPosition, aircraft_altaz
from lunacross.moon import MoonPosition, MoonTracker
from lunacross import notifier as notifier_mod
from lunacross.notifier import TransitNotifier, is_in_quiet_hours
from lunacross.visualize import render_pass_svg
from lunacross.weather import WeatherClient, parse_open_meteo_payload
from lunacross.web import LiveState, LunaCrossWebServer


def test_parse_open_meteo_overcast_cirrus_and_fog() -> None:
    now = datetime(2026, 11, 27, 19, 0, 0, tzinfo=timezone.utc)

    # Low stratus overcast -> 100% obstructed
    overcast = parse_open_meteo_payload(
        {
            "current": {
                "temperature_2m": 0.6,
                "pressure_msl": 1027.9,
                "surface_pressure": 1025.4,
                "cloud_cover": 100,
                "cloud_cover_low": 100,
                "cloud_cover_mid": 0,
                "cloud_cover_high": 1,
                "visibility": 50000.0,
                "weather_code": 3,
            }
        },
        fetched_utc=now,
    )
    assert overcast.sky_obstruction_pct == 100.0
    assert "Overcast" in overcast.condition_label
    assert overcast.pressure_msl_hpa == pytest.approx(1027.9)

    # Pure high thin cirrus -> translucent (45% obstruction), should NOT block 85% threshold
    cirrus = parse_open_meteo_payload(
        {
            "current": {
                "temperature_2m": -2.0,
                "pressure_msl": 1020.0,
                "surface_pressure": 1018.0,
                "cloud_cover": 90,
                "cloud_cover_low": 0,
                "cloud_cover_mid": 0,
                "cloud_cover_high": 90,
                "visibility": 35000.0,
                "weather_code": 2,
            }
        },
        fetched_utc=now,
    )
    assert cirrus.sky_obstruction_pct == pytest.approx(45.0)
    assert "Thin high clouds" in cirrus.condition_label

    # Dense fog -> >= 95% obstructed even when model cloud layers are low
    fog = parse_open_meteo_payload(
        {
            "current": {
                "temperature_2m": 1.0,
                "pressure_msl": 1015.0,
                "surface_pressure": 1014.0,
                "cloud_cover": 10,
                "cloud_cover_low": 10,
                "cloud_cover_mid": 0,
                "cloud_cover_high": 0,
                "visibility": 600.0,
                "weather_code": 45,
            }
        },
        fetched_utc=now,
    )
    assert fog.sky_obstruction_pct >= 95.0
    assert "Fog" in fog.condition_label


def test_weather_client_caches_requests(monkeypatch) -> None:
    call_count = 0

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "current": {
                    "temperature_2m": 4.5,
                    "pressure_msl": 1019.0,
                    "surface_pressure": 1017.0,
                    "cloud_cover": 12,
                    "cloud_cover_low": 5,
                    "cloud_cover_mid": 7,
                    "cloud_cover_high": 0,
                    "visibility": 30000.0,
                    "weather_code": 1,
                }
            }

    def fake_get(self: Any, url: str, **kwargs: Any) -> _FakeResp:
        nonlocal call_count
        call_count += 1
        return _FakeResp()

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    with WeatherClient(52.3731, 4.8926, enabled=True, refresh_interval_s=600.0) as client:
        wx1 = client.get_weather()
        wx2 = client.get_weather()
        assert wx1 is not None
        assert wx2 is wx1
        assert call_count == 1

    with WeatherClient(52.3731, 4.8926, enabled=False) as disabled:
        assert disabled.get_weather() is None
        assert call_count == 1


def test_live_refraction_and_fallback_qnh() -> None:
    # Low-elevation aircraft (~2° above horizon): cold high-pressure air bends rays upward more than warm low-pressure air
    cold_high = aircraft_altaz(
        52.3731,
        4.8926,
        10.0,
        54.0,
        4.8926,
        11000.0,
        temperature_c=-10.0,
        pressure_hpa=1040.0,
    )
    warm_low = aircraft_altaz(
        52.3731,
        4.8926,
        10.0,
        54.0,
        4.8926,
        11000.0,
        temperature_c=30.0,
        pressure_hpa=985.0,
    )
    assert cold_high.altitude_deg > warm_low.altitude_deg + 0.03

    tracker = MoonTracker(52.3731, 4.8926, elevation_m=2.0, ephemeris_dir=".cache/skyfield")
    dt = datetime(2026, 11, 25, 15, 30, 0, tzinfo=timezone.utc)
    m_cold = tracker.get_position(dt, temperature_c=-10.0, pressure_hpa=1040.0)
    m_warm = tracker.get_position(dt, temperature_c=30.0, pressure_hpa=985.0)
    if m_cold.altitude_deg > 0.0:
        assert m_cold.altitude_deg > m_warm.altitude_deg

    # Fallback QNH from weather when aircraft only sends alt_baro without nav_qnh
    snap = parse_aircraft_payload(
        {
            "now": 1700000000.0,
            "aircraft": [
                {"hex": "aaa111", "lat": 52.1, "lon": 5.1, "alt_baro": 10000},
                {"hex": "bbb222", "lat": 52.1, "lon": 5.1, "alt_baro": 10000, "nav_qnh": 1000.0},
            ],
        },
        geoid_undulation_m=44.0,
        fallback_qnh_hpa=1030.0,
    )
    by_hex = {ac.hex: ac for ac in snap.aircraft}
    # aaa111 uses fallback_qnh_hpa=1030.0 (> 1013.25), whereas bbb222 uses its own nav_qnh=1000.0 (< 1013.25)
    assert by_hex["aaa111"].altitude_m > by_hex["bbb222"].altitude_m + 200.0


def test_quiet_hours_and_cloud_cover_pushover_suppression(monkeypatch, tmp_path) -> None:
    dt_02 = datetime(2026, 11, 27, 2, 15, 0, tzinfo=timezone.utc)
    dt_20 = datetime(2026, 11, 27, 20, 15, 0, tzinfo=timezone.utc)
    assert is_in_quiet_hours(dt_02, "23:30-07:00") is True
    assert is_in_quiet_hours(dt_20, "23:30-07:00") is False
    assert is_in_quiet_hours(dt_20, "19:00-22:00") is True
    assert is_in_quiet_hours(dt_02, None) is False

    calls: list[dict[str, str]] = []

    class _DummyResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"status": 1}

    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, *, data=None, **kw: (calls.append(dict(data or {})) or _DummyResp()),
    )

    moon = MoonPosition(
        timestamp_utc=dt_20,
        altitude_deg=35.0,
        azimuth_deg=140.0,
        distance_km=360000.0,
        angular_radius_deg=0.275,
        angular_diameter_deg=0.55,
        illumination_pct=85.0,
        phase_deg=140.0,
        phase_name="Waxing Gibbous",
        is_above_horizon=True,
    )
    ac = Aircraft(
        hex="484161",
        flight="KLM999",
        registration="PH-BVA",
        aircraft_type="B77W",
        category="A5",
        lat=52.0,
        lon=5.0,
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
        sky_pos=SkyPosition(35.0, 140.0, 18.0, 14.5),
        angular_sep_deg=0.10,
        moon_radii=0.36,
        plane_angular_size_deg=0.20,
        status="TRANSIT",
        min_predicted_sep_deg=None,
        time_to_closest_s=None,
        predicted_transit_duration_s=None,
        is_disk_crossing=True,
    )
    ev = TransitEvent("TRANSIT", dt_20, sighting, moon)

    log_path = tmp_path / "transits.jsonl"
    notifier = TransitNotifier(
        log_file=log_path,
        notify_macos=False,
        pushover_app_token="tok",
        pushover_user_key="usr",
        pushover_max_cloud_cover_pct=85.0,
        pushover_quiet_hours="23:30-06:00",
    )

    wx_overcast = parse_open_meteo_payload(
        {
            "current": {
                "temperature_2m": 1.2,
                "pressure_msl": 1025.0,
                "surface_pressure": 1023.0,
                "cloud_cover": 100,
                "cloud_cover_low": 100,
                "cloud_cover_mid": 0,
                "cloud_cover_high": 0,
                "weather_code": 3,
            }
        },
        fetched_utc=dt_20,
    )
    notifier.notify(ev, weather=wx_overcast)
    assert calls == []

    wx_clear = parse_open_meteo_payload(
        {
            "current": {
                "temperature_2m": 2.5,
                "pressure_msl": 1025.0,
                "surface_pressure": 1023.0,
                "cloud_cover": 8,
                "cloud_cover_low": 5,
                "cloud_cover_mid": 3,
                "cloud_cover_high": 0,
                "weather_code": 0,
            }
        },
        fetched_utc=dt_20,
    )
    notifier.notify(ev, weather=wx_clear)
    assert len(calls) == 1
    assert "Sky: Clear (8% clouds)" in calls[0]["message"]

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["pushover_suppressed_reason"] == "cloud_cover"
    assert "pushover_suppressed_reason" not in rows[1]


def test_legacy_moon_phase_reconstruction_and_web_pass_cache(monkeypatch, tmp_path) -> None:
    # Legacy PassRecord dict without moon_phase_deg (Waning Crescent, 15% lit -> phase > 270°)
    legacy_dict = {
        "pass_id": "20261125T010000Z_484161",
        "flight": "KLM123",
        "hex": "484161",
        "registration": None,
        "aircraft_type": "B738",
        "category": "A3",
        "observer_lat": 52.3731,
        "observer_lon": 4.8926,
        "observer_elevation_m": 2.0,
        "start_utc": "2026-11-25T00:59:50+00:00",
        "end_utc": "2026-11-25T01:00:10+00:00",
        "closest_utc": "2026-11-25T01:00:00+00:00",
        "classification": "TRANSIT",
        "crossed_lunar_disk": True,
        "min_angular_sep_deg": 0.12,
        "min_moon_radii": 0.43,
        "closest_aircraft_alt_deg": 30.0,
        "closest_aircraft_az_deg": 150.0,
        "closest_altitude_ft": 34000.0,
        "closest_slant_range_km": 20.0,
        "moon_alt_deg": 30.0,
        "moon_az_deg": 150.0,
        "moon_angular_diameter_deg": 0.55,
        "moon_illumination_pct": 15.0,
        "moon_phase_name": "Waning Crescent",
        "track": [
            PassPoint(
                timestamp_utc="2026-11-25T01:00:00+00:00",
                aircraft_alt_deg=30.0,
                aircraft_az_deg=150.0,
                moon_alt_deg=30.0,
                moon_az_deg=150.0,
                moon_angular_radius_deg=0.275,
                angular_sep_deg=0.12,
                altitude_ft=34000.0,
                slant_range_km=20.0,
                ground_distance_km=17.3,
            ).to_dict()
        ],
    }
    rec = PassRecord.from_dict(legacy_dict)
    assert 290.0 <= rec.moon_phase_deg <= 330.0
    svg = render_pass_svg(rec)
    assert "<path d=" in svg  # Rendered as true crescent path, not full circle!

    # Verify _PassCache only reads passes.jsonl once across multiple HTTP requests until modified
    passes_path = tmp_path / "passes.jsonl"
    passes_path.write_text(json.dumps(rec.to_dict()) + "\n", encoding="utf-8")

    load_calls = 0
    orig_load = notifier_mod.load_passes

    def counting_load(p: Any) -> list[PassRecord]:
        nonlocal load_calls
        load_calls += 1
        return orig_load(p)

    import lunacross.web as web_mod

    monkeypatch.setattr(web_mod, "load_passes", counting_load)

    server = LunaCrossWebServer("127.0.0.1", 0, Config(passes_file=str(passes_path)), LiveState())
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        assert httpx.get(f"{base}/api/passes").status_code == 200
        assert httpx.get(f"{base}/api/passes").status_code == 200
        assert httpx.get(f"{base}/api/passes/{rec.pass_id}.svg").status_code == 200
        assert httpx.get(f"{base}/api/passes/{rec.pass_id}.json").status_code == 200
        assert load_calls == 1
    finally:
        server.stop()
