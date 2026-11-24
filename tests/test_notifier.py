"""Unit tests for TransitNotifier (JSONL logging and Pushover)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from lunacross.adsb import Aircraft
from lunacross.detector import AircraftSighting, EventType, SightingStatus, TransitEvent
from lunacross.geometry import SkyPosition
from lunacross.moon import MoonPosition
from lunacross.notifier import PUSHOVER_API_URL, TransitNotifier, build_tar1090_aircraft_url


def _sample_event(event_type: EventType = "INCOMING") -> TransitEvent:
    now = datetime(2026, 11, 24, 22, 55, 0, tzinfo=timezone.utc)
    moon = MoonPosition(
        timestamp_utc=now,
        altitude_deg=57.0,
        azimuth_deg=132.5,
        distance_km=354800.0,
        angular_radius_deg=0.28,
        angular_diameter_deg=0.56,
        illumination_pct=99.7,
        phase_deg=180.0,
        phase_name="Full Moon",
        is_above_horizon=True,
    )
    ac = Aircraft(
        hex="484161",
        flight="KLM999",
        registration="PH-BVA",
        aircraft_type="B77W",
        category="A5",
        lat=51.9,
        lon=6.0,
        altitude_m=10668.0,
        altitude_ft=35000.0,
        altitude_source="geom",
        ground_speed_kt=450.0,
        track_deg=90.0,
        vertical_rate_fpm=0.0,
        seen_pos_s=0.0,
        squawk="1000",
        rssi=-12.0,
    )
    status: SightingStatus = event_type
    sighting = AircraftSighting(
        aircraft=ac,
        sky_pos=SkyPosition(
            altitude_deg=56.5,
            azimuth_deg=132.0,
            slant_range_km=12.8,
            ground_distance_km=7.1,
        ),
        angular_sep_deg=12.5 if event_type == "INCOMING" else 0.12,
        moon_radii=44.6 if event_type == "INCOMING" else 0.43,
        plane_angular_size_deg=0.29,
        status=status,
        min_predicted_sep_deg=0.14 if event_type == "INCOMING" else None,
        time_to_closest_s=38.0 if event_type == "INCOMING" else None,
        predicted_transit_duration_s=0.6 if event_type == "INCOMING" else None,
        is_disk_crossing=(event_type == "TRANSIT"),
    )
    return TransitEvent(
        event_type=event_type,
        timestamp_utc=now,
        sighting=sighting,
        moon=moon,
    )


class _DummyResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"status": 1, "request": "req-123"}


def test_build_tar1090_aircraft_url() -> None:
    assert (
        build_tar1090_aircraft_url("http://adsb.local:8090/data/aircraft.json", "484161")
        == "http://adsb.local:8090/?icao=484161"
    )
    assert build_tar1090_aircraft_url(None, "484161") is None


def test_pushover_sends_incoming_and_transit_and_filters_near_miss(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_post(url: str, *, data: dict[str, str] | None = None, **kwargs: Any) -> _DummyResponse:
        calls.append((url, dict(data or {})))
        return _DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    log_path = tmp_path / "transits.jsonl"
    notifier = TransitNotifier(
        log_file=log_path,
        notify_macos=False,
        pushover_app_token="app-tok-123",
        pushover_user_key="usr-key-456",
        pushover_device="iphone",
        pushover_sound="cosmic",
        pushover_priority=1,
        pushover_events="INCOMING,TRANSIT",
        adsb_url="http://adsb.local:8090/data/aircraft.json",
    )

    notifier.notify(_sample_event("INCOMING"))
    notifier.notify(_sample_event("NEAR_MISS"))
    notifier.notify(_sample_event("TRANSIT"))

    assert len(calls) == 2

    url1, data1 = calls[0]
    assert url1 == PUSHOVER_API_URL
    assert data1["token"] == "app-tok-123"
    assert data1["user"] == "usr-key-456"
    assert data1["device"] == "iphone"
    assert data1["sound"] == "cosmic"
    assert data1["priority"] == "1"
    assert data1["url"] == "http://adsb.local:8090/?icao=484161"
    assert "Lunar transit in 38s: KLM999" in data1["title"]
    assert "approaching the Moon" in data1["message"]
    assert "Crossing lunar disk" in data1["message"]

    _, data2 = calls[1]
    assert "Crossing Moon NOW: KLM999" in data2["title"]
    assert "crossing in front of the Moon now" in data2["message"]

    # All 3 events (including NEAR_MISS) are still written to JSONL
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed_types = [json.loads(line)["event_type"] for line in lines]
    assert parsed_types == ["INCOMING", "NEAR_MISS", "TRANSIT"]


def test_pushover_skipped_without_credentials(monkeypatch, tmp_path) -> None:
    called = False

    def fake_post(*args: Any, **kwargs: Any) -> _DummyResponse:
        nonlocal called
        called = True
        return _DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    notifier = TransitNotifier(log_file=tmp_path / "transits.jsonl", notify_macos=False)
    assert notifier.pushover_enabled is False
    notifier.notify(_sample_event("INCOMING"))
    assert called is False
