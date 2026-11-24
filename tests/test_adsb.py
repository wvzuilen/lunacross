"""Unit tests voor ADS-B JSON-parsing."""

from __future__ import annotations

import pytest

from lunacross.adsb import parse_aircraft_payload


def test_parse_aircraft_payload_filters_and_prefers_geom() -> None:
    payload = {
        "now": 1795556720.0,
        "aircraft": [
            {
                "hex": "407567",
                "flight": "EZY98ZW ",
                "r": "G-UZHH",
                "t": "A20N",
                "category": "A3",
                "alt_baro": 36000,
                "alt_geom": 35650,
                "gs": 388.3,
                "track": 274.73,
                "geom_rate": 64,
                "lat": 51.827584,
                "lon": 5.275955,
                "seen_pos": 11.9,
                "squawk": "3220",
                "rssi": -36.6,
            },
            {
                "hex": "abcdef",
                "flight": "   ",
                "alt_baro": 24000,
                "lat": 52.1,
                "lon": 6.1,
                "seen_pos": 5.0,
            },
            {
                "hex": "111111",
                "alt_baro": "ground",
                "lat": 52.0,
                "lon": 5.9,
                "seen_pos": 1.0,
            },
            {
                "hex": "222222",
                "alt_baro": 10000,
                "seen_pos": 2.0,
            },
            {
                "hex": "333333",
                "alt_baro": 30000,
                "lat": 52.0,
                "lon": 5.9,
                "seen_pos": 45.0,
            },
        ],
    }

    snap = parse_aircraft_payload(payload, max_seen_pos_s=30.0)
    assert snap.total_raw == 5
    assert snap.timestamp_unix == 1795556720.0
    assert len(snap.aircraft) == 2

    first = snap.aircraft[0]
    assert first.flight == "EZY98ZW"
    assert first.altitude_ft == 35650.0
    assert first.altitude_source == "geom"
    assert first.altitude_m == pytest.approx(35650.0 * 0.3048)
    assert first.vertical_rate_fpm == 64.0

    second = snap.aircraft[1]
    assert second.flight == "ABCDEF"
    assert second.altitude_ft == 24000.0
    assert second.altitude_source == "baro"
