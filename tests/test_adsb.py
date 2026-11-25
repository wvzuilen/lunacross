"""Unit tests for ADS-B JSON parsing."""

from __future__ import annotations

import pytest

from lunacross.adsb import merge_snapshots, parse_aircraft_payload


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


def test_parse_community_v2_payload_and_merge() -> None:
    local_payload = {
        "now": 1795556720.0,
        "aircraft": [
            {
                "hex": "407567",
                "flight": "EZY98ZW ",
                "alt_geom": 35650,
                "gs": 388.3,
                "track": 274.73,
                "lat": 51.827584,
                "lon": 5.275955,
                "seen_pos": 1.0,
            }
        ],
    }
    community_payload = {
        "now": 1795556721000.0,
        "ac": [
            {
                "hex": "407567",
                "flight": "EZY98ZW ",
                "r": "G-UZHH",
                "t": "A20N",
                "category": "A3",
                "alt_geom": 35600,
                "lat": 51.82,
                "lon": 5.27,
                "seen_pos": 2.0,
            },
            {
                "hex": "486482",
                "flight": "KLC52L  ",
                "r": "PH-NXE",
                "t": "E295",
                "category": "A3",
                "alt_geom": 5175,
                "gs": 228.0,
                "track": 57.36,
                "lat": 52.133881,
                "lon": 4.336548,
                "seen_pos": 0.5,
            },
            {
                "hex": "999999",
                "flight": "STALE1",
                "alt_geom": 10000,
                "lat": 52.0,
                "lon": 5.0,
                "seen_pos": 28.0,
            },
        ],
    }

    primary = parse_aircraft_payload(local_payload, max_seen_pos_s=30.0, source="local")
    supp = parse_aircraft_payload(community_payload, max_seen_pos_s=30.0, source="supplement")
    assert supp.timestamp_unix == pytest.approx(1795556721.0)
    assert len(supp.aircraft) == 3

    merged = merge_snapshots(primary, supp, supplement_age_s=4.0, max_seen_pos_s=30.0)
    # 407567 keeps local lat/lon/alt + gains r/t/category; 486482 added aged by 4s; 999999 dropped (28+4 > 30)
    assert len(merged.aircraft) == 2
    by_hex = {a.hex: a for a in merged.aircraft}

    local_ac = by_hex["407567"]
    assert local_ac.source == "local"
    assert local_ac.lat == pytest.approx(51.827584)
    assert local_ac.registration == "G-UZHH"
    assert local_ac.aircraft_type == "A20N"
    assert local_ac.category == "A3"

    supp_ac = by_hex["486482"]
    assert supp_ac.source == "supplement"
    assert supp_ac.seen_pos_s == pytest.approx(4.5)
    assert supp_ac.aircraft_type == "E295"
