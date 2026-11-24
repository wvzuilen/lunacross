"""Transitdetectie en baanvoorspelling voor vliegtuigen langs de maan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from lunacross.adsb import Aircraft
from lunacross.geometry import (
    SkyPosition,
    aircraft_altaz,
    angular_separation_deg,
    estimate_aircraft_angular_size_deg,
    extrapolate_position,
)
from lunacross.moon import MoonPosition

SightingStatus = Literal["TRANSIT", "NEAR_MISS", "INCOMING", "TRACKING"]
EventType = Literal["TRANSIT", "NEAR_MISS", "INCOMING"]

_STATUS_PRIORITY: dict[SightingStatus, int] = {
    "TRANSIT": 0,
    "NEAR_MISS": 1,
    "INCOMING": 2,
    "TRACKING": 3,
}

# Hogere ernst mag de cooldown voor een lagere ernst meteen doorbreken
_EVENT_SEVERITY: dict[EventType, int] = {
    "INCOMING": 1,
    "NEAR_MISS": 2,
    "TRANSIT": 3,
}


@dataclass(frozen=True, slots=True)
class AircraftSighting:
    """Evaluatie van één vliegtuig ten opzichte van de maan."""

    aircraft: Aircraft
    sky_pos: SkyPosition
    angular_sep_deg: float
    moon_radii: float
    plane_angular_size_deg: float
    status: SightingStatus
    min_predicted_sep_deg: float | None
    time_to_closest_s: float | None
    predicted_transit_duration_s: float | None
    is_disk_crossing: bool


@dataclass(frozen=True, slots=True)
class TransitEvent:
    """Gebeurtenis om te loggen en/of een melding voor te sturen."""

    event_type: EventType
    timestamp_utc: datetime
    sighting: AircraftSighting
    moon: MoonPosition


class TransitDetector:
    """Detecteert directe maanovergangen, near-misses en naderende transits."""

    def __init__(
        self,
        obs_lat: float,
        obs_lon: float,
        obs_alt_m: float = 0.0,
        alert_margin_deg: float = 0.5,
        predict_horizon_s: float = 120.0,
        min_moon_alt_deg: float = 0.0,
        cooldown_s: float = 180.0,
    ) -> None:
        self.obs_lat = obs_lat
        self.obs_lon = obs_lon
        self.obs_alt_m = obs_alt_m
        self.alert_margin_deg = alert_margin_deg
        self.predict_horizon_s = predict_horizon_s
        self.min_moon_alt_deg = min_moon_alt_deg
        self.cooldown_s = cooldown_s

        # Per vliegtuig-hex: (laatste tijdstip, laatste ernst)
        self._last_alerts: dict[str, tuple[datetime, int]] = {}

    def _current_aircraft_coords(self, ac: Aircraft) -> tuple[float, float, float]:
        dt_dead_reckon = max(0.0, min(ac.seen_pos_s, 30.0))
        if (
            dt_dead_reckon > 0.0
            and ac.ground_speed_kt is not None
            and ac.track_deg is not None
            and ac.ground_speed_kt > 5.0
        ):
            return extrapolate_position(
                ac.lat,
                ac.lon,
                ac.altitude_m,
                ac.ground_speed_kt,
                ac.track_deg,
                ac.vertical_rate_fpm,
                dt_dead_reckon,
            )
        return ac.lat, ac.lon, ac.altitude_m

    def _predict_closest_approach(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        ac: Aircraft,
        moon: MoonPosition,
        disk_threshold_deg: float,
        alert_threshold_deg: float,
    ) -> tuple[float | None, float | None, float | None]:
        if (
            self.predict_horizon_s <= 0.0
            or ac.ground_speed_kt is None
            or ac.track_deg is None
            or ac.ground_speed_kt <= 20.0
        ):
            return None, None, None

        def eval_sep(t_s: float) -> tuple[float, float]:
            plat, plon, palt = extrapolate_position(
                lat,
                lon,
                alt_m,
                ac.ground_speed_kt,  # type: ignore[arg-type]
                ac.track_deg,  # type: ignore[arg-type]
                ac.vertical_rate_fpm,
                t_s,
            )
            sky = aircraft_altaz(self.obs_lat, self.obs_lon, self.obs_alt_m, plat, plon, palt)
            sep = angular_separation_deg(
                sky.altitude_deg,
                sky.azimuth_deg,
                moon.altitude_deg,
                moon.azimuth_deg,
            )
            return sep, sky.altitude_deg

        # Grove scan om de 2 seconden
        coarse_step = 2.0
        steps = int(self.predict_horizon_s / coarse_step)
        best_t = 0.0
        best_sep = math.inf
        for i in range(steps + 1):
            t_s = i * coarse_step
            sep, _ = eval_sep(t_s)
            if sep < best_sep:
                best_sep = sep
                best_t = t_s

        # Verfijn rondom het minimum met stappen van 0.1s
        win_start = max(0.0, best_t - coarse_step)
        win_end = min(self.predict_horizon_s, best_t + coarse_step)
        fine_step = 0.1
        fine_count = int(round((win_end - win_start) / fine_step))

        fine_best_t = best_t
        fine_best_sep = math.inf
        fine_best_alt = -90.0
        transit_samples = 0

        for i in range(fine_count + 1):
            t_s = win_start + i * fine_step
            sep, sky_alt = eval_sep(t_s)
            if sep < fine_best_sep:
                fine_best_sep = sep
                fine_best_t = t_s
                fine_best_alt = sky_alt
            if sky_alt > 0.0 and sep <= disk_threshold_deg:
                transit_samples += 1

        if fine_best_t <= 0.2 or fine_best_alt <= 0.0 or fine_best_sep > alert_threshold_deg:
            return None, None, None

        duration_s = round(transit_samples * fine_step, 1) if transit_samples > 0 else None
        return fine_best_sep, fine_best_t, duration_s

    def _should_emit(self, hex_code: str, event_type: EventType, now_utc: datetime) -> bool:
        severity = _EVENT_SEVERITY[event_type]
        prev = self._last_alerts.get(hex_code)
        if prev is not None:
            prev_time, prev_severity = prev
            elapsed = (now_utc - prev_time).total_seconds()
            if elapsed < self.cooldown_s and severity <= prev_severity:
                return False
        self._last_alerts[hex_code] = (now_utc, severity)
        return True

    def evaluate(
        self,
        moon: MoonPosition,
        aircraft_list: list[Aircraft],
        now_utc: datetime | None = None,
    ) -> tuple[list[AircraftSighting], list[TransitEvent]]:
        """Beoordeel alle vliegtuigen t.o.v. de maan en retourneer gesorteerde waarnemingen + events."""
        current_utc = now_utc or datetime.now(timezone.utc)
        if current_utc.tzinfo is None:
            current_utc = current_utc.replace(tzinfo=timezone.utc)

        moon_eligible = moon.altitude_deg >= self.min_moon_alt_deg
        sightings: list[AircraftSighting] = []
        events: list[TransitEvent] = []

        for ac in aircraft_list:
            cur_lat, cur_lon, cur_alt_m = self._current_aircraft_coords(ac)
            sky_pos = aircraft_altaz(
                self.obs_lat,
                self.obs_lon,
                self.obs_alt_m,
                cur_lat,
                cur_lon,
                cur_alt_m,
            )
            sep_deg = angular_separation_deg(
                sky_pos.altitude_deg,
                sky_pos.azimuth_deg,
                moon.altitude_deg,
                moon.azimuth_deg,
            )
            plane_ang = estimate_aircraft_angular_size_deg(sky_pos.slant_range_km, ac.category)
            disk_threshold = moon.angular_radius_deg + plane_ang / 2.0
            alert_threshold = moon.angular_radius_deg + self.alert_margin_deg
            moon_radii = sep_deg / moon.angular_radius_deg if moon.angular_radius_deg > 0 else math.inf
            is_disk_crossing = moon_eligible and sky_pos.altitude_deg > 0.0 and sep_deg <= disk_threshold

            min_pred_sep: float | None = None
            tca_s: float | None = None
            duration_s: float | None = None
            if moon_eligible:
                min_pred_sep, tca_s, duration_s = self._predict_closest_approach(
                    cur_lat,
                    cur_lon,
                    cur_alt_m,
                    ac,
                    moon,
                    disk_threshold,
                    alert_threshold,
                )

            status: SightingStatus = "TRACKING"
            if moon_eligible and sky_pos.altitude_deg > 0.0:
                if sep_deg <= disk_threshold:
                    status = "TRANSIT"
                elif sep_deg <= alert_threshold:
                    status = "NEAR_MISS"
                elif min_pred_sep is not None and tca_s is not None and tca_s > 0.0:
                    status = "INCOMING"

            sighting = AircraftSighting(
                aircraft=ac,
                sky_pos=sky_pos,
                angular_sep_deg=sep_deg,
                moon_radii=moon_radii,
                plane_angular_size_deg=plane_ang,
                status=status,
                min_predicted_sep_deg=min_pred_sep,
                time_to_closest_s=tca_s,
                predicted_transit_duration_s=duration_s,
                is_disk_crossing=is_disk_crossing,
            )
            sightings.append(sighting)

            if status in ("TRANSIT", "NEAR_MISS", "INCOMING"):
                event_type: EventType = status  # type: ignore[assignment]
                if self._should_emit(ac.hex, event_type, current_utc):
                    events.append(
                        TransitEvent(
                            event_type=event_type,
                            timestamp_utc=current_utc,
                            sighting=sighting,
                            moon=moon,
                        )
                    )

        sightings.sort(
            key=lambda s: (
                _STATUS_PRIORITY[s.status],
                min(
                    s.angular_sep_deg,
                    s.min_predicted_sep_deg if s.min_predicted_sep_deg is not None else math.inf,
                ),
            )
        )
        return sightings, events
