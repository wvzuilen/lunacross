"""Lunar transit detection, sub-second interpolation, pass recording, and trajectory prediction."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from lunacross.adsb import Aircraft
from lunacross.geometry import (
    SkyPosition,
    aircraft_altaz,
    angular_separation_deg,
    estimate_aircraft_angular_size_deg,
    extrapolate_position,
    segment_closest_approach,
)
from lunacross.moon import MoonPosition

SightingStatus = Literal["TRANSIT", "NEAR_MISS", "INCOMING", "TRACKING"]
EventType = Literal["TRANSIT", "NEAR_MISS", "INCOMING"]
PassClassification = Literal["TRANSIT", "NEAR_MISS", "CLOSE_PASS"]

_STATUS_PRIORITY: dict[SightingStatus, int] = {
    "TRANSIT": 0,
    "NEAR_MISS": 1,
    "INCOMING": 2,
    "TRACKING": 3,
}

# Higher-severity statuses bypass the cooldown of lower-severity alerts immediately
_EVENT_SEVERITY: dict[EventType, int] = {
    "INCOMING": 1,
    "NEAR_MISS": 2,
    "TRANSIT": 3,
}


@dataclass(frozen=True, slots=True)
class PassPoint:
    """Single topocentric sample along an aircraft pass near the Moon."""

    timestamp_utc: str
    aircraft_alt_deg: float
    aircraft_az_deg: float
    moon_alt_deg: float
    moon_az_deg: float
    moon_angular_radius_deg: float
    angular_sep_deg: float
    altitude_ft: float
    slant_range_km: float
    ground_distance_km: float
    interpolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "aircraft_alt_deg": round(self.aircraft_alt_deg, 4),
            "aircraft_az_deg": round(self.aircraft_az_deg, 4),
            "moon_alt_deg": round(self.moon_alt_deg, 4),
            "moon_az_deg": round(self.moon_az_deg, 4),
            "moon_angular_radius_deg": round(self.moon_angular_radius_deg, 4),
            "angular_sep_deg": round(self.angular_sep_deg, 4),
            "altitude_ft": round(self.altitude_ft, 1),
            "slant_range_km": round(self.slant_range_km, 3),
            "ground_distance_km": round(self.ground_distance_km, 3),
            "interpolated": self.interpolated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PassPoint:
        return cls(
            timestamp_utc=str(data["timestamp_utc"]),
            aircraft_alt_deg=float(data["aircraft_alt_deg"]),
            aircraft_az_deg=float(data["aircraft_az_deg"]),
            moon_alt_deg=float(data["moon_alt_deg"]),
            moon_az_deg=float(data["moon_az_deg"]),
            moon_angular_radius_deg=float(data["moon_angular_radius_deg"]),
            angular_sep_deg=float(data["angular_sep_deg"]),
            altitude_ft=float(data["altitude_ft"]),
            slant_range_km=float(data["slant_range_km"]),
            ground_distance_km=float(data["ground_distance_km"]),
            interpolated=bool(data.get("interpolated", False)),
        )


@dataclass(frozen=True, slots=True)
class PassRecord:
    """Complete summary and sky track of a single aircraft pass near the Moon."""

    pass_id: str
    flight: str
    hex: str
    registration: str | None
    aircraft_type: str | None
    category: str | None
    observer_lat: float
    observer_lon: float
    observer_elevation_m: float
    start_utc: str
    end_utc: str
    closest_utc: str
    classification: PassClassification
    crossed_lunar_disk: bool
    min_angular_sep_deg: float
    min_moon_radii: float
    closest_aircraft_alt_deg: float
    closest_aircraft_az_deg: float
    closest_altitude_ft: float
    closest_slant_range_km: float
    moon_alt_deg: float
    moon_az_deg: float
    moon_angular_diameter_deg: float
    moon_illumination_pct: float
    moon_phase_name: str
    track: list[PassPoint]
    moon_phase_deg: float = 180.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "flight": self.flight,
            "hex": self.hex,
            "registration": self.registration,
            "aircraft_type": self.aircraft_type,
            "category": self.category,
            "observer_lat": self.observer_lat,
            "observer_lon": self.observer_lon,
            "observer_elevation_m": self.observer_elevation_m,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "closest_utc": self.closest_utc,
            "classification": self.classification,
            "crossed_lunar_disk": self.crossed_lunar_disk,
            "min_angular_sep_deg": round(self.min_angular_sep_deg, 4),
            "min_moon_radii": round(self.min_moon_radii, 3),
            "closest_aircraft_alt_deg": round(self.closest_aircraft_alt_deg, 4),
            "closest_aircraft_az_deg": round(self.closest_aircraft_az_deg, 4),
            "closest_altitude_ft": round(self.closest_altitude_ft, 1),
            "closest_slant_range_km": round(self.closest_slant_range_km, 3),
            "moon_alt_deg": round(self.moon_alt_deg, 4),
            "moon_az_deg": round(self.moon_az_deg, 4),
            "moon_angular_diameter_deg": round(self.moon_angular_diameter_deg, 4),
            "moon_illumination_pct": round(self.moon_illumination_pct, 1),
            "moon_phase_deg": round(self.moon_phase_deg, 2),
            "moon_phase_name": self.moon_phase_name,
            "track": [pt.to_dict() for pt in self.track],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PassRecord:
        illum = float(data["moon_illumination_pct"])
        raw_phase_deg = data.get("moon_phase_deg")
        if raw_phase_deg is not None:
            phase_deg = float(raw_phase_deg)
        else:
            # Reconstruct approximate phase angle from illumination + phase_name for legacy records
            frac = max(0.0, min(1.0, illum / 100.0))
            # frac = (1 - cos(phase)) / 2  =>  cos(phase) = 1 - 2 * frac
            base_ang = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * frac))))
            phase_name = str(data.get("moon_phase_name", ""))
            phase_deg = (360.0 - base_ang) % 360.0 if "Waning" in phase_name or "Last" in phase_name else base_ang
        return cls(
            pass_id=str(data["pass_id"]),
            flight=str(data["flight"]),
            hex=str(data["hex"]),
            registration=data.get("registration"),
            aircraft_type=data.get("aircraft_type"),
            category=data.get("category"),
            observer_lat=float(data["observer_lat"]),
            observer_lon=float(data["observer_lon"]),
            observer_elevation_m=float(data["observer_elevation_m"]),
            start_utc=str(data["start_utc"]),
            end_utc=str(data["end_utc"]),
            closest_utc=str(data["closest_utc"]),
            classification=data["classification"],
            crossed_lunar_disk=bool(data["crossed_lunar_disk"]),
            min_angular_sep_deg=float(data["min_angular_sep_deg"]),
            min_moon_radii=float(data["min_moon_radii"]),
            closest_aircraft_alt_deg=float(data["closest_aircraft_alt_deg"]),
            closest_aircraft_az_deg=float(data["closest_aircraft_az_deg"]),
            closest_altitude_ft=float(data["closest_altitude_ft"]),
            closest_slant_range_km=float(data["closest_slant_range_km"]),
            moon_alt_deg=float(data["moon_alt_deg"]),
            moon_az_deg=float(data["moon_az_deg"]),
            moon_angular_diameter_deg=float(data["moon_angular_diameter_deg"]),
            moon_illumination_pct=illum,
            moon_phase_name=str(data["moon_phase_name"]),
            track=[PassPoint.from_dict(pt) for pt in data.get("track", [])],
            moon_phase_deg=phase_deg,
        )


@dataclass(frozen=True, slots=True)
class AircraftSighting:
    """Evaluation of a single aircraft relative to the Moon."""

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
    """Alert event emitted for logging and notifications."""

    event_type: EventType
    timestamp_utc: datetime
    sighting: AircraftSighting
    moon: MoonPosition


@dataclass(slots=True)
class _ActivePass:
    """Internal accumulator for an ongoing aircraft pass near the Moon."""

    pass_id: str
    flight: str
    hex: str
    registration: str | None
    aircraft_type: str | None
    category: str | None
    start_utc: datetime
    last_seen_utc: datetime
    closest_utc: datetime
    min_sep_deg: float
    min_moon_radii: float
    crossed_lunar_disk: bool
    classification: PassClassification
    closest_aircraft_alt_deg: float
    closest_aircraft_az_deg: float
    closest_altitude_ft: float
    closest_slant_range_km: float
    moon_alt_deg: float
    moon_az_deg: float
    moon_angular_diameter_deg: float
    moon_illumination_pct: float
    moon_phase_deg: float
    moon_phase_name: str
    track: list[PassPoint]
    outside_count: int = 0
    outside_since_utc: datetime | None = None


class TransitDetector:
    """Detects live lunar transits, near misses, incoming passes, and logs close tracks."""

    def __init__(
        self,
        obs_lat: float,
        obs_lon: float,
        obs_alt_m: float = 0.0,
        alert_margin_deg: float = 0.5,
        predict_horizon_s: float = 120.0,
        min_moon_alt_deg: float = 0.0,
        cooldown_s: float = 180.0,
        *,
        log_margin_deg: float = 5.0,
        incoming_alert_lead_s: float = 60.0,
        incoming_alert_margin_deg: float = 0.25,
    ) -> None:
        self.obs_lat = obs_lat
        self.obs_lon = obs_lon
        self.obs_alt_m = obs_alt_m
        self.alert_margin_deg = alert_margin_deg
        self.predict_horizon_s = predict_horizon_s
        self.min_moon_alt_deg = min_moon_alt_deg
        self.cooldown_s = cooldown_s
        self.log_margin_deg = log_margin_deg
        self.incoming_alert_lead_s = incoming_alert_lead_s
        self.incoming_alert_margin_deg = incoming_alert_margin_deg

        # Per aircraft hex: (last alert timestamp, last severity)
        self._last_alerts: dict[str, tuple[datetime, int]] = {}
        # Per aircraft hex: previous sky sample for sub-second chord interpolation
        self._prev_sky: dict[str, tuple[datetime, SkyPosition, float]] = {}
        # Rolling pre-pass track buffer for aircraft approaching within wide margin
        self._pre_tracks: dict[str, deque[PassPoint]] = {}
        self._last_seen_any: dict[str, datetime] = {}
        self._active_passes: dict[str, _ActivePass] = {}
        self._completed_passes: list[PassRecord] = []

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
                track_rate_deg_s=ac.track_rate_deg_s,
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
        temperature_c: float = 10.0,
        pressure_hpa: float = 1010.0,
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
                track_rate_deg_s=ac.track_rate_deg_s,
            )
            sky = aircraft_altaz(
                self.obs_lat,
                self.obs_lon,
                self.obs_alt_m,
                plat,
                plon,
                palt,
                temperature_c=temperature_c,
                pressure_hpa=pressure_hpa,
            )
            m_alt, m_az = moon.at_offset(t_s)
            sep = angular_separation_deg(
                sky.altitude_deg,
                sky.azimuth_deg,
                m_alt,
                m_az,
            )
            return sep, sky.altitude_deg

        # Coarse scan every 2 seconds
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

        # Refine around the minimum in 0.1s steps
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

    def _classify_pass(
        self,
        min_sep_deg: float,
        disk_threshold_deg: float,
        alert_threshold_deg: float,
    ) -> tuple[bool, PassClassification]:
        if min_sep_deg <= disk_threshold_deg:
            return True, "TRANSIT"
        if min_sep_deg <= alert_threshold_deg:
            return False, "NEAR_MISS"
        return False, "CLOSE_PASS"

    def _finalize_pass(self, active: _ActivePass) -> PassRecord:
        return PassRecord(
            pass_id=active.pass_id,
            flight=active.flight,
            hex=active.hex,
            registration=active.registration,
            aircraft_type=active.aircraft_type,
            category=active.category,
            observer_lat=self.obs_lat,
            observer_lon=self.obs_lon,
            observer_elevation_m=self.obs_alt_m,
            start_utc=active.start_utc.isoformat(),
            end_utc=active.last_seen_utc.isoformat(),
            closest_utc=active.closest_utc.isoformat(),
            classification=active.classification,
            crossed_lunar_disk=active.crossed_lunar_disk,
            min_angular_sep_deg=active.min_sep_deg,
            min_moon_radii=active.min_moon_radii,
            closest_aircraft_alt_deg=active.closest_aircraft_alt_deg,
            closest_aircraft_az_deg=active.closest_aircraft_az_deg,
            closest_altitude_ft=active.closest_altitude_ft,
            closest_slant_range_km=active.closest_slant_range_km,
            moon_alt_deg=active.moon_alt_deg,
            moon_az_deg=active.moon_az_deg,
            moon_angular_diameter_deg=active.moon_angular_diameter_deg,
            moon_illumination_pct=active.moon_illumination_pct,
            moon_phase_name=active.moon_phase_name,
            track=list(active.track),
            moon_phase_deg=active.moon_phase_deg,
        )

    def _update_pass_tracking(
        self,
        ac: Aircraft,
        moon: MoonPosition,
        current_utc: datetime,
        sky_pos: SkyPosition,
        sep_deg: float,
        interp_point: PassPoint | None,
        interp_sep_deg: float | None,
        disk_threshold_deg: float,
        alert_threshold_deg: float,
        moon_eligible: bool,
    ) -> None:
        pt = PassPoint(
            timestamp_utc=current_utc.isoformat(),
            aircraft_alt_deg=sky_pos.altitude_deg,
            aircraft_az_deg=sky_pos.azimuth_deg,
            moon_alt_deg=moon.altitude_deg,
            moon_az_deg=moon.azimuth_deg,
            moon_angular_radius_deg=moon.angular_radius_deg,
            angular_sep_deg=sep_deg,
            altitude_ft=ac.altitude_ft,
            slant_range_km=sky_pos.slant_range_km,
            ground_distance_km=sky_pos.ground_distance_km,
            interpolated=False,
        )

        # Keep rolling buffer for live sky view and pre-pass context
        buf_margin = max(self.log_margin_deg * 3.0, 25.0)
        if sky_pos.altitude_deg > 0.0 and sep_deg <= buf_margin:
            buf = self._pre_tracks.setdefault(ac.hex, deque(maxlen=25))
            if not buf or buf[-1].timestamp_utc != pt.timestamp_utc:
                buf.append(pt)

        eff_sep = interp_sep_deg if (interp_sep_deg is not None and interp_sep_deg < sep_deg) else sep_deg
        active = self._active_passes.get(ac.hex)

        if active is None:
            if not (moon_eligible and sky_pos.altitude_deg > 0.0 and eff_sep <= self.log_margin_deg):
                return

            pre_pts = list(self._pre_tracks.get(ac.hex, ()))
            # Exclude duplicate of current pt if already appended to pre_pts
            if pre_pts and pre_pts[-1].timestamp_utc == pt.timestamp_utc:
                pre_pts = pre_pts[:-1]

            track = pre_pts[-15:]
            if interp_point is not None:
                track.append(interp_point)
            track.append(pt)

            best_pt = min(
                (p for p in (interp_point, pt) if p is not None),
                key=lambda p: p.angular_sep_deg,
            )
            best_dt = datetime.fromisoformat(best_pt.timestamp_utc)
            crossed, classification = self._classify_pass(
                best_pt.angular_sep_deg, disk_threshold_deg, alert_threshold_deg
            )
            start_dt = datetime.fromisoformat(track[0].timestamp_utc)
            pass_id = f"{current_utc.strftime('%Y%m%dT%H%M%SZ')}_{ac.hex}"
            self._active_passes[ac.hex] = _ActivePass(
                pass_id=pass_id,
                flight=ac.flight,
                hex=ac.hex,
                registration=ac.registration,
                aircraft_type=ac.aircraft_type,
                category=ac.category,
                start_utc=start_dt,
                last_seen_utc=current_utc,
                closest_utc=best_dt,
                min_sep_deg=best_pt.angular_sep_deg,
                min_moon_radii=(
                    best_pt.angular_sep_deg / moon.angular_radius_deg
                    if moon.angular_radius_deg > 0
                    else math.inf
                ),
                crossed_lunar_disk=crossed,
                classification=classification,
                closest_aircraft_alt_deg=best_pt.aircraft_alt_deg,
                closest_aircraft_az_deg=best_pt.aircraft_az_deg,
                closest_altitude_ft=best_pt.altitude_ft,
                closest_slant_range_km=best_pt.slant_range_km,
                moon_alt_deg=moon.altitude_deg,
                moon_az_deg=moon.azimuth_deg,
                moon_angular_diameter_deg=moon.angular_diameter_deg,
                moon_illumination_pct=moon.illumination_pct,
                moon_phase_deg=moon.phase_deg,
                moon_phase_name=moon.phase_name,
                track=track,
            )
            return

        # Update existing active pass
        if interp_point is not None:
            active.track.append(interp_point)
        active.track.append(pt)
        active.last_seen_utc = current_utc
        active.flight = ac.flight or active.flight

        for cand in (interp_point, pt):
            if cand is not None and cand.angular_sep_deg < active.min_sep_deg:
                active.min_sep_deg = cand.angular_sep_deg
                active.min_moon_radii = (
                    cand.angular_sep_deg / moon.angular_radius_deg
                    if moon.angular_radius_deg > 0
                    else math.inf
                )
                active.closest_utc = datetime.fromisoformat(cand.timestamp_utc)
                active.closest_aircraft_alt_deg = cand.aircraft_alt_deg
                active.closest_aircraft_az_deg = cand.aircraft_az_deg
                active.closest_altitude_ft = cand.altitude_ft
                active.closest_slant_range_km = cand.slant_range_km
                active.moon_alt_deg = moon.altitude_deg
                active.moon_az_deg = moon.azimuth_deg
                active.moon_angular_diameter_deg = moon.angular_diameter_deg
                active.moon_illumination_pct = moon.illumination_pct
                active.moon_phase_deg = moon.phase_deg
                active.moon_phase_name = moon.phase_name
                crossed, classification = self._classify_pass(
                    active.min_sep_deg, disk_threshold_deg, alert_threshold_deg
                )
                active.crossed_lunar_disk = active.crossed_lunar_disk or crossed
                active.classification = classification

        exit_margin = self.log_margin_deg + 1.5
        if sky_pos.altitude_deg <= 0.0 or sep_deg > exit_margin:
            active.outside_count += 1
            if active.outside_since_utc is None:
                active.outside_since_utc = current_utc
            if (
                active.outside_count >= 5
                or (current_utc - active.outside_since_utc).total_seconds() >= 10.0
            ):
                self._completed_passes.append(self._finalize_pass(active))
                del self._active_passes[ac.hex]
        else:
            active.outside_count = 0
            active.outside_since_utc = None

    def pop_completed_passes(self) -> list[PassRecord]:
        """Return and clear all passes completed since the last call."""
        completed = list(self._completed_passes)
        self._completed_passes.clear()
        return completed

    def flush_active_passes(self) -> list[PassRecord]:
        """Finalize and return all currently active passes (e.g. on shutdown)."""
        flushed = [self._finalize_pass(active) for active in self._active_passes.values()]
        self._active_passes.clear()
        return flushed

    def get_recent_tracks(self) -> dict[str, list[PassPoint]]:
        """Return recent observed sky points keyed by aircraft hex."""
        tracks: dict[str, list[PassPoint]] = {
            hex_code: list(buf) for hex_code, buf in self._pre_tracks.items() if buf
        }
        for hex_code, active in self._active_passes.items():
            tracks[hex_code] = list(active.track)
        return tracks

    def evaluate(
        self,
        moon: MoonPosition,
        aircraft_list: list[Aircraft],
        now_utc: datetime | None = None,
        *,
        temperature_c: float = 10.0,
        pressure_hpa: float = 1010.0,
    ) -> tuple[list[AircraftSighting], list[TransitEvent]]:
        """Evaluate all aircraft against the Moon and return sorted sightings and new events."""
        current_utc = now_utc or datetime.now(timezone.utc)
        if current_utc.tzinfo is None:
            current_utc = current_utc.replace(tzinfo=timezone.utc)

        moon_eligible = moon.altitude_deg >= self.min_moon_alt_deg
        sightings: list[AircraftSighting] = []
        events: list[TransitEvent] = []
        seen_hexes: set[str] = set()

        for ac in aircraft_list:
            seen_hexes.add(ac.hex)
            self._last_seen_any[ac.hex] = current_utc

            cur_lat, cur_lon, cur_alt_m = self._current_aircraft_coords(ac)
            sky_pos = aircraft_altaz(
                self.obs_lat,
                self.obs_lon,
                self.obs_alt_m,
                cur_lat,
                cur_lon,
                cur_alt_m,
                temperature_c=temperature_c,
                pressure_hpa=pressure_hpa,
            )
            sep_deg = angular_separation_deg(
                sky_pos.altitude_deg,
                sky_pos.azimuth_deg,
                moon.altitude_deg,
                moon.azimuth_deg,
            )

            # Check sub-second segment crossing between previous and current poll
            interp_point: PassPoint | None = None
            interp_sep_deg: float | None = None
            prev = self._prev_sky.get(ac.hex)
            if prev is not None:
                prev_dt, prev_sky, prev_alt_ft = prev
                dt_gap = (current_utc - prev_dt).total_seconds()
                if 0.0 < dt_gap <= 5.0 and (prev_sky.altitude_deg > 0.0 or sky_pos.altitude_deg > 0.0):
                    mid_m_alt, mid_m_az = moon.at_offset(-0.5 * dt_gap)
                    seg_sep, u, interp_alt, interp_az = segment_closest_approach(
                        prev_sky.altitude_deg,
                        prev_sky.azimuth_deg,
                        sky_pos.altitude_deg,
                        sky_pos.azimuth_deg,
                        mid_m_alt,
                        mid_m_az,
                    )
                    m_u_alt, m_u_az = moon.at_offset((u - 1.0) * dt_gap)
                    true_u_sep = angular_separation_deg(interp_alt, interp_az, m_u_alt, m_u_az)
                    prev_m_alt, prev_m_az = moon.at_offset(-dt_gap)
                    prev_sep = angular_separation_deg(prev_sky.altitude_deg, prev_sky.azimuth_deg, prev_m_alt, prev_m_az)
                    if 0.0 < u < 1.0 and true_u_sep < min(sep_deg, prev_sep):
                        interp_dt = prev_dt + timedelta(seconds=dt_gap * u)
                        interp_sep_deg = true_u_sep
                        interp_point = PassPoint(
                            timestamp_utc=interp_dt.isoformat(),
                            aircraft_alt_deg=interp_alt,
                            aircraft_az_deg=interp_az,
                            moon_alt_deg=m_u_alt,
                            moon_az_deg=m_u_az,
                            moon_angular_radius_deg=moon.angular_radius_deg,
                            angular_sep_deg=true_u_sep,
                            altitude_ft=prev_alt_ft + (ac.altitude_ft - prev_alt_ft) * u,
                            slant_range_km=prev_sky.slant_range_km + (sky_pos.slant_range_km - prev_sky.slant_range_km) * u,
                            ground_distance_km=prev_sky.ground_distance_km + (sky_pos.ground_distance_km - prev_sky.ground_distance_km) * u,
                            interpolated=True,
                        )
            self._prev_sky[ac.hex] = (current_utc, sky_pos, ac.altitude_ft)

            plane_ang = estimate_aircraft_angular_size_deg(
                sky_pos.slant_range_km,
                ac.category,
                ac.aircraft_type,
            )
            disk_threshold = moon.angular_radius_deg + plane_ang / 2.0
            alert_threshold = moon.angular_radius_deg + self.alert_margin_deg
            effective_sep = interp_sep_deg if (interp_sep_deg is not None and interp_sep_deg < sep_deg) else sep_deg

            self._update_pass_tracking(
                ac=ac,
                moon=moon,
                current_utc=current_utc,
                sky_pos=sky_pos,
                sep_deg=sep_deg,
                interp_point=interp_point,
                interp_sep_deg=interp_sep_deg,
                disk_threshold_deg=disk_threshold,
                alert_threshold_deg=alert_threshold,
                moon_eligible=moon_eligible,
            )

            moon_radii = effective_sep / moon.angular_radius_deg if moon.angular_radius_deg > 0 else math.inf
            is_disk_crossing = moon_eligible and sky_pos.altitude_deg > 0.0 and effective_sep <= disk_threshold

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
                    temperature_c=temperature_c,
                    pressure_hpa=pressure_hpa,
                )

            incoming_threshold = moon.angular_radius_deg + self.incoming_alert_margin_deg
            status: SightingStatus = "TRACKING"
            if moon_eligible and (sky_pos.altitude_deg > 0.0 or (interp_point is not None and interp_point.aircraft_alt_deg > 0.0)):
                if effective_sep <= disk_threshold:
                    status = "TRANSIT"
                elif effective_sep <= alert_threshold:
                    status = "NEAR_MISS"
                elif (
                    min_pred_sep is not None
                    and tca_s is not None
                    and 0.0 < tca_s <= self.incoming_alert_lead_s
                    and min_pred_sep <= incoming_threshold
                ):
                    status = "INCOMING"

            sighting = AircraftSighting(
                aircraft=ac,
                sky_pos=sky_pos,
                angular_sep_deg=effective_sep,
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

        # Finalize active passes for aircraft that dropped out of feed > 15s, and prune stale buffers
        for hex_code in list(self._active_passes):
            if hex_code not in seen_hexes:
                active = self._active_passes[hex_code]
                if (current_utc - active.last_seen_utc).total_seconds() >= 15.0:
                    self._completed_passes.append(self._finalize_pass(active))
                    del self._active_passes[hex_code]

        for hex_code in list(self._last_seen_any):
            if (current_utc - self._last_seen_any[hex_code]).total_seconds() > 60.0:
                self._last_seen_any.pop(hex_code, None)
                self._prev_sky.pop(hex_code, None)
                self._pre_tracks.pop(hex_code, None)

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
