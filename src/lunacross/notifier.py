"""JSONL logging, macOS notifications, Pushover alerts, and optional webhooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from lunacross.detector import PassRecord, TransitEvent
from lunacross.geometry import classify_visual_visibility
from lunacross.moon import compass_direction
from lunacross.weather import WeatherSnapshot

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_tar1090_aircraft_url(adsb_url: str | None, hex_code: str) -> str | None:
    """Build a direct tar1090 map URL pointing to the aircraft by ICAO hex."""
    if not adsb_url or not hex_code:
        return None
    parts = urlsplit(adsb_url)
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}/?icao={hex_code.strip().lower()}"


def load_passes(passes_file: str | Path) -> list[PassRecord]:
    """Load recorded PassRecord entries from a JSONL file (oldest first)."""
    path = Path(passes_file)
    if not path.is_file():
        return []
    records: list[PassRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(PassRecord.from_dict(json.loads(stripped)))
            except Exception:
                continue
    return records


def prune_old_close_passes(
    passes_file: str | Path,
    retention_days: int,
    *,
    now_utc: datetime | None = None,
) -> int:
    """Remove CLOSE_PASS records older than retention_days; always keep TRANSIT & NEAR_MISS."""
    if retention_days <= 0:
        return 0
    path = Path(passes_file)
    if not path.is_file():
        return 0

    cutoff = (now_utc or datetime.now(timezone.utc)) - timedelta(days=retention_days)
    kept_lines: list[str] = []
    removed = 0

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                if data.get("classification") == "CLOSE_PASS":
                    closest_dt = datetime.fromisoformat(str(data["closest_utc"]))
                    if closest_dt.tzinfo is None:
                        closest_dt = closest_dt.replace(tzinfo=timezone.utc)
                    if closest_dt < cutoff:
                        removed += 1
                        continue
            except Exception:
                pass
            kept_lines.append(stripped)

    if removed > 0:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for line in kept_lines:
                fh.write(line + "\n")
        os.replace(tmp_path, path)
    return removed


def _parse_hhmm_minutes(token: str) -> int:
    parts = token.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM token: {token!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Out-of-range HH:MM token: {token!r}")
    return hh * 60 + mm


def is_in_quiet_hours(local_dt: datetime, quiet_hours_spec: str | None) -> bool:
    """Return True if local_dt falls inside an 'HH:MM-HH:MM' quiet window."""
    if not quiet_hours_spec or not quiet_hours_spec.strip():
        return False
    raw = quiet_hours_spec.strip()
    if "-" not in raw:
        return False
    start_s, end_s = raw.split("-", 1)
    try:
        start_m = _parse_hhmm_minutes(start_s)
        end_m = _parse_hhmm_minutes(end_s)
    except ValueError:
        return False
    if start_m == end_m:
        return False
    cur_m = local_dt.hour * 60 + local_dt.minute
    if start_m < end_m:
        return start_m <= cur_m < end_m
    # Overnight window (e.g. 23:30-07:00)
    return cur_m >= start_m or cur_m < end_m


class TransitNotifier:
    """Dispatches detected lunar transit events to JSONL log, macOS, Pushover, and webhook."""

    def __init__(
        self,
        log_file: str | Path | None = "transits.jsonl",
        notify_macos: bool = True,
        webhook_url: str | None = None,
        *,
        passes_file: str | Path | None = "passes.jsonl",
        pushover_app_token: str | None = None,
        pushover_user_key: str | None = None,
        pushover_device: str | None = None,
        pushover_sound: str | None = None,
        pushover_priority: int = 1,
        pushover_events: str = "INCOMING,TRANSIT",
        pushover_min_angular_arcmin: float = 0.0,
        pushover_max_cloud_cover_pct: float = 100.0,
        pushover_quiet_hours: str | None = None,
        adsb_url: str | None = None,
    ) -> None:
        self.log_path = Path(log_file) if log_file else None
        self.passes_path = Path(passes_file) if passes_file else None
        self.notify_macos = notify_macos
        self.webhook_url = webhook_url
        self.pushover_app_token = pushover_app_token.strip() if pushover_app_token else None
        self.pushover_user_key = pushover_user_key.strip() if pushover_user_key else None
        self.pushover_device = pushover_device.strip() if pushover_device else None
        self.pushover_sound = pushover_sound.strip() if pushover_sound else None
        self.pushover_priority = max(-2, min(1, int(pushover_priority)))
        self._pushover_events = {
            part.strip().upper()
            for part in pushover_events.split(",")
            if part.strip()
        }
        self.pushover_min_angular_arcmin = max(0.0, float(pushover_min_angular_arcmin))
        self.pushover_max_cloud_cover_pct = max(0.0, min(100.0, float(pushover_max_cloud_cover_pct)))
        self.pushover_quiet_hours = pushover_quiet_hours.strip() if pushover_quiet_hours else None
        self.adsb_url = adsb_url

    @property
    def pushover_enabled(self) -> bool:
        """Return True when both Pushover credentials are configured."""
        return bool(self.pushover_app_token and self.pushover_user_key)

    def _build_payload(
        self,
        event: TransitEvent,
        weather: WeatherSnapshot | None = None,
        suppressed_reason: str | None = None,
    ) -> dict[str, object]:
        s = event.sighting
        ac = s.aircraft
        m = event.moon
        vis_tier, _ = classify_visual_visibility(s.plane_angular_size_deg, m.illumination_pct)
        payload: dict[str, object] = {
            "timestamp_utc": event.timestamp_utc.isoformat(),
            "event_type": event.event_type,
            "flight": ac.flight,
            "hex": ac.hex,
            "aircraft_type": ac.aircraft_type,
            "category": ac.category,
            "altitude_ft": round(ac.altitude_ft),
            "slant_range_km": round(s.sky_pos.slant_range_km, 2),
            "aircraft_alt_deg": round(s.sky_pos.altitude_deg, 3),
            "aircraft_az_deg": round(s.sky_pos.azimuth_deg, 3),
            "angular_sep_deg": round(s.angular_sep_deg, 4),
            "moon_radii": round(s.moon_radii, 2),
            "plane_angular_arcmin": round(s.plane_angular_size_deg * 60.0, 2),
            "visibility": vis_tier,
            "min_predicted_sep_deg": (
                round(s.min_predicted_sep_deg, 4) if s.min_predicted_sep_deg is not None else None
            ),
            "time_to_closest_s": (
                round(s.time_to_closest_s, 1) if s.time_to_closest_s is not None else None
            ),
            "predicted_transit_duration_s": s.predicted_transit_duration_s,
            "moon_alt_deg": round(m.altitude_deg, 3),
            "moon_az_deg": round(m.azimuth_deg, 3),
            "moon_illumination_pct": round(m.illumination_pct, 1),
            "moon_angular_diameter_deg": round(m.angular_diameter_deg, 4),
        }
        if weather is not None:
            payload["cloud_cover_pct"] = round(weather.cloud_cover_pct, 1)
            payload["sky_obstruction_pct"] = round(weather.sky_obstruction_pct, 1)
            payload["weather_condition"] = weather.condition_label
            payload["temperature_c"] = round(weather.temperature_c, 1)
        if suppressed_reason is not None:
            payload["pushover_suppressed_reason"] = suppressed_reason
        return payload

    def _write_jsonl(self, payload: dict[str, object]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def write_pass(self, record: PassRecord) -> None:
        """Append a completed close-pass track record to passes.jsonl."""
        if self.passes_path is None:
            return
        self.passes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.passes_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")

    def _send_macos_notification(self, event: TransitEvent) -> None:
        if not self.notify_macos or sys.platform != "darwin":
            return

        s = event.sighting
        ac = s.aircraft
        if event.event_type == "TRANSIT":
            title = f"Lunar Transit NOW: {ac.flight}"
            body = (
                f"Separation {s.angular_sep_deg:.2f}° ({s.moon_radii:.1f}R) "
                f"at {ac.altitude_ft:,.0f} ft"
            )
        elif event.event_type == "INCOMING":
            title = f"Lunar Transit in {s.time_to_closest_s:.0f}s: {ac.flight}"
            body = (
                f"Min. separation {s.min_predicted_sep_deg:.2f}° "
                f"at {ac.altitude_ft:,.0f} ft"
            )
        else:
            title = f"Aircraft near Moon: {ac.flight}"
            body = f"Separation {s.angular_sep_deg:.2f}° at {ac.altitude_ft:,.0f} ft"

        script = (
            f'display notification "{_escape_applescript(body)}" '
            f'with title "{_escape_applescript(title)}"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                timeout=2.0,
            )
        except Exception:
            pass

    def _format_pushover(
        self,
        event: TransitEvent,
        weather: WeatherSnapshot | None = None,
    ) -> tuple[str, str]:
        s = event.sighting
        ac = s.aircraft
        m = event.moon
        type_suffix = f" ({ac.aircraft_type})" if ac.aircraft_type else ""
        _, vis_label = classify_visual_visibility(s.plane_angular_size_deg, m.illumination_pct)
        moon_info = (
            f"Moon: alt {m.altitude_deg:.1f}°, az {m.azimuth_deg:.1f}° "
            f"({compass_direction(m.azimuth_deg)})"
        )
        if weather is not None:
            moon_info += f"\nSky: {weather.condition_label} ({weather.temperature_c:+.1f}°C)"
        plane_info = (
            f"Altitude: {ac.altitude_ft:,.0f} ft | Distance: {s.sky_pos.slant_range_km:.1f} km\n"
            f"Visibility: {vis_label}"
        )

        if event.event_type == "INCOMING" and s.time_to_closest_s is not None:
            eta_local = (event.timestamp_utc + timedelta(seconds=s.time_to_closest_s)).astimezone()
            min_sep = s.min_predicted_sep_deg if s.min_predicted_sep_deg is not None else s.angular_sep_deg
            pred_radii = min_sep / m.angular_radius_deg if m.angular_radius_deg > 0 else 0.0
            crosses_disk = min_sep <= m.angular_radius_deg
            cross_txt = "Crossing lunar disk" if crosses_disk else "Skimming lunar limb"
            dur_txt = (
                f" (~{s.predicted_transit_duration_s:.1f}s)"
                if s.predicted_transit_duration_s
                else ""
            )
            title = f"Lunar transit in {s.time_to_closest_s:.0f}s: {ac.flight}"
            message = (
                f"{ac.flight}{type_suffix} is approaching the Moon!\n"
                f"Expected: {eta_local.strftime('%H:%M:%S')} ({cross_txt}{dur_txt})\n"
                f"Min. separation: {min_sep:.2f}° ({pred_radii:.2f}R)\n"
                f"{plane_info}\n"
                f"{moon_info}"
            )
            return title, message

        if event.event_type == "TRANSIT":
            title = f"Crossing Moon NOW: {ac.flight}"
            message = (
                f"{ac.flight}{type_suffix} is crossing in front of the Moon now!\n"
                f"Separation: {s.angular_sep_deg:.3f}° ({s.moon_radii:.2f}R)\n"
                f"{plane_info}\n"
                f"{moon_info}"
            )
            return title, message

        title = f"Near Moon: {ac.flight}"
        message = (
            f"{ac.flight}{type_suffix} is passing close to the Moon.\n"
            f"Separation: {s.angular_sep_deg:.2f}° ({s.moon_radii:.2f}R)\n"
            f"{plane_info}\n"
            f"{moon_info}"
        )
        return title, message

    def send_pushover(
        self,
        title: str,
        message: str,
        *,
        url: str | None = None,
        url_title: str | None = None,
        raise_on_error: bool = False,
    ) -> bool:
        """Send a notification via the Pushover Messages API."""
        if not self.pushover_enabled:
            return False

        form_data: dict[str, str] = {
            "token": self.pushover_app_token or "",
            "user": self.pushover_user_key or "",
            "title": title,
            "message": message,
            "priority": str(self.pushover_priority),
        }
        if self.pushover_device:
            form_data["device"] = self.pushover_device
        if self.pushover_sound:
            form_data["sound"] = self.pushover_sound
        if url:
            form_data["url"] = url
            if url_title:
                form_data["url_title"] = url_title

        try:
            resp = httpx.post(PUSHOVER_API_URL, data=form_data, timeout=5.0)
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") != 1:
                raise RuntimeError(f"Pushover API error: {body}")
            return True
        except Exception:
            if raise_on_error:
                raise
            return False

    def _send_webhook(self, payload: dict[str, object]) -> None:
        if not self.webhook_url:
            return
        try:
            httpx.post(self.webhook_url, json=payload, timeout=2.0)
        except Exception:
            pass

    def _pushover_suppression_reason(
        self,
        event: TransitEvent,
        weather: WeatherSnapshot | None,
    ) -> str | None:
        local_dt = event.timestamp_utc.astimezone()
        if is_in_quiet_hours(local_dt, self.pushover_quiet_hours):
            return "quiet_hours"
        if (
            weather is not None
            and self.pushover_max_cloud_cover_pct < 100.0
            and weather.sky_obstruction_pct > self.pushover_max_cloud_cover_pct
        ):
            return "cloud_cover"
        arcmin = event.sighting.plane_angular_size_deg * 60.0
        if self.pushover_min_angular_arcmin > 0.0 and arcmin < self.pushover_min_angular_arcmin:
            return "angular_size"
        return None

    def notify(
        self,
        event: TransitEvent,
        *,
        weather: WeatherSnapshot | None = None,
    ) -> dict[str, object]:
        """Log event and dispatch all configured notifications."""
        want_pushover = self.pushover_enabled and event.event_type in self._pushover_events
        suppressed = self._pushover_suppression_reason(event, weather) if want_pushover else None

        payload = self._build_payload(event, weather=weather, suppressed_reason=suppressed)
        self._write_jsonl(payload)
        if suppressed is None:
            self._send_macos_notification(event)
        if want_pushover and suppressed is None:
            title, message = self._format_pushover(event, weather=weather)
            tar_url = build_tar1090_aircraft_url(self.adsb_url, event.sighting.aircraft.hex)
            self.send_pushover(
                title,
                message,
                url=tar_url,
                url_title="View on radar (tar1090)" if tar_url else None,
                raise_on_error=False,
            )
        self._send_webhook(payload)
        return payload
