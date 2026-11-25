"""JSONL logging, macOS notifications, Pushover alerts, and optional webhooks."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from lunacross.detector import PassRecord, TransitEvent
from lunacross.moon import compass_direction

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
        self.adsb_url = adsb_url

    @property
    def pushover_enabled(self) -> bool:
        """Return True when both Pushover credentials are configured."""
        return bool(self.pushover_app_token and self.pushover_user_key)

    def _build_payload(self, event: TransitEvent) -> dict[str, object]:
        s = event.sighting
        ac = s.aircraft
        m = event.moon
        return {
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

    def _format_pushover(self, event: TransitEvent) -> tuple[str, str]:
        s = event.sighting
        ac = s.aircraft
        m = event.moon
        type_suffix = f" ({ac.aircraft_type})" if ac.aircraft_type else ""
        moon_info = (
            f"Moon: alt {m.altitude_deg:.1f}°, az {m.azimuth_deg:.1f}° "
            f"({compass_direction(m.azimuth_deg)})"
        )
        plane_info = f"Altitude: {ac.altitude_ft:,.0f} ft | Distance: {s.sky_pos.slant_range_km:.1f} km"

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

    def notify(self, event: TransitEvent) -> dict[str, object]:
        """Log event and dispatch all configured notifications."""
        payload = self._build_payload(event)
        self._write_jsonl(payload)
        self._send_macos_notification(event)
        if self.pushover_enabled and event.event_type in self._pushover_events:
            title, message = self._format_pushover(event)
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
