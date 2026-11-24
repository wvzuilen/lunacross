"""Logging (JSONL), macOS-meldingen, Pushover en optionele webhooks voor transits."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from lunacross.detector import TransitEvent
from lunacross.moon import compass_direction

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_tar1090_aircraft_url(adsb_url: str | None, hex_code: str) -> str | None:
    """Bouw een directe tar1090-link naar het vliegtuig op de lokale radar."""
    if not adsb_url or not hex_code:
        return None
    parts = urlsplit(adsb_url)
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}/?icao={hex_code.strip().lower()}"


class TransitNotifier:
    """Verwerkt gedetecteerde maanovergangen naar logbestand, macOS, Pushover en webhook."""

    def __init__(
        self,
        log_file: str | Path | None = "transits.jsonl",
        notify_macos: bool = True,
        webhook_url: str | None = None,
        *,
        pushover_app_token: str | None = None,
        pushover_user_key: str | None = None,
        pushover_device: str | None = None,
        pushover_sound: str | None = None,
        pushover_priority: int = 1,
        pushover_events: str = "INCOMING,TRANSIT",
        adsb_url: str | None = None,
    ) -> None:
        self.log_path = Path(log_file) if log_file else None
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
        """Geef aan of Pushover-credentials compleet zijn."""
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

    def _send_macos_notification(self, event: TransitEvent) -> None:
        if not self.notify_macos or sys.platform != "darwin":
            return

        s = event.sighting
        ac = s.aircraft
        if event.event_type == "TRANSIT":
            title = f"Maanovergang NU: {ac.flight}"
            body = (
                f"Afstand {s.angular_sep_deg:.2f}° ({s.moon_radii:.1f}R) "
                f"op {ac.altitude_ft:.0f} ft"
            )
        elif event.event_type == "INCOMING":
            title = f"Maanovergang over {s.time_to_closest_s:.0f}s: {ac.flight}"
            body = (
                f"Min. scheiding {s.min_predicted_sep_deg:.2f}° "
                f"op {ac.altitude_ft:.0f} ft"
            )
        else:
            title = f"Vliegtuig vlakbij maan: {ac.flight}"
            body = f"Scheiding {s.angular_sep_deg:.2f}° op {ac.altitude_ft:.0f} ft"

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
            f"Maan: alt {m.altitude_deg:.1f}°, az {m.azimuth_deg:.1f}° "
            f"({compass_direction(m.azimuth_deg)})"
        )
        plane_info = f"Hoogte: {ac.altitude_ft:,.0f} ft | Afstand: {s.sky_pos.slant_range_km:.1f} km".replace(
            ",", "."
        )

        if event.event_type == "INCOMING" and s.time_to_closest_s is not None:
            eta_local = (event.timestamp_utc + timedelta(seconds=s.time_to_closest_s)).astimezone()
            min_sep = s.min_predicted_sep_deg if s.min_predicted_sep_deg is not None else s.angular_sep_deg
            pred_radii = min_sep / m.angular_radius_deg if m.angular_radius_deg > 0 else 0.0
            crosses_disk = min_sep <= m.angular_radius_deg
            cross_txt = "Kruist maanschijf" if crosses_disk else "Langs de maanrand"
            dur_txt = (
                f" (~{s.predicted_transit_duration_s:.1f}s)"
                if s.predicted_transit_duration_s
                else ""
            )
            title = f"Maanovergang over {s.time_to_closest_s:.0f}s: {ac.flight}"
            message = (
                f"{ac.flight}{type_suffix} nadert de maan!\n"
                f"Verwacht: {eta_local.strftime('%H:%M:%S')} ({cross_txt}{dur_txt})\n"
                f"Min. scheiding: {min_sep:.2f}° ({pred_radii:.2f}R)\n"
                f"{plane_info}\n"
                f"{moon_info}"
            )
            return title, message

        if event.event_type == "TRANSIT":
            title = f"NU voor de maan: {ac.flight}"
            message = (
                f"{ac.flight}{type_suffix} vliegt nu voor de maan langs!\n"
                f"Scheiding: {s.angular_sep_deg:.3f}° ({s.moon_radii:.2f}R)\n"
                f"{plane_info}\n"
                f"{moon_info}"
            )
            return title, message

        title = f"Vlakbij de maan: {ac.flight}"
        message = (
            f"{ac.flight}{type_suffix} passeert vlak langs de maan.\n"
            f"Scheiding: {s.angular_sep_deg:.2f}° ({s.moon_radii:.2f}R)\n"
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
        """Verstuur een notificatie via de Pushover Messages API."""
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
                raise RuntimeError(f"Pushover API fout: {body}")
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
        """Log event en verstuur geconfigureerde meldingen."""
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
                url_title="Bekijk op radar (tar1090)" if tar_url else None,
                raise_on_error=False,
            )
        self._send_webhook(payload)
        return payload
