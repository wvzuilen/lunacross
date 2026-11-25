"""CLI and live Rich dashboard for LunaCross."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import httpx
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lunacross.adsb import AdsbClient, Aircraft
from lunacross.config import Config, load_config
from lunacross.detector import AircraftSighting, SightingStatus, TransitDetector, TransitEvent
from lunacross.geometry import KNOTS_TO_MPS, extrapolate_position
from lunacross.moon import MoonPosition, MoonRiseSet, MoonTracker, compass_direction
from lunacross.notifier import TransitNotifier, load_passes
from lunacross.visualize import (
    render_live_snapshot_html,
    render_live_snapshot_svg,
    render_pass_html,
    render_pass_svg,
)
from lunacross.web import LiveState, LunaCrossWebServer

console = Console()


def _format_local(dt_utc: datetime | None) -> str:
    if dt_utc is None:
        return "Not within 36h"
    local_dt = dt_utc.astimezone()
    return local_dt.strftime("%Y-%m-%d %H:%M:%S (%Z)")


def _status_badge(status: SightingStatus) -> Text:
    if status == "TRANSIT":
        return Text("TRANSIT!", style="bold white on red")
    if status == "NEAR_MISS":
        return Text("NEAR MISS", style="bold black on yellow")
    if status == "INCOMING":
        return Text("INCOMING", style="bold black on cyan")
    return Text("Tracking", style="dim")


def _build_moon_panel(
    moon: MoonPosition,
    rise_set: MoonRiseSet,
    cfg: Config,
) -> Panel:
    local_time = moon.timestamp_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    horizon_txt = (
        "[bold green]Above horizon[/]"
        if moon.is_above_horizon
        else "[bold red]Below horizon[/]"
    )
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row(
        "Location:",
        f"{cfg.latitude:.6f}° N, {cfg.longitude:.6f}° E ({cfg.elevation_m:.0f} m)",
        "Time:",
        f"{local_time}",
    )
    table.add_row(
        "Altitude (Alt):",
        f"{moon.altitude_deg:+.2f}°",
        "Azimuth (Az):",
        f"{moon.azimuth_deg:.2f}° ({compass_direction(moon.azimuth_deg)})",
    )
    table.add_row(
        "Angular size:",
        f"{moon.angular_diameter_deg:.3f}° ({moon.angular_diameter_deg * 60:.1f}')",
        "Distance:",
        f"{moon.distance_km:,.0f} km",
    )
    table.add_row(
        "Phase:",
        f"{moon.phase_name} ({moon.illumination_pct:.1f}% lit)",
        "Visibility:",
        horizon_txt,
    )
    table.add_row(
        "Next moonrise:",
        _format_local(rise_set.next_rise_utc),
        "Next moonset:",
        _format_local(rise_set.next_set_utc),
    )

    return Panel(
        table,
        title="[bold yellow]Moon Position (Topocentric)[/]",
        border_style="yellow",
    )


def _build_aircraft_table(
    sightings: list[AircraftSighting],
    top_n: int,
    total_raw: int,
    feed_status: str,
) -> Table:
    table = Table(
        title=f"Closest Aircraft to the Moon  |  {feed_status}  |  Airborne: {len(sightings)} (feed total: {total_raw})",
        box=box.ROUNDED,
        expand=True,
    )
    table.add_column("Flight", style="bold")
    table.add_column("Type/Cat", justify="center")
    table.add_column("Altitude", justify="right")
    table.add_column("Range", justify="right")
    table.add_column("Sky (Az / Alt)", justify="right")
    table.add_column("Δ Moon", justify="right")
    table.add_column("Prediction", justify="right")
    table.add_column("Status", justify="center")

    shown = sightings[:top_n]
    if not shown:
        table.add_row("-", "-", "-", "-", "-", "-", "-", Text("No aircraft", style="dim"))
        return table

    for s in shown:
        ac = s.aircraft
        type_str = ac.aircraft_type or ac.category or "-"
        alt_str = f"{ac.altitude_ft:,.0f} ft"
        dist_str = f"{s.sky_pos.slant_range_km:.1f} km"
        sky_str = f"{s.sky_pos.azimuth_deg:5.1f}° / {s.sky_pos.altitude_deg:+5.1f}°"
        sep_str = f"{s.angular_sep_deg:.2f}° ({s.moon_radii:.1f}R)"
        if s.min_predicted_sep_deg is not None and s.time_to_closest_s is not None:
            pred_str = f"{s.min_predicted_sep_deg:.2f}° in {s.time_to_closest_s:.0f}s"
            if s.predicted_transit_duration_s:
                pred_str += f" (~{s.predicted_transit_duration_s:.1f}s)"
        else:
            pred_str = "-"

        table.add_row(
            ac.flight,
            type_str,
            alt_str,
            dist_str,
            sky_str,
            sep_str,
            pred_str,
            _status_badge(s.status),
        )
    return table


def _build_alerts_panel(
    active: list[AircraftSighting],
    recent_events: list[TransitEvent],
    moon: MoonPosition,
    min_moon_alt_deg: float,
) -> Panel:
    if moon.altitude_deg < min_moon_alt_deg:
        content = Text(
            f"Moon is below the configured threshold ({moon.altitude_deg:.1f}° < {min_moon_alt_deg:.1f}°). "
            "Aircraft are still tracked; transit alerts will activate once the Moon rises above the horizon.",
            style="yellow",
        )
        return Panel(content, title="Transit Status", border_style="yellow")

    lines: list[str] = []
    for s in active:
        ac = s.aircraft
        if s.status == "TRANSIT":
            lines.append(
                f"[bold red]CROSSING MOON NOW:[/] {ac.flight} ({ac.hex}) | "
                f"separation {s.angular_sep_deg:.3f}° ({s.moon_radii:.2f}R) | {ac.altitude_ft:,.0f} ft"
            )
        elif s.status == "NEAR_MISS":
            lines.append(
                f"[bold yellow]NEAR MOON:[/] {ac.flight} ({ac.hex}) | "
                f"separation {s.angular_sep_deg:.2f}° | {ac.altitude_ft:,.0f} ft"
            )
        elif s.status == "INCOMING" and s.time_to_closest_s is not None:
            lines.append(
                f"[bold cyan]INCOMING TRANSIT:[/] {ac.flight} ({ac.hex}) | "
                f"min. separation {s.min_predicted_sep_deg:.2f}° in {s.time_to_closest_s:.0f}s"
            )

    if recent_events:
        if lines:
            lines.append("")
        lines.append("[bold]Recent alerts:[/]")
        for ev in recent_events[-5:]:
            ts = ev.timestamp_utc.astimezone().strftime("%H:%M:%S")
            lines.append(
                f"  {ts}  [{ev.event_type}] {ev.sighting.aircraft.flight} — {ev.sighting.angular_sep_deg:.2f}°"
            )

    if not lines:
        lines.append("[dim]No active or incoming lunar transits within the prediction horizon.[/]")

    border = "red" if any(s.status == "TRANSIT" for s in active) else "cyan"
    return Panel("\n".join(lines), title="Lunar Transit Alerts", border_style=border)


def _sync_receiver_if_requested(cfg: Config, sync_receiver: bool) -> Config:
    if not sync_receiver:
        return cfg
    try:
        with AdsbClient(cfg.adsb_url) as client:
            coords = client.fetch_receiver_location()
        if coords is not None:
            cfg.latitude, cfg.longitude = coords
            console.print(
                f"[green]Synced receiver location from server: "
                f"{cfg.latitude:.6f}, {cfg.longitude:.6f}[/]"
            )
        else:
            console.print("[yellow]Warning: receiver.json did not contain lat/lon.[/]")
    except httpx.HTTPError as exc:
        console.print(f"[yellow]Could not fetch receiver.json ({exc}); using configured location.[/]")
    return cfg


def _cmd_moon(cfg: Config) -> int:
    tracker = MoonTracker(cfg.latitude, cfg.longitude, cfg.elevation_m, cfg.ephemeris_dir)
    now = datetime.now(timezone.utc)
    moon = tracker.get_position(now)
    rise_set = tracker.next_rise_set(now)
    console.print(_build_moon_panel(moon, rise_set, cfg))
    return 0


def _cmd_check(cfg: Config) -> int:
    tracker = MoonTracker(cfg.latitude, cfg.longitude, cfg.elevation_m, cfg.ephemeris_dir)
    now = datetime.now(timezone.utc)
    moon = tracker.get_position(now)
    rise_set = tracker.next_rise_set(now)

    detector = TransitDetector(
        obs_lat=cfg.latitude,
        obs_lon=cfg.longitude,
        obs_alt_m=cfg.elevation_m,
        alert_margin_deg=cfg.alert_margin_deg,
        predict_horizon_s=cfg.predict_horizon_s,
        min_moon_alt_deg=cfg.min_moon_alt_deg,
    )

    try:
        with AdsbClient(cfg.adsb_url) as client:
            snapshot = client.fetch_snapshot(max_seen_pos_s=cfg.max_seen_pos_s)
    except httpx.HTTPError as exc:
        console.print(_build_moon_panel(moon, rise_set, cfg))
        console.print(f"[bold red]Failed to fetch ADS-B data ({cfg.adsb_url}):[/] {exc}")
        return 1

    sightings, _ = detector.evaluate(moon, snapshot.aircraft, now_utc=now)
    active = [s for s in sightings if s.status != "TRACKING"]

    console.print(_build_moon_panel(moon, rise_set, cfg))
    console.print(_build_alerts_panel(active, [], moon, cfg.min_moon_alt_deg))
    console.print(
        _build_aircraft_table(
            sightings,
            cfg.top_n,
            snapshot.total_raw,
            f"[green]Connected[/] ({cfg.adsb_url})",
        )
    )
    return 0


def _cmd_monitor(cfg: Config) -> int:
    tracker = MoonTracker(cfg.latitude, cfg.longitude, cfg.elevation_m, cfg.ephemeris_dir)
    detector = TransitDetector(
        obs_lat=cfg.latitude,
        obs_lon=cfg.longitude,
        obs_alt_m=cfg.elevation_m,
        alert_margin_deg=cfg.alert_margin_deg,
        predict_horizon_s=cfg.predict_horizon_s,
        min_moon_alt_deg=cfg.min_moon_alt_deg,
        log_margin_deg=cfg.log_margin_deg,
        incoming_alert_lead_s=cfg.pushover_max_lead_s,
        incoming_alert_margin_deg=cfg.pushover_margin_deg,
    )
    notifier = TransitNotifier(
        log_file=cfg.log_file,
        notify_macos=cfg.notify_macos,
        webhook_url=cfg.webhook_url,
        passes_file=cfg.passes_file,
        pushover_app_token=cfg.pushover_app_token,
        pushover_user_key=cfg.pushover_user_key,
        pushover_device=cfg.pushover_device,
        pushover_sound=cfg.pushover_sound,
        pushover_priority=cfg.pushover_priority,
        pushover_events=cfg.pushover_events,
        adsb_url=cfg.adsb_url,
    )

    live_state = LiveState()
    web_server: LunaCrossWebServer | None = None
    if cfg.web_port > 0:
        web_server = LunaCrossWebServer("0.0.0.0", cfg.web_port, cfg, live_state)
        web_server.start()
        if not console.is_terminal:
            console.print(f"Web UI listening on http://0.0.0.0:{web_server.port}")
            sys.stdout.flush()

    stop_requested = False

    def _handle_sigterm(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    recent_events: list[TransitEvent] = []
    cached_rise_set: MoonRiseSet | None = None
    last_rise_set_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    first_non_tty_print = True

    def poll_once() -> tuple[MoonPosition, MoonRiseSet, list[AircraftSighting], list[TransitEvent], int, str]:
        nonlocal cached_rise_set, last_rise_set_at
        now = datetime.now(timezone.utc)
        moon = tracker.get_position(now)
        if (
            cached_rise_set is None
            or last_rise_set_at is None
            or (now - last_rise_set_at) > timedelta(minutes=30)
        ):
            cached_rise_set = tracker.next_rise_set(now)
            last_rise_set_at = now

        try:
            snapshot = client.fetch_snapshot(max_seen_pos_s=cfg.max_seen_pos_s)
            sightings, events = detector.evaluate(moon, snapshot.aircraft, now_utc=now)
            for ev in events:
                notifier.notify(ev)
                recent_events.append(ev)
            for rec in detector.pop_completed_passes():
                notifier.write_pass(rec)
                if not console.is_terminal:
                    ts = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
                    console.print(
                        f"[{ts}] PASS SAVED [{rec.classification}]: {rec.flight} ({rec.hex}) "
                        f"| min_sep={rec.min_angular_sep_deg:.3f}° ({rec.min_moon_radii:.2f}R) "
                        f"| {len(rec.track)} pts"
                    )
            if len(recent_events) > 20:
                del recent_events[:-20]
            feed_status = "[bold green]Live connected[/]"
            total_raw = snapshot.total_raw
            live_state.update(
                moon=moon,
                sightings=sightings,
                tracks=detector.get_recent_tracks(),
                total_raw=total_raw,
                feed_ok=True,
            )
        except httpx.HTTPError as exc:
            sightings = []
            events = []
            total_raw = 0
            feed_status = f"[bold red]Connection error: {exc}[/]"
            live_state.update(
                moon=moon,
                sightings=[],
                tracks=detector.get_recent_tracks(),
                total_raw=0,
                feed_ok=False,
            )

        return moon, cached_rise_set, sightings, events, total_raw, feed_status

    try:
        with AdsbClient(cfg.adsb_url) as client:
            if console.is_terminal:
                with Live(console=console, refresh_per_second=2, screen=False) as live:
                    try:
                        while not stop_requested:
                            moon, rise_set, sightings, _, total_raw, feed_status = poll_once()
                            active = [s for s in sightings if s.status != "TRACKING"]
                            renderable = Group(
                                _build_moon_panel(moon, rise_set, cfg),
                                _build_alerts_panel(active, recent_events, moon, cfg.min_moon_alt_deg),
                                _build_aircraft_table(sightings, cfg.top_n, total_raw, feed_status),
                            )
                            live.update(renderable)
                            time.sleep(cfg.poll_interval_s)
                    except KeyboardInterrupt:
                        pass
            else:
                try:
                    while not stop_requested:
                        moon, rise_set, sightings, events, total_raw, feed_status = poll_once()
                        active = [s for s in sightings if s.status != "TRACKING"]
                        if first_non_tty_print:
                            console.print(_build_moon_panel(moon, rise_set, cfg))
                            console.print(_build_alerts_panel(active, recent_events, moon, cfg.min_moon_alt_deg))
                            console.print(_build_aircraft_table(sightings, cfg.top_n, total_raw, feed_status))
                            sys.stdout.flush()
                            first_non_tty_print = False
                            last_heartbeat_at = moon.timestamp_utc
                        else:
                            for ev in events:
                                ts = ev.timestamp_utc.astimezone().strftime("%H:%M:%S")
                                s = ev.sighting
                                console.print(
                                    f"[{ts}] EVENT {ev.event_type}: {s.aircraft.flight} ({s.aircraft.hex}) "
                                    f"| sep={s.angular_sep_deg:.3f}° ({s.moon_radii:.2f}R) "
                                    f"| alt={s.aircraft.altitude_ft:.0f} ft"
                                )
                            if (
                                last_heartbeat_at is None
                                or (moon.timestamp_utc - last_heartbeat_at).total_seconds() >= 15.0
                            ):
                                ts = moon.timestamp_utc.astimezone().strftime("%H:%M:%S")
                                closest = (
                                    f"{sightings[0].aircraft.flight} ({sightings[0].angular_sep_deg:.2f}° / {sightings[0].moon_radii:.1f}R)"
                                    if sightings
                                    else "none"
                                )
                                console.print(
                                    f"[{ts}] Moon Alt={moon.altitude_deg:+.2f}° Az={moon.azimuth_deg:.2f}° "
                                    f"| aircraft={len(sightings)}/{total_raw} | closest={closest}"
                                )
                                last_heartbeat_at = moon.timestamp_utc
                            sys.stdout.flush()
                        time.sleep(cfg.poll_interval_s)
                except KeyboardInterrupt:
                    pass
    finally:
        for rec in detector.flush_active_passes():
            notifier.write_pass(rec)
        if web_server is not None:
            web_server.stop()

    console.print("\n[dim]Monitoring stopped.[/]")
    return 0


def _cmd_test_pushover(cfg: Config) -> int:
    if not (cfg.pushover_app_token and cfg.pushover_user_key):
        console.print(
            "[bold yellow]Pushover is not configured yet.[/]\n"
            "Set [cyan]pushover_app_token[/] and [cyan]pushover_user_key[/] via:\n"
            "  1. [bold]config.toml[/] (copy [cyan]config.example.toml[/] to [cyan]config.toml[/]), or\n"
            "  2. Environment variables [cyan]LUNACROSS_PUSHOVER_APP_TOKEN[/] and "
            "[cyan]LUNACROSS_PUSHOVER_USER_KEY[/]."
        )
        return 1

    tracker = MoonTracker(cfg.latitude, cfg.longitude, cfg.elevation_m, cfg.ephemeris_dir)
    moon = tracker.get_position()
    notifier = TransitNotifier(
        log_file=None,
        notify_macos=False,
        pushover_app_token=cfg.pushover_app_token,
        pushover_user_key=cfg.pushover_user_key,
        pushover_device=cfg.pushover_device,
        pushover_sound=cfg.pushover_sound,
        pushover_priority=cfg.pushover_priority,
        pushover_events=cfg.pushover_events,
        adsb_url=cfg.adsb_url,
    )
    try:
        notifier.send_pushover(
            title="LunaCross test notification",
            message=(
                f"Pushover is working! Current Moon: alt {moon.altitude_deg:+.1f}°, "
                f"az {moon.azimuth_deg:.1f}° ({compass_direction(moon.azimuth_deg)}), "
                f"{moon.illumination_pct:.1f}% lit."
            ),
            url=cfg.adsb_url.replace("/data/aircraft.json", "/"),
            url_title="Open tar1090 radar",
            raise_on_error=True,
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to send Pushover notification:[/] {exc}")
        return 1

    console.print("[bold green]Test notification sent successfully via Pushover![/]")
    return 0


def _write_render_output(output_path: Path, is_svg: bool, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    kind = "SVG" if is_svg else "HTML"
    console.print(f"[green]{kind} sky visualization saved to[/] {output_path}")


def _cmd_render(
    cfg: Config,
    *,
    list_only: bool = False,
    live: bool = False,
    pass_id: str | None = None,
    output: Path | None = None,
) -> int:
    if live:
        tracker = MoonTracker(cfg.latitude, cfg.longitude, cfg.elevation_m, cfg.ephemeris_dir)
        now = datetime.now(timezone.utc)
        moon = tracker.get_position(now)
        detector = TransitDetector(
            obs_lat=cfg.latitude,
            obs_lon=cfg.longitude,
            obs_alt_m=cfg.elevation_m,
            alert_margin_deg=cfg.alert_margin_deg,
            predict_horizon_s=cfg.predict_horizon_s,
            min_moon_alt_deg=cfg.min_moon_alt_deg,
            log_margin_deg=cfg.log_margin_deg,
        )
        try:
            with AdsbClient(cfg.adsb_url) as client:
                snapshot = client.fetch_snapshot(max_seen_pos_s=cfg.max_seen_pos_s)
        except httpx.HTTPError as exc:
            console.print(f"[bold red]Failed to fetch ADS-B data ({cfg.adsb_url}):[/] {exc}")
            return 1

        sightings, _ = detector.evaluate(moon, snapshot.aircraft, now_utc=now)
        out_path = output or Path("lunacross_live.html")
        is_svg = out_path.suffix.lower() == ".svg"
        content = (
            render_live_snapshot_svg(moon, sightings, detector.get_recent_tracks(), cfg)
            if is_svg
            else render_live_snapshot_html(moon, sightings, detector.get_recent_tracks(), cfg)
        )
        _write_render_output(out_path, is_svg, content)
        return 0

    passes = load_passes(cfg.passes_file)
    if list_only:
        if not passes:
            console.print(f"[yellow]No passes recorded yet in {cfg.passes_file}.[/]")
            return 0
        table = Table(title=f"Recorded Close Passes ({cfg.passes_file})", box=box.ROUNDED)
        table.add_column("Pass ID")
        table.add_column("Closest (Local)")
        table.add_column("Flight", style="bold")
        table.add_column("Class", justify="center")
        table.add_column("Min Sep", justify="right")
        table.add_column("Alt (ft)", justify="right")
        table.add_column("Pts", justify="right")
        for p in reversed(passes):
            dt_str = datetime.fromisoformat(p.closest_utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            table.add_row(
                p.pass_id,
                dt_str,
                p.flight,
                p.classification,
                f"{p.min_angular_sep_deg:.3f}° ({p.min_moon_radii:.2f}R)",
                f"{p.closest_altitude_ft:,.0f}",
                str(len(p.track)),
            )
        console.print(table)
        return 0

    if not passes:
        console.print(f"[yellow]No passes recorded yet in {cfg.passes_file}.[/]")
        return 1

    target = passes[-1]
    if pass_id is not None:
        matched = next((p for p in passes if p.pass_id == pass_id), None)
        if matched is None:
            console.print(f"[bold red]Pass ID not found:[/] {pass_id}")
            return 1
        target = matched

    out_path = output or Path(f"{target.pass_id}.html")
    is_svg = out_path.suffix.lower() == ".svg"
    content = render_pass_svg(target) if is_svg else render_pass_html(target)
    _write_render_output(out_path, is_svg, content)
    return 0


def _cmd_simulate(
    cfg: Config,
    write_log: bool = False,
    send_pushover: bool = False,
    render_path: Path | None = None,
) -> int:
    tracker = MoonTracker(cfg.latitude, cfg.longitude, cfg.elevation_m, cfg.ephemeris_dir)
    now = datetime.now(timezone.utc)
    moon = tracker.get_position(now)

    using_synthetic_moon = not moon.is_above_horizon or moon.altitude_deg < 10.0
    if using_synthetic_moon:
        console.print(
            f"[yellow]Note: the real Moon is currently at {moon.altitude_deg:+.1f}° "
            "altitude. Using a temporary synthetic Moon at +35.0° altitude, 180.0° azimuth for this simulation.[/]"
        )
        moon = MoonPosition(
            timestamp_utc=now,
            altitude_deg=35.0,
            azimuth_deg=180.0,
            distance_km=moon.distance_km,
            angular_radius_deg=moon.angular_radius_deg,
            angular_diameter_deg=moon.angular_diameter_deg,
            illumination_pct=moon.illumination_pct,
            phase_deg=moon.phase_deg,
            phase_name=moon.phase_name,
            is_above_horizon=True,
        )

    # Place a synthetic aircraft at 35,000 ft directly on the line of sight to the Moon
    import math

    alt_ft = 35000.0
    alt_m = alt_ft * 0.3048
    alt_rad = math.radians(moon.altitude_deg)
    ground_dist_m = (alt_m - cfg.elevation_m) / math.tan(alt_rad)
    # Ground point from observer along the Moon's azimuth:
    cross_lat, cross_lon, _ = extrapolate_position(
        cfg.latitude,
        cfg.longitude,
        alt_m,
        ground_speed_kt=ground_dist_m / KNOTS_TO_MPS,
        track_deg=moon.azimuth_deg,
        vertical_rate_fpm=0.0,
        dt_s=1.0,
    )

    # Fly perpendicular to the line of sight, starting 40s before the crossing point
    track_deg = (moon.azimuth_deg + 90.0) % 360.0
    reverse_track = (track_deg + 180.0) % 360.0
    gs_kt = 450.0
    start_lat, start_lon, _ = extrapolate_position(
        cross_lat,
        cross_lon,
        alt_m,
        ground_speed_kt=gs_kt,
        track_deg=reverse_track,
        vertical_rate_fpm=0.0,
        dt_s=40.0,
    )

    detector = TransitDetector(
        obs_lat=cfg.latitude,
        obs_lon=cfg.longitude,
        obs_alt_m=cfg.elevation_m,
        alert_margin_deg=cfg.alert_margin_deg,
        predict_horizon_s=cfg.predict_horizon_s,
        min_moon_alt_deg=0.0,
        log_margin_deg=max(cfg.log_margin_deg, 10.0),
    )
    notifier = (
        TransitNotifier(
            log_file=cfg.log_file if write_log else None,
            passes_file=cfg.passes_file if write_log else None,
            notify_macos=False,
            webhook_url=None,
            pushover_app_token=cfg.pushover_app_token if send_pushover else None,
            pushover_user_key=cfg.pushover_user_key if send_pushover else None,
            pushover_device=cfg.pushover_device,
            pushover_sound=cfg.pushover_sound,
            pushover_priority=cfg.pushover_priority,
            pushover_events=cfg.pushover_events,
            adsb_url=cfg.adsb_url,
        )
        if (write_log or send_pushover)
        else None
    )
    if send_pushover and (not cfg.pushover_app_token or not cfg.pushover_user_key):
        console.print(
            "[yellow]Warning: --pushover specified, but pushover_app_token / "
            "pushover_user_key are missing in config.[/]"
        )

    table = Table(
        title="Lunar Transit Simulation (virtual aircraft KLM999 at 35,000 ft, 450 kt)",
        box=box.ROUNDED,
    )
    table.add_column("T (s)", justify="right")
    table.add_column("Separation", justify="right")
    table.add_column("Moon Radii", justify="right")
    table.add_column("Prediction", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Event", justify="center")

    for step_s in (0, 10, 20, 30, 38, 39, 40, 41, 42, 45, 50):
        lat, lon, cur_alt_m = extrapolate_position(
            start_lat,
            start_lon,
            alt_m,
            ground_speed_kt=gs_kt,
            track_deg=track_deg,
            vertical_rate_fpm=0.0,
            dt_s=float(step_s),
        )
        ac = Aircraft(
            hex="484161",
            flight="KLM999",
            registration="PH-BVA",
            aircraft_type="B77W",
            category="A5",
            lat=lat,
            lon=lon,
            altitude_m=cur_alt_m,
            altitude_ft=alt_ft,
            altitude_source="geom",
            ground_speed_kt=gs_kt,
            track_deg=track_deg,
            vertical_rate_fpm=0.0,
            seen_pos_s=0.0,
            squawk="1000",
            rssi=-12.0,
        )
        sim_time = now + timedelta(seconds=step_s)
        sightings, events = detector.evaluate(moon, [ac], now_utc=sim_time)
        s = sightings[0]
        if notifier and events:
            for ev in events:
                notifier.notify(ev)
        pred = (
            f"{s.min_predicted_sep_deg:.3f}° in {s.time_to_closest_s:.1f}s"
            if s.min_predicted_sep_deg is not None and s.time_to_closest_s is not None
            else "-"
        )
        ev_str = ", ".join(e.event_type for e in events) if events else "-"
        table.add_row(
            f"{step_s - 40:+d}s",
            f"{s.angular_sep_deg:.3f}°",
            f"{s.moon_radii:.2f}R",
            pred,
            _status_badge(s.status),
            ev_str,
        )

    console.print(table)
    if render_path is not None:
        records = detector.pop_completed_passes() + detector.flush_active_passes()
        if records:
            is_svg = render_path.suffix.lower() == ".svg"
            content = render_pass_svg(records[0]) if is_svg else render_pass_html(records[0])
            _write_render_output(render_path, is_svg, content)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunacross",
        description="Compute exact topocentric Moon position and monitor ADS-B aircraft for lunar transits.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.toml")
    parser.add_argument("--lat", type=float, default=None, help="Observer latitude (degrees N)")
    parser.add_argument("--lon", type=float, default=None, help="Observer longitude (degrees E)")
    parser.add_argument("--elevation", type=float, default=None, help="Observer elevation in meters")
    parser.add_argument("--url", type=str, default=None, help="URL to aircraft.json")
    parser.add_argument("--margin", type=float, default=None, help="Alert margin around Moon (degrees)")
    parser.add_argument("--horizon", type=float, default=None, help="Prediction horizon (seconds)")
    parser.add_argument("--web-port", type=int, default=None, help="Enable Live Sky Web UI on port")
    parser.add_argument(
        "--sync-receiver",
        action="store_true",
        help="Fetch lat/lon from the ADS-B server's receiver.json",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("moon", help="Show the current Moon position for your location")
    sub.add_parser("check", help="One-shot check of Moon position and live aircraft")
    sub.add_parser("monitor", help="Live dashboard monitoring continuously for lunar transits")
    sub.add_parser("test-pushover", help="Send a test notification via Pushover")

    render_parser = sub.add_parser(
        "render",
        help="Generate an observer-perspective SVG/HTML sky visualization",
    )
    render_parser.add_argument("--list", action="store_true", help="List recorded close passes")
    render_parser.add_argument("--live", action="store_true", help="Render current live sky snapshot")
    render_parser.add_argument("--pass-id", type=str, default=None, help="Recorded pass_id to render")
    render_parser.add_argument("--output", "-o", type=Path, default=None, help="Output .html or .svg path")

    sim_parser = sub.add_parser("simulate", help="Simulate an aircraft crossing in front of the Moon")
    sim_parser.add_argument("--log", action="store_true", help="Write simulated events to the log file")
    sim_parser.add_argument(
        "--pushover",
        action="store_true",
        help="Also send simulated transit alerts via Pushover",
    )
    sim_parser.add_argument(
        "--render",
        type=Path,
        default=None,
        help="Render the simulated pass to an .html or .svg file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = {
        "latitude": args.lat,
        "longitude": args.lon,
        "elevation_m": args.elevation,
        "adsb_url": args.url,
        "alert_margin_deg": args.margin,
        "predict_horizon_s": args.horizon,
        "web_port": args.web_port,
    }
    cfg = load_config(config_path=args.config, cli_overrides=overrides)
    cfg = _sync_receiver_if_requested(cfg, args.sync_receiver)

    if args.command == "moon":
        return _cmd_moon(cfg)
    if args.command == "check":
        return _cmd_check(cfg)
    if args.command == "monitor":
        return _cmd_monitor(cfg)
    if args.command == "test-pushover":
        return _cmd_test_pushover(cfg)
    if args.command == "render":
        return _cmd_render(
            cfg,
            list_only=args.list,
            live=args.live,
            pass_id=args.pass_id,
            output=args.output,
        )
    if args.command == "simulate":
        return _cmd_simulate(
            cfg,
            write_log=args.log,
            send_pushover=args.pushover,
            render_path=args.render,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
