"""CLI en live Rich-dashboard voor LunaCross."""

from __future__ import annotations

import argparse
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
from lunacross.notifier import TransitNotifier

console = Console()


def _format_local(dt_utc: datetime | None) -> str:
    if dt_utc is None:
        return "Niet binnen 36u"
    local_dt = dt_utc.astimezone()
    return local_dt.strftime("%d-%m %H:%M:%S (%Z)")


def _status_badge(status: SightingStatus) -> Text:
    if status == "TRANSIT":
        return Text("TRANSIT!", style="bold white on red")
    if status == "NEAR_MISS":
        return Text("VLAKBIJ", style="bold black on yellow")
    if status == "INCOMING":
        return Text("NADERT", style="bold black on cyan")
    return Text("Volgen", style="dim")


def _build_moon_panel(
    moon: MoonPosition,
    rise_set: MoonRiseSet,
    cfg: Config,
) -> Panel:
    local_time = moon.timestamp_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    horizon_txt = (
        "[bold green]Boven de horizon[/]"
        if moon.is_above_horizon
        else "[bold red]Onder de horizon[/]"
    )
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row(
        "Locatie:",
        f"{cfg.latitude:.6f}° N, {cfg.longitude:.6f}° E ({cfg.elevation_m:.0f} m)",
        "Tijd:",
        f"{local_time}",
    )
    table.add_row(
        "Hoogte (Alt):",
        f"{moon.altitude_deg:+.2f}°",
        "Azimut (Az):",
        f"{moon.azimuth_deg:.2f}° ({compass_direction(moon.azimuth_deg)})",
    )
    table.add_row(
        "Hoekdiameter:",
        f"{moon.angular_diameter_deg:.3f}° ({moon.angular_diameter_deg * 60:.1f}')",
        "Afstand:",
        f"{moon.distance_km:,.0f} km".replace(",", "."),
    )
    table.add_row(
        "Fase:",
        f"{moon.phase_name} ({moon.illumination_pct:.1f}% verlicht)",
        "Zichtbaarheid:",
        horizon_txt,
    )
    table.add_row(
        "Volgende opkomst:",
        _format_local(rise_set.next_rise_utc),
        "Volgende ondergang:",
        _format_local(rise_set.next_set_utc),
    )

    return Panel(
        table,
        title="[bold yellow]Maanpositie (Topocentrisch)[/]",
        border_style="yellow",
    )


def _build_aircraft_table(
    sightings: list[AircraftSighting],
    top_n: int,
    total_raw: int,
    feed_status: str,
) -> Table:
    table = Table(
        title=f"Dichtstbijzijnde vliegtuigen t.o.v. de maan  |  {feed_status}  |  In de lucht: {len(sightings)} (totaal feed: {total_raw})",
        box=box.ROUNDED,
        expand=True,
    )
    table.add_column("Vlucht", style="bold")
    table.add_column("Type/Cat", justify="center")
    table.add_column("Hoogte", justify="right")
    table.add_column("Afstand", justify="right")
    table.add_column("Hemel (Az / Alt)", justify="right")
    table.add_column("Δ Maan", justify="right")
    table.add_column("Voorspelling", justify="right")
    table.add_column("Status", justify="center")

    shown = sightings[:top_n]
    if not shown:
        table.add_row("-", "-", "-", "-", "-", "-", "-", Text("Geen vliegtuigen", style="dim"))
        return table

    for s in shown:
        ac = s.aircraft
        type_str = ac.aircraft_type or ac.category or "-"
        alt_str = f"{ac.altitude_ft:,.0f} ft".replace(",", ".")
        dist_str = f"{s.sky_pos.slant_range_km:.1f} km"
        sky_str = f"{s.sky_pos.azimuth_deg:5.1f}° / {s.sky_pos.altitude_deg:+5.1f}°"
        sep_str = f"{s.angular_sep_deg:.2f}° ({s.moon_radii:.1f}R)"
        if s.min_predicted_sep_deg is not None and s.time_to_closest_s is not None:
            pred_str = f"{s.min_predicted_sep_deg:.2f}° over {s.time_to_closest_s:.0f}s"
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
            f"Maan staat onder de ingestelde drempelwaarde ({moon.altitude_deg:.1f}° < {min_moon_alt_deg:.1f}°). "
            "Vliegtuigen worden wel gevolgd; transit-waarschuwingen starten zodra de maan boven de horizon staat.",
            style="yellow",
        )
        return Panel(content, title="Transit-status", border_style="yellow")

    lines: list[str] = []
    for s in active:
        ac = s.aircraft
        if s.status == "TRANSIT":
            lines.append(
                f"[bold red]NU VOOR DE MAAN:[/] {ac.flight} ({ac.hex}) | "
                f"scheiding {s.angular_sep_deg:.3f}° ({s.moon_radii:.2f}R) | {ac.altitude_ft:.0f} ft"
            )
        elif s.status == "NEAR_MISS":
            lines.append(
                f"[bold yellow]VLAKBIJ DE MAAN:[/] {ac.flight} ({ac.hex}) | "
                f"scheiding {s.angular_sep_deg:.2f}° | {ac.altitude_ft:.0f} ft"
            )
        elif s.status == "INCOMING" and s.time_to_closest_s is not None:
            lines.append(
                f"[bold cyan]NADERT MAAN:[/] {ac.flight} ({ac.hex}) | "
                f"min. scheiding {s.min_predicted_sep_deg:.2f}° over {s.time_to_closest_s:.0f}s"
            )

    if recent_events:
        if lines:
            lines.append("")
        lines.append("[bold]Recente meldingen:[/]")
        for ev in recent_events[-5:]:
            ts = ev.timestamp_utc.astimezone().strftime("%H:%M:%S")
            lines.append(
                f"  {ts}  [{ev.event_type}] {ev.sighting.aircraft.flight} — {ev.sighting.angular_sep_deg:.2f}°"
            )

    if not lines:
        lines.append("[dim]Geen actieve of naderende maanovergangen binnen de horizon.[/]")

    border = "red" if any(s.status == "TRANSIT" for s in active) else "cyan"
    return Panel("\n".join(lines), title="Maanovergang Alerts", border_style=border)


def _sync_receiver_if_requested(cfg: Config, sync_receiver: bool) -> Config:
    if not sync_receiver:
        return cfg
    try:
        with AdsbClient(cfg.adsb_url) as client:
            coords = client.fetch_receiver_location()
        if coords is not None:
            cfg.latitude, cfg.longitude = coords
            console.print(
                f"[green]Ontvangerlocatie gesynchroniseerd vanaf server: "
                f"{cfg.latitude:.6f}, {cfg.longitude:.6f}[/]"
            )
        else:
            console.print("[yellow]Waarschuwing: receiver.json bevatte geen lat/lon.[/]")
    except httpx.HTTPError as exc:
        console.print(f"[yellow]Kon receiver.json niet ophalen ({exc}); gebruik config-locatie.[/]")
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
        console.print(f"[bold red]Fout bij ophalen ADS-B data ({cfg.adsb_url}):[/] {exc}")
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
            f"[green]Verbonden[/] ({cfg.adsb_url})",
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
    )
    notifier = TransitNotifier(
        log_file=cfg.log_file,
        notify_macos=cfg.notify_macos,
        webhook_url=cfg.webhook_url,
        pushover_app_token=cfg.pushover_app_token,
        pushover_user_key=cfg.pushover_user_key,
        pushover_device=cfg.pushover_device,
        pushover_sound=cfg.pushover_sound,
        pushover_priority=cfg.pushover_priority,
        pushover_events=cfg.pushover_events,
        adsb_url=cfg.adsb_url,
    )

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
            if len(recent_events) > 20:
                del recent_events[:-20]
            feed_status = "[bold green]Live verbonden[/]"
            total_raw = snapshot.total_raw
        except httpx.HTTPError as exc:
            sightings = []
            events = []
            total_raw = 0
            feed_status = f"[bold red]Verbindingsfout: {exc}[/]"

        return moon, cached_rise_set, sightings, events, total_raw, feed_status

    with AdsbClient(cfg.adsb_url) as client:
        if console.is_terminal:
            with Live(console=console, refresh_per_second=2, screen=False) as live:
                try:
                    while True:
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
                while True:
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
                                else "geen"
                            )
                            console.print(
                                f"[{ts}] Maan Alt={moon.altitude_deg:+.2f}° Az={moon.azimuth_deg:.2f}° "
                                f"| vliegtuigen={len(sightings)}/{total_raw} | dichtstbij={closest}"
                            )
                            last_heartbeat_at = moon.timestamp_utc
                        sys.stdout.flush()
                    time.sleep(cfg.poll_interval_s)
            except KeyboardInterrupt:
                pass

    console.print("\n[dim]Monitoring gestopt.[/]")
    return 0


def _cmd_test_pushover(cfg: Config) -> int:
    if not (cfg.pushover_app_token and cfg.pushover_user_key):
        console.print(
            "[bold yellow]Pushover is nog niet geconfigureerd.[/]\n"
            "Vul [cyan]pushover_app_token[/] en [cyan]pushover_user_key[/] in via:\n"
            "  1. [bold]config.toml[/] (kopieer [cyan]config.example.toml[/] naar [cyan]config.toml[/]), of\n"
            "  2. Omgevingsvariabelen [cyan]LUNACROSS_PUSHOVER_APP_TOKEN[/] en "
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
            title="LunaCross testmelding",
            message=(
                f"Pushover werkt! Actuele maan: alt {moon.altitude_deg:+.1f}°, "
                f"az {moon.azimuth_deg:.1f}° ({compass_direction(moon.azimuth_deg)}), "
                f"{moon.illumination_pct:.1f}% verlicht."
            ),
            url=cfg.adsb_url.replace("/data/aircraft.json", "/"),
            url_title="Open tar1090 radar",
            raise_on_error=True,
        )
    except Exception as exc:
        console.print(f"[bold red]Fout bij versturen van Pushover-melding:[/] {exc}")
        return 1

    console.print("[bold green]Testmelding succesvol verstuurd via Pushover![/]")
    return 0


def _cmd_simulate(cfg: Config, write_log: bool = False, send_pushover: bool = False) -> int:
    tracker = MoonTracker(cfg.latitude, cfg.longitude, cfg.elevation_m, cfg.ephemeris_dir)
    now = datetime.now(timezone.utc)
    moon = tracker.get_position(now)

    using_synthetic_moon = not moon.is_above_horizon or moon.altitude_deg < 10.0
    if using_synthetic_moon:
        console.print(
            f"[yellow]Let op: de echte maan staat momenteel op {moon.altitude_deg:+.1f}° "
            "hoogte. Voor de simulatie gebruiken we tijdelijk een testmaan op +35.0° hoogte, 180.0° azimut.[/]"
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

    # Plaats een synthetisch vliegtuig op 35.000 ft precies op de kijklijn naar de maan
    import math

    alt_ft = 35000.0
    alt_m = alt_ft * 0.3048
    alt_rad = math.radians(moon.altitude_deg)
    ground_dist_m = (alt_m - cfg.elevation_m) / math.tan(alt_rad)
    # Grondpunt vanaf waarnemer richting maanazimut:
    cross_lat, cross_lon, _ = extrapolate_position(
        cfg.latitude,
        cfg.longitude,
        alt_m,
        ground_speed_kt=ground_dist_m / KNOTS_TO_MPS,  # 1s stap over exact ground_dist_m
        track_deg=moon.azimuth_deg,
        vertical_rate_fpm=0.0,
        dt_s=1.0,
    )

    # Laat het vliegtuig loodrecht op de kijklijn vliegen, beginnend 40s vóór het kruispunt
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
    )
    notifier = (
        TransitNotifier(
            log_file=cfg.log_file if write_log else None,
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
            "[yellow]Waarschuwing: --pushover opgegeven, maar pushover_app_token / "
            "pushover_user_key ontbreken in config.[/]"
        )

    table = Table(
        title="Simulatie maanovergang (virtueel vliegtuig KLM999 op 35.000 ft, 450 kt)",
        box=box.ROUNDED,
    )
    table.add_column("T (s)", justify="right")
    table.add_column("Scheiding", justify="right")
    table.add_column("Maanstralen", justify="right")
    table.add_column("Voorspelling", justify="right")
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
            f"{s.min_predicted_sep_deg:.3f}° over {s.time_to_closest_s:.1f}s"
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunacross",
        description="Bepaal de exacte maanpositie en monitor ADS-B vliegtuigen op maanovergangen.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Pad naar config.toml")
    parser.add_argument("--lat", type=float, default=None, help="Breedtegraad (graden N)")
    parser.add_argument("--lon", type=float, default=None, help="Lengtegraad (graden E)")
    parser.add_argument("--elevation", type=float, default=None, help="Hoogte in meters")
    parser.add_argument("--url", type=str, default=None, help="URL naar aircraft.json")
    parser.add_argument("--margin", type=float, default=None, help="Alert-marge rond maan (graden)")
    parser.add_argument("--horizon", type=float, default=None, help="Voorspellingshorizon (sec)")
    parser.add_argument(
        "--sync-receiver",
        action="store_true",
        help="Haal lat/lon op uit receiver.json van de ADS-B server",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("moon", help="Toon de actuele maanpositie voor jouw locatie")
    sub.add_parser("check", help="Eenmalige check van maanpositie en live vliegtuigen")
    sub.add_parser("monitor", help="Live dashboard dat continu op maanovergangen monitort")
    sub.add_parser("test-pushover", help="Stuur een testnotificatie via Pushover")
    sim_parser = sub.add_parser("simulate", help="Simuleer een vliegtuigtransit voor de maan")
    sim_parser.add_argument("--log", action="store_true", help="Schrijf simulatie-events naar logbestand")
    sim_parser.add_argument(
        "--pushover",
        action="store_true",
        help="Verstuur de gesimuleerde transitmeldingen ook via Pushover",
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
    if args.command == "simulate":
        return _cmd_simulate(cfg, write_log=args.log, send_pushover=args.pushover)
    return 2


if __name__ == "__main__":
    sys.exit(main())
