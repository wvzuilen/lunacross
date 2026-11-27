"""Observer-perspective sky visualization (horizon, Moon phase, aircraft route) as SVG/HTML."""

from __future__ import annotations

import html
import math
from datetime import datetime
from typing import Iterable

from lunacross.config import Config
from lunacross.detector import AircraftSighting, PassPoint, PassRecord
from lunacross.geometry import aircraft_altaz, extrapolate_position, sky_tangent_offset_deg
from lunacross.moon import MoonPosition, compass_direction, compute_north_angle_deg
from lunacross.weather import WeatherSnapshot

_CLASS_COLORS = {
    "TRANSIT": "#ef4444",
    "NEAR_MISS": "#f59e0b",
    "CLOSE_PASS": "#38bdf8",
    "INCOMING": "#22d3ee",
    "TRACKING": "#94a3b8",
}

_CARDINAL_LABELS: dict[int, str] = {
    0: "N",
    45: "NE",
    90: "E",
    135: "SE",
    180: "S",
    225: "SW",
    270: "W",
    315: "NW",
}


def _wrap_az_delta(az_deg: float, center_az_deg: float) -> float:
    return ((az_deg - center_az_deg + 180.0) % 360.0) - 180.0


def _fmt_local_time(iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_utc).astimezone()
        return dt.strftime("%H:%M:%S")
    except Exception:
        return iso_utc


def _fmt_local_datetime(iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return iso_utc


def _lunar_maria_svg(cx: float, cy: float, r: float, north_angle_deg: float, opacity: float) -> str:
    """Render recognizable near-side lunar maria rotated by topocentric Lunar North angle."""
    if r < 7.0:
        return ""
    # Normalized selenographic coordinates (dx right, dy down when Lunar North is straight up)
    maria_ellipses: list[tuple[float, float, float, float, float]] = [
        (-0.22, -0.26, 0.26, 0.22, -15.0),  # Mare Imbrium
        (0.16, -0.22, 0.17, 0.15, 10.0),    # Mare Serenitatis
        (0.32, -0.02, 0.20, 0.16, 20.0),    # Mare Tranquillitatis
        (0.62, -0.14, 0.11, 0.14, -10.0),   # Mare Crisium
        (0.46, 0.20, 0.16, 0.19, 15.0),     # Mare Foecunditatis
        (-0.46, 0.02, 0.24, 0.38, 12.0),    # Oceanus Procellarum
        (-0.14, 0.34, 0.19, 0.15, -8.0),    # Mare Nubium
        (-0.42, 0.36, 0.13, 0.12, 0.0),     # Mare Humorum
    ]
    shapes: list[str] = [
        f'<g transform="rotate({north_angle_deg:.2f} {cx:.2f} {cy:.2f})" opacity="{opacity:.3f}">'
    ]
    for dx, dy, rx_n, ry_n, tilt in maria_ellipses:
        ex = cx + dx * r
        ey = cy + dy * r
        erx = rx_n * r
        ery = ry_n * r
        shapes.append(
            f'<ellipse cx="{ex:.2f}" cy="{ey:.2f}" rx="{erx:.2f}" ry="{ery:.2f}" '
            f'transform="rotate({tilt:.1f} {ex:.2f} {ey:.2f})" fill="#64748b"/>'
        )
    # Subtle bright ray craters (Tycho in south, Copernicus mid-west)
    shapes.append(
        f'<circle cx="{cx - 0.08 * r:.2f}" cy="{cy + 0.58 * r:.2f}" r="{max(1.2, 0.035 * r):.2f}" fill="#ffffff" opacity="0.55"/>'
    )
    shapes.append(
        f'<circle cx="{cx - 0.24 * r:.2f}" cy="{cy - 0.06 * r:.2f}" r="{max(1.0, 0.03 * r):.2f}" fill="#ffffff" opacity="0.45"/>'
    )
    shapes.append("</g>")
    return "".join(shapes)


def _lit_phase_path_d(cx: float, cy: float, r: float, illum_fraction: float) -> str:
    """Build canonical SVG path (lit on the RIGHT / +x side) for illuminated fraction f in [0, 1]."""
    f = max(0.0, min(1.0, illum_fraction))
    top_y = cy - r
    bot_y = cy + r
    # Outer sunlit semicircle from top (0, -r) via right (+r, 0) to bottom (0, +r): sweep=1
    limb_arc = f"M {cx:.2f} {top_y:.2f} A {r:.2f} {r:.2f} 0 0 1 {cx:.2f} {bot_y:.2f}"
    rx = r * abs(2.0 * f - 1.0)
    if rx < 1e-3 * r:
        return f"{limb_arc} L {cx:.2f} {top_y:.2f} Z"
    # Return from bottom (0, +r) to top (0, -r):
    # - Gibbous (f > 0.5): bulge into LEFT (-x) half -> angle increases (+90° -> +180° -> +270°) -> sweep=1
    # - Crescent (f < 0.5): bulge into RIGHT (+x) half -> angle decreases (+90° -> 0° -> -90°) -> sweep=0
    sweep_term = 1 if f > 0.5 else 0
    return f"{limb_arc} A {rx:.2f} {r:.2f} 0 0 {sweep_term} {cx:.2f} {top_y:.2f} Z"


def _moon_phase_svg(
    cx: float,
    cy: float,
    r: float,
    phase_deg: float,
    uid: str,
    *,
    illumination_pct: float | None = None,
    bright_limb_angle_deg: float | None = None,
    north_angle_deg: float = 0.0,
    tag: str = "m",
) -> str:
    """Render a realistic topocentric Moon disc with true sky tilt, limb darkening, and maria."""
    angle = phase_deg % 360.0
    if illumination_pct is not None:
        f = max(0.0, min(1.0, illumination_pct / 100.0))
    else:
        f = 0.5 * (1.0 - math.cos(math.radians(angle)))

    # Default bright-limb direction if not provided: right (90°) when waxing, left (270°) when waning
    if bright_limb_angle_deg is None:
        limb_cw = 90.0 if angle < 180.0 else 270.0
    else:
        limb_cw = bright_limb_angle_deg % 360.0

    # Canonical path is lit at 90° CW from Up (+x right); rotate by (limb_cw - 90°)
    rot_deg = limb_cw - 90.0
    disc_clip_id = f"{uid}-disc-{tag}"
    lit_clip_id = f"{uid}-lit-{tag}"

    parts = [
        f'<defs><clipPath id="{disc_clip_id}"><circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/></clipPath>',
    ]
    if f > 0.005:
        if f >= 0.995:
            parts.append(
                f'<clipPath id="{lit_clip_id}"><circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r + 0.5:.2f}"/></clipPath>'
            )
        else:
            path_d = _lit_phase_path_d(cx, cy, r, f)
            parts.append(
                f'<clipPath id="{lit_clip_id}"><path d="{path_d}" transform="rotate({rot_deg:.2f} {cx:.2f} {cy:.2f})"/></clipPath>'
            )
    parts.append("</defs>")

    # Subtle outer lunar glow scaled by illumination
    if f > 0.05:
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r * 1.04:.2f}" fill="#f8fafc" '
            f'opacity="{0.08 + 0.20 * f:.2f}" filter="url(#{uid}-glow)"/>'
        )

    # Disc interior: earthshine night side + sunlit hemisphere with maria
    parts.append(f'<g clip-path="url(#{disc_clip_id})">')
    parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="url(#{uid}-earthshine)"/>')
    parts.append(_lunar_maria_svg(cx, cy, r, north_angle_deg, opacity=0.06))

    if f > 0.005:
        parts.append(f'<g clip-path="url(#{lit_clip_id})">')
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="url(#{uid}-lunar-lit)"/>')
        parts.append(_lunar_maria_svg(cx, cy, r, north_angle_deg, opacity=0.23))
        parts.append("</g>")

    parts.append("</g>")
    parts.append(
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="none" stroke="#cbd5e1" stroke-width="0.85" stroke-opacity="0.5"/>'
    )
    return "".join(parts)


def _svg_defs(uid: str, left_clip: tuple[float, float, float, float], right_clip: tuple[float, float, float, float]) -> str:
    lx, ly, lw, lh = left_clip
    rx, ry, rw, rh = right_clip
    return f"""
<defs>
  <linearGradient id="{uid}-sky" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#020617"/>
    <stop offset="65%" stop-color="#0f172a"/>
    <stop offset="100%" stop-color="#1e293b"/>
  </linearGradient>
  <linearGradient id="{uid}-ground" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#0f172a"/>
    <stop offset="100%" stop-color="#020617"/>
  </linearGradient>
  <radialGradient id="{uid}-earthshine" cx="50%" cy="50%" r="52%">
    <stop offset="0%" stop-color="#1e293b"/>
    <stop offset="82%" stop-color="#0f172a"/>
    <stop offset="100%" stop-color="#1e293b"/>
  </radialGradient>
  <radialGradient id="{uid}-lunar-lit" cx="50%" cy="50%" r="52%">
    <stop offset="0%" stop-color="#fefce8"/>
    <stop offset="68%" stop-color="#f1f5f9"/>
    <stop offset="90%" stop-color="#cbd5e1"/>
    <stop offset="100%" stop-color="#94a3b8"/>
  </radialGradient>
  <filter id="{uid}-glow" x="-40%" y="-40%" width="180%" height="180%">
    <feGaussianBlur stdDeviation="2.5" result="blur"/>
    <feMerge>
      <feMergeNode in="blur"/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>
  <clipPath id="{uid}-clip-left">
    <rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="8"/>
  </clipPath>
  <clipPath id="{uid}-clip-right">
    <rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" rx="8"/>
  </clipPath>
</defs>"""


def _render_horizon_panel(
    *,
    uid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    center_az_deg: float,
    half_az_span_deg: float,
    min_alt_deg: float,
    max_alt_deg: float,
    moon_alt_deg: float,
    moon_az_deg: float,
    moon_phase_deg: float,
    moon_illumination_pct: float | None = None,
    moon_bright_limb_deg: float | None = None,
    moon_north_angle_deg: float = 0.0,
    moon_phase_badge: str | None = None,
    moon_label: str,
    body_elements: str,
) -> str:
    def map_pt(alt_deg: float, az_deg: float) -> tuple[float, float]:
        d_az = _wrap_az_delta(az_deg, center_az_deg)
        px = x + ((d_az + half_az_span_deg) / (2.0 * half_az_span_deg)) * w
        py = y + h - ((alt_deg - min_alt_deg) / (max_alt_deg - min_alt_deg)) * h
        return px, py

    # Sky background inside clip
    _, horizon_y = map_pt(0.0, center_az_deg)
    ground_h = max(0.0, (y + h) - horizon_y)

    parts = [
        f'<g clip-path="url(#{uid}-clip-left)">',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="url(#{uid}-sky)"/>',
    ]

    # Altitude grid lines (15° increments)
    alt_step = 15
    start_alt = int(math.ceil(min_alt_deg / alt_step) * alt_step)
    for alt_tick in range(start_alt, int(max_alt_deg) + 1, alt_step):
        if alt_tick == 0:
            continue
        _, gy = map_pt(float(alt_tick), center_az_deg)
        parts.append(
            f'<line x1="{x:.1f}" y1="{gy:.1f}" x2="{x + w:.1f}" y2="{gy:.1f}" '
            f'stroke="#334155" stroke-width="0.8" stroke-dasharray="4 4" opacity="0.7"/>'
        )
        parts.append(
            f'<text x="{x + 8:.1f}" y="{gy - 4:.1f}" fill="#64748b" font-size="10" font-family="ui-monospace, monospace">{alt_tick:+d}°</text>'
        )

    # Azimuth compass ticks every 15°
    az_start = int(math.floor((center_az_deg - half_az_span_deg) / 15.0) * 15)
    az_end = int(math.ceil((center_az_deg + half_az_span_deg) / 15.0) * 15)
    for az_raw in range(az_start, az_end + 15, 15):
        az_norm = az_raw % 360
        d_az = _wrap_az_delta(float(az_norm), center_az_deg)
        if abs(d_az) > half_az_span_deg:
            continue
        gx, _ = map_pt(0.0, float(az_norm))
        is_card = az_norm in _CARDINAL_LABELS
        parts.append(
            f'<line x1="{gx:.1f}" y1="{y:.1f}" x2="{gx:.1f}" y2="{horizon_y:.1f}" '
            f'stroke="#334155" stroke-width="{1.0 if is_card else 0.6}" stroke-dasharray="3 5" opacity="{0.65 if is_card else 0.35}"/>'
        )

    # Ground below horizon
    if ground_h > 0.0:
        parts.append(
            f'<rect x="{x:.1f}" y="{horizon_y:.1f}" width="{w:.1f}" height="{ground_h:.1f}" fill="url(#{uid}-ground)"/>'
        )
    # Glowing horizon line
    parts.append(
        f'<line x1="{x:.1f}" y1="{horizon_y:.1f}" x2="{x + w:.1f}" y2="{horizon_y:.1f}" '
        f'stroke="#38bdf8" stroke-width="1.6" opacity="0.8"/>'
    )
    parts.append(
        f'<text x="{x + w - 10:.1f}" y="{horizon_y - 6:.1f}" text-anchor="end" fill="#38bdf8" '
        f'font-size="10" font-family="system-ui, sans-serif" opacity="0.85">HORIZON (0°)</text>'
    )

    # Compass labels along horizon
    for az_raw in range(az_start, az_end + 15, 15):
        az_norm = az_raw % 360
        d_az = _wrap_az_delta(float(az_norm), center_az_deg)
        if abs(d_az) > half_az_span_deg - 3.0:
            continue
        gx, _ = map_pt(0.0, float(az_norm))
        lbl = _CARDINAL_LABELS.get(az_norm, f"{az_norm}°")
        is_card = az_norm in _CARDINAL_LABELS
        parts.append(
            f'<line x1="{gx:.1f}" y1="{horizon_y - 4:.1f}" x2="{gx:.1f}" y2="{horizon_y + 4:.1f}" stroke="#94a3b8" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{horizon_y + 17:.1f}" text-anchor="middle" '
            f'fill="{"#e2e8f0" if is_card else "#94a3b8"}" font-weight="{"600" if is_card else "400"}" '
            f'font-size="{"11" if is_card else "10"}" font-family="ui-monospace, monospace">{lbl}</text>'
        )

    # Moon marker in wide horizon view
    mx, my = map_pt(moon_alt_deg, moon_az_deg)
    parts.append(
        _moon_phase_svg(
            mx,
            my,
            9.5,
            moon_phase_deg,
            uid,
            illumination_pct=moon_illumination_pct,
            bright_limb_angle_deg=moon_bright_limb_deg,
            north_angle_deg=moon_north_angle_deg,
            tag="horiz",
        )
    )
    parts.append(
        f'<text x="{mx + 14:.1f}" y="{my + 4:.1f}" fill="#fef3c7" font-size="11" '
        f'font-weight="600" font-family="system-ui, sans-serif">{html.escape(moon_label)}</text>'
    )

    # Custom aircraft/track body elements
    parts.append(body_elements)

    # Enlarged topocentric Moon phase preview inset in top-right sky corner
    if moon_phase_badge:
        bx, by, bw, bh = x + w - 98.0, y + 10.0, 88.0, 88.0
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="8" '
            f'fill="#020617" fill-opacity="0.78" stroke="#334155" stroke-width="0.9"/>'
        )
        parts.append(
            _moon_phase_svg(
                bx + bw / 2.0,
                by + 36.0,
                25.0,
                moon_phase_deg,
                uid,
                illumination_pct=moon_illumination_pct,
                bright_limb_angle_deg=moon_bright_limb_deg,
                north_angle_deg=moon_north_angle_deg,
                tag="inset",
            )
        )
        parts.append(
            f'<text x="{bx + bw / 2.0:.1f}" y="{by + bh - 8.0:.1f}" text-anchor="middle" '
            f'fill="#cbd5e1" font-size="9.5" font-family="ui-monospace, monospace">{html.escape(moon_phase_badge)}</text>'
        )

    parts.append("</g>")
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="none" stroke="#334155" stroke-width="1.2"/>'
    )
    return "\n".join(parts)


def _render_zoom_panel(
    *,
    uid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    half_fov_deg: float,
    moon_diameter_deg: float,
    moon_phase_deg: float,
    moon_illumination_pct: float | None = None,
    moon_bright_limb_deg: float | None = None,
    moon_north_angle_deg: float = 0.0,
    title: str,
    footer_lines: list[str],
    body_elements: str,
) -> str:
    cx = x + w / 2.0
    cy = y + h / 2.0 - 10.0
    usable_r = min(w, h - 40.0) * 0.45
    px_per_deg = usable_r / half_fov_deg
    moon_r_px = max(4.0, (moon_diameter_deg / 2.0) * px_per_deg)

    parts = [
        f'<g clip-path="url(#{uid}-clip-right)">',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#020617"/>',
    ]

    # Angular reference rings
    ring_steps = [1.0, 2.0, 4.0, 6.0]
    for deg in ring_steps:
        if deg > half_fov_deg * 1.05:
            continue
        rr = deg * px_per_deg
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rr:.1f}" fill="none" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>'
        )
        parts.append(
            f'<text x="{cx + 4:.1f}" y="{cy - rr + 11:.1f}" fill="#475569" font-size="9" font-family="ui-monospace, monospace">{deg:.0f}°</text>'
        )

    # Crosshair axes (East/West, Up/Down)
    parts.append(
        f'<line x1="{x + 12:.1f}" y1="{cy:.1f}" x2="{x + w - 12:.1f}" y2="{cy:.1f}" stroke="#1e293b" stroke-width="0.9"/>'
    )
    parts.append(
        f'<line x1="{cx:.1f}" y1="{y + 28:.1f}" x2="{cx:.1f}" y2="{y + h - 46:.1f}" stroke="#1e293b" stroke-width="0.9"/>'
    )

    # Moon rendered to exact angular scale and topocentric orientation
    parts.append(
        _moon_phase_svg(
            cx,
            cy,
            moon_r_px,
            moon_phase_deg,
            uid,
            illumination_pct=moon_illumination_pct,
            bright_limb_angle_deg=moon_bright_limb_deg,
            north_angle_deg=moon_north_angle_deg,
            tag="zoom",
        )
    )

    # Aircraft track / positions
    parts.append(body_elements)

    # Panel title & footer
    parts.append(
        f'<text x="{x + 12:.1f}" y="{y + 20:.1f}" fill="#e2e8f0" font-size="12" font-weight="600" font-family="system-ui, sans-serif">{html.escape(title)}</text>'
    )
    parts.append(
        f'<text x="{x + w - 12:.1f}" y="{y + 20:.1f}" text-anchor="end" fill="#64748b" font-size="10" font-family="ui-monospace, monospace">FOV ±{half_fov_deg:.1f}°</text>'
    )
    for idx, line in enumerate(footer_lines[:2]):
        parts.append(
            f'<text x="{x + 12:.1f}" y="{y + h - 22 + idx * 14:.1f}" fill="#94a3b8" font-size="10.5" font-family="ui-monospace, monospace">{html.escape(line)}</text>'
        )

    parts.append("</g>")
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="none" stroke="#334155" stroke-width="1.2"/>'
    )
    return "\n".join(parts)


def render_pass_svg(record: PassRecord, *, width: int = 1120, height: int = 540) -> str:
    """Render an observer-perspective sky view and zoomed lunar inset for a recorded PassRecord."""
    uid = "lp"
    header_h = 54.0
    pad = 16.0
    left_x, left_y = pad, header_h
    left_w = width * 0.64 - pad * 1.5
    left_h = height - header_h - pad
    right_x = left_x + left_w + pad
    right_y = header_h
    right_w = width - right_x - pad
    right_h = left_h

    track = record.track
    center_az = record.moon_az_deg

    max_d_az = max(
        [abs(_wrap_az_delta(pt.aircraft_az_deg, center_az)) for pt in track] + [20.0]
    )
    half_az_span = max(50.0, min(110.0, max_d_az + 12.0))

    max_track_alt = max([pt.aircraft_alt_deg for pt in track] + [record.moon_alt_deg])
    min_track_alt = min([pt.aircraft_alt_deg for pt in track] + [0.0])
    min_alt = min(-6.0, min_track_alt - 4.0)
    max_alt = min(92.0, max(75.0, max_track_alt + 10.0, record.moon_alt_deg + 14.0))

    def map_left(alt_deg: float, az_deg: float) -> tuple[float, float]:
        d_az = _wrap_az_delta(az_deg, center_az)
        px = left_x + ((d_az + half_az_span) / (2.0 * half_az_span)) * left_w
        py = left_y + left_h - ((alt_deg - min_alt) / (max_alt - min_alt)) * left_h
        return px, py

    color = _CLASS_COLORS.get(record.classification, "#38bdf8")

    # Left panel track polyline + closest marker
    left_body: list[str] = []
    if track:
        pts_str = " ".join(
            f"{px:.1f},{py:.1f}"
            for px, py in (map_left(pt.aircraft_alt_deg, pt.aircraft_az_deg) for pt in track)
        )
        left_body.append(
            f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>'
        )
        sx, sy = map_left(track[0].aircraft_alt_deg, track[0].aircraft_az_deg)
        ex, ey = map_left(track[-1].aircraft_alt_deg, track[-1].aircraft_az_deg)
        left_body.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3.5" fill="{color}"/>')
        left_body.append(
            f'<text x="{sx + 6:.1f}" y="{sy - 6:.1f}" fill="#cbd5e1" font-size="10" font-family="ui-monospace, monospace">{_fmt_local_time(track[0].timestamp_utc)}</text>'
        )
        # Arrowhead at end of track
        if len(track) >= 2:
            px_prev, py_prev = map_left(track[-2].aircraft_alt_deg, track[-2].aircraft_az_deg)
            ang = math.atan2(ey - py_prev, ex - px_prev)
            a1x = ex - 8.0 * math.cos(ang - 0.45)
            a1y = ey - 8.0 * math.sin(ang - 0.45)
            a2x = ex - 8.0 * math.cos(ang + 0.45)
            a2y = ey - 8.0 * math.sin(ang + 0.45)
            left_body.append(
                f'<polygon points="{ex:.1f},{ey:.1f} {a1x:.1f},{a1y:.1f} {a2x:.1f},{a2y:.1f}" fill="{color}"/>'
            )
        left_body.append(
            f'<text x="{ex + 6:.1f}" y="{ey + 12:.1f}" fill="#cbd5e1" font-size="10" font-family="ui-monospace, monospace">{_fmt_local_time(track[-1].timestamp_utc)}</text>'
        )

    cx_m, cy_m = map_left(record.closest_aircraft_alt_deg, record.closest_aircraft_az_deg)
    left_body.append(
        f'<circle cx="{cx_m:.1f}" cy="{cy_m:.1f}" r="5.0" fill="#ffffff" stroke="{color}" stroke-width="2.2"/>'
    )
    left_body.append(
        f'<text x="{cx_m + 9:.1f}" y="{cy_m - 8:.1f}" fill="#ffffff" font-size="11" font-weight="600" font-family="system-ui, sans-serif">'
        f'{html.escape(record.flight)} ({record.min_angular_sep_deg:.2f}°)</text>'
    )

    # Right panel zoomed tangent-plane view
    half_fov = max(1.5, min(8.0, record.min_angular_sep_deg * 1.6 + 0.9))
    rcx = right_x + right_w / 2.0
    rcy = right_y + right_h / 2.0 - 10.0
    usable_r = min(right_w, right_h - 40.0) * 0.45
    px_per_deg = usable_r / half_fov

    def map_right(alt_deg: float, az_deg: float) -> tuple[float, float]:
        dx_deg, dy_deg = sky_tangent_offset_deg(alt_deg, az_deg, record.moon_alt_deg, record.moon_az_deg)
        return rcx + dx_deg * px_per_deg, rcy - dy_deg * px_per_deg

    right_body: list[str] = []
    if track:
        z_pts = " ".join(
            f"{zx:.1f},{zy:.1f}"
            for zx, zy in (map_right(pt.aircraft_alt_deg, pt.aircraft_az_deg) for pt in track)
        )
        right_body.append(
            f'<polyline points="{z_pts}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for pt in track:
            if not pt.interpolated:
                zx, zy = map_right(pt.aircraft_alt_deg, pt.aircraft_az_deg)
                right_body.append(f'<circle cx="{zx:.1f}" cy="{zy:.1f}" r="2.2" fill="{color}" opacity="0.8"/>')

    zcx, zcy = map_right(record.closest_aircraft_alt_deg, record.closest_aircraft_az_deg)
    right_body.append(
        f'<circle cx="{zcx:.1f}" cy="{zcy:.1f}" r="5.0" fill="#ffffff" stroke="{color}" stroke-width="2.2"/>'
    )
    right_body.append(
        f'<text x="{zcx + 8:.1f}" y="{zcy - 6:.1f}" fill="#ffffff" font-size="10.5" font-weight="600" font-family="ui-monospace, monospace">{_fmt_local_time(record.closest_utc)}</text>'
    )

    # Header banner
    type_str = f" ({record.aircraft_type})" if record.aircraft_type else ""
    header_title = (
        f"LunaCross Pass Replay — {record.flight}{type_str} [{record.classification}]"
    )
    header_sub = (
        f"Observer {record.observer_lat:.4f}°N, {record.observer_lon:.4f}°E | "
        f"Closest {_fmt_local_datetime(record.closest_utc)} | "
        f"Min sep {record.min_angular_sep_deg:.3f}° ({record.min_moon_radii:.2f}R) | "
        f"{record.closest_altitude_ft:,.0f} ft @ {record.closest_slant_range_km:.1f} km"
    )

    moon_lbl = f"Moon ({record.moon_alt_deg:.1f}° / {record.moon_az_deg:.0f}° {compass_direction(record.moon_az_deg)})"
    footer_lines = [
        f"Min sep: {record.min_angular_sep_deg:.3f}° ({record.min_moon_radii:.2f}R) | Disk: {record.moon_angular_diameter_deg:.3f}°",
        f"Moon: {record.moon_phase_name} ({record.moon_illumination_pct:.0f}% lit) | Track pts: {len(track)}",
    ]

    rec_north = (
        record.moon_north_angle_deg
        if record.moon_north_angle_deg is not None
        else compute_north_angle_deg(record.observer_lat, record.moon_alt_deg, record.moon_az_deg)
    )
    rec_limb = (
        record.moon_bright_limb_deg
        if record.moon_bright_limb_deg is not None
        else (rec_north + (90.0 if (record.moon_phase_deg % 360.0) < 180.0 else 270.0)) % 360.0
    )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" style="background:#020617;border-radius:10px;display:block;">',
            _svg_defs(uid, (left_x, left_y, left_w, left_h), (right_x, right_y, right_w, right_h)),
            f'<text x="{pad:.1f}" y="24" fill="#f8fafc" font-size="16" font-weight="700" font-family="system-ui, sans-serif">{html.escape(header_title)}</text>',
            f'<text x="{pad:.1f}" y="42" fill="#94a3b8" font-size="11.5" font-family="ui-monospace, monospace">{html.escape(header_sub)}</text>',
            _render_horizon_panel(
                uid=uid,
                x=left_x,
                y=left_y,
                w=left_w,
                h=left_h,
                center_az_deg=center_az,
                half_az_span_deg=half_az_span,
                min_alt_deg=min_alt,
                max_alt_deg=max_alt,
                moon_alt_deg=record.moon_alt_deg,
                moon_az_deg=record.moon_az_deg,
                moon_phase_deg=record.moon_phase_deg,
                moon_illumination_pct=record.moon_illumination_pct,
                moon_bright_limb_deg=rec_limb,
                moon_north_angle_deg=rec_north,
                moon_phase_badge=f"{record.moon_illumination_pct:.0f}% lit",
                moon_label=moon_lbl,
                body_elements="\n".join(left_body),
            ),
            _render_zoom_panel(
                uid=uid,
                x=right_x,
                y=right_y,
                w=right_w,
                h=right_h,
                half_fov_deg=half_fov,
                moon_diameter_deg=record.moon_angular_diameter_deg,
                moon_phase_deg=record.moon_phase_deg,
                moon_illumination_pct=record.moon_illumination_pct,
                moon_bright_limb_deg=rec_limb,
                moon_north_angle_deg=rec_north,
                title="Zoomed Lunar View",
                footer_lines=footer_lines,
                body_elements="\n".join(right_body),
            ),
            "</svg>",
        ]
    )


def render_live_snapshot_svg(
    moon: MoonPosition,
    sightings: Iterable[AircraftSighting],
    tracks: dict[str, list[PassPoint]],
    cfg: Config,
    *,
    weather: WeatherSnapshot | None = None,
    predict_s: float | None = None,
    width: int = 1120,
    height: int = 540,
) -> str:
    """Render an observer-perspective live sky SVG with horizon, Moon, and aircraft routes."""
    forecast_s = max(0.0, cfg.predict_horizon_s if predict_s is None else predict_s)
    uid = "ll"
    header_h = 54.0
    pad = 16.0
    left_x, left_y = pad, header_h
    left_w = width * 0.64 - pad * 1.5
    left_h = height - header_h - pad
    right_x = left_x + left_w + pad
    right_y = header_h
    right_w = width - right_x - pad
    right_h = left_h

    sightings_list = list(sightings)
    center_az = moon.azimuth_deg
    half_az_span = 90.0
    min_alt = -6.0
    max_alt = max(75.0, moon.altitude_deg + 14.0)

    def map_left(alt_deg: float, az_deg: float) -> tuple[float, float]:
        d_az = _wrap_az_delta(az_deg, center_az)
        px = left_x + ((d_az + half_az_span) / (2.0 * half_az_span)) * left_w
        py = left_y + left_h - ((alt_deg - min_alt) / (max_alt - min_alt)) * left_h
        return px, py

    half_fov = 6.0
    rcx = right_x + right_w / 2.0
    rcy = right_y + right_h / 2.0 - 10.0
    usable_r = min(right_w, right_h - 40.0) * 0.45
    px_per_deg = usable_r / half_fov

    def map_right_at(alt_deg: float, az_deg: float, dt_s: float = 0.0) -> tuple[float, float]:
        m_alt, m_az = moon.at_offset(dt_s)
        dx_deg, dy_deg = sky_tangent_offset_deg(alt_deg, az_deg, m_alt, m_az)
        return rcx + dx_deg * px_per_deg, rcy - dy_deg * px_per_deg

    obs_hae_m = cfg.elevation_m + cfg.geoid_undulation_m
    left_body: list[str] = []
    right_body: list[str] = []

    for s in sightings_list:
        ac = s.aircraft
        if s.sky_pos.altitude_deg <= -1.0:
            continue
        color = _CLASS_COLORS.get(s.status, "#94a3b8")

        # Past track (solid)
        past_pts = tracks.get(ac.hex, [])
        if len(past_pts) >= 2:
            p_str = " ".join(
                f"{px:.1f},{py:.1f}"
                for px, py in (map_left(pt.aircraft_alt_deg, pt.aircraft_az_deg) for pt in past_pts)
            )
            left_body.append(
                f'<polyline points="{p_str}" fill="none" stroke="{color}" stroke-width="1.6" opacity="0.65"/>'
            )
            rz_str = " ".join(
                f"{zx:.1f},{zy:.1f}"
                for zx, zy in (
                    (
                        rcx + sky_tangent_offset_deg(pt.aircraft_alt_deg, pt.aircraft_az_deg, pt.moon_alt_deg, pt.moon_az_deg)[0] * px_per_deg,
                        rcy - sky_tangent_offset_deg(pt.aircraft_alt_deg, pt.aircraft_az_deg, pt.moon_alt_deg, pt.moon_az_deg)[1] * px_per_deg,
                    )
                    for pt in past_pts
                )
            )
            right_body.append(
                f'<polyline points="{rz_str}" fill="none" stroke="{color}" stroke-width="1.8" opacity="0.7"/>'
            )

        # Future trajectory (dashed)
        if forecast_s > 0.0 and ac.ground_speed_kt and ac.track_deg is not None and ac.ground_speed_kt > 20.0:
            fut_coords: list[tuple[float, float, float]] = [(s.sky_pos.altitude_deg, s.sky_pos.azimuth_deg, 0.0)]
            for step in range(5, int(forecast_s) + 1, 5):
                plat, plon, palt = extrapolate_position(
                    ac.lat,
                    ac.lon,
                    ac.altitude_m,
                    ac.ground_speed_kt,
                    ac.track_deg,
                    ac.vertical_rate_fpm,
                    float(step),
                    track_rate_deg_s=ac.track_rate_deg_s,
                )
                sky_f = aircraft_altaz(
                    cfg.latitude,
                    cfg.longitude,
                    obs_hae_m,
                    plat,
                    plon,
                    palt,
                    temperature_c=weather.temperature_c if weather else 10.0,
                    pressure_hpa=weather.surface_pressure_hpa if weather else 1010.0,
                )
                if sky_f.altitude_deg > -2.0:
                    fut_coords.append((sky_f.altitude_deg, sky_f.azimuth_deg, float(step)))
            if len(fut_coords) >= 2:
                f_str = " ".join(
                    f"{px:.1f},{py:.1f}" for px, py in (map_left(a, z) for a, z, _ in fut_coords)
                )
                left_body.append(
                    f'<polyline points="{f_str}" fill="none" stroke="{color}" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.75"/>'
                )
                rf_str = " ".join(
                    f"{zx:.1f},{zy:.1f}" for zx, zy in (map_right_at(a, z, dt_s) for a, z, dt_s in fut_coords)
                )
                right_body.append(
                    f'<polyline points="{rf_str}" fill="none" stroke="{color}" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.8"/>'
                )

        # Current position dot + label
        ax_px, ay_px = map_left(s.sky_pos.altitude_deg, s.sky_pos.azimuth_deg)
        left_body.append(f'<circle cx="{ax_px:.1f}" cy="{ay_px:.1f}" r="4.0" fill="{color}" stroke="#0f172a" stroke-width="1.2"/>')
        left_body.append(
            f'<text x="{ax_px + 7:.1f}" y="{ay_px + 3:.1f}" fill="#e2e8f0" font-size="10.5" font-weight="600" font-family="ui-monospace, monospace">'
            f'{html.escape(ac.flight)} ({s.angular_sep_deg:.1f}°)</text>'
        )

        zx_px, zy_px = map_right_at(s.sky_pos.altitude_deg, s.sky_pos.azimuth_deg, 0.0)
        right_body.append(f'<circle cx="{zx_px:.1f}" cy="{zy_px:.1f}" r="4.0" fill="{color}" stroke="#0f172a" stroke-width="1.2"/>')
        right_body.append(
            f'<text x="{zx_px + 7:.1f}" y="{zy_px + 3:.1f}" fill="#e2e8f0" font-size="10" font-family="ui-monospace, monospace">{html.escape(ac.flight)}</text>'
        )

    local_str = moon.timestamp_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header_title = f"LunaCross Live Sky View — {local_str}"
    wx_suffix = (
        f" | Sky: {weather.condition_label}, {weather.temperature_c:+.1f}°C"
        if weather is not None
        else ""
    )
    header_sub = (
        f"Observer {cfg.latitude:.4f}°N, {cfg.longitude:.4f}°E | "
        f"Moon {moon.altitude_deg:+.1f}° / {moon.azimuth_deg:.1f}° ({compass_direction(moon.azimuth_deg)}) | "
        f"Airborne tracked: {len(sightings_list)}{wx_suffix}"
    )
    moon_lbl = f"Moon ({moon.altitude_deg:+.1f}° / {moon.azimuth_deg:.0f}° {compass_direction(moon.azimuth_deg)})"
    footer_lines = [
        f"Moon: {moon.phase_name} ({moon.illumination_pct:.1f}% lit, Ø {moon.angular_diameter_deg:.3f}°)",
        f"Solid = observed track | Dashed = +{forecast_s:.0f}s forecast",
    ]

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" style="background:#020617;border-radius:10px;display:block;">',
            _svg_defs(uid, (left_x, left_y, left_w, left_h), (right_x, right_y, right_w, right_h)),
            f'<text x="{pad:.1f}" y="24" fill="#f8fafc" font-size="16" font-weight="700" font-family="system-ui, sans-serif">{html.escape(header_title)}</text>',
            f'<text x="{pad:.1f}" y="42" fill="#94a3b8" font-size="11.5" font-family="ui-monospace, monospace">{html.escape(header_sub)}</text>',
            _render_horizon_panel(
                uid=uid,
                x=left_x,
                y=left_y,
                w=left_w,
                h=left_h,
                center_az_deg=center_az,
                half_az_span_deg=half_az_span,
                min_alt_deg=min_alt,
                max_alt_deg=max_alt,
                moon_alt_deg=moon.altitude_deg,
                moon_az_deg=moon.azimuth_deg,
                moon_phase_deg=moon.phase_deg,
                moon_illumination_pct=moon.illumination_pct,
                moon_bright_limb_deg=moon.effective_bright_limb_deg,
                moon_north_angle_deg=moon.north_angle_deg,
                moon_phase_badge=f"{moon.illumination_pct:.0f}% lit",
                moon_label=moon_lbl,
                body_elements="\n".join(left_body),
            ),
            _render_zoom_panel(
                uid=uid,
                x=right_x,
                y=right_y,
                w=right_w,
                h=right_h,
                half_fov_deg=half_fov,
                moon_diameter_deg=moon.angular_diameter_deg,
                moon_phase_deg=moon.phase_deg,
                moon_illumination_pct=moon.illumination_pct,
                moon_bright_limb_deg=moon.effective_bright_limb_deg,
                moon_north_angle_deg=moon.north_angle_deg,
                title="Zoomed Lunar View (±6°)",
                footer_lines=footer_lines,
                body_elements="\n".join(right_body),
            ),
            "</svg>",
        ]
    )


def _wrap_html(title: str, svg: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 24px; background: #020617; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  .bar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 12px; flex-wrap: wrap; }}
  .btn {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .btn:hover {{ background: #334155; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <strong>{html.escape(title)}</strong>
    <button class="btn" onclick="downloadSvg()">Download SVG</button>
  </div>
  <div id="sky">{svg}</div>
</div>
<script>
function downloadSvg() {{
  const svg = document.querySelector('#sky svg').outerHTML;
  const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'lunacross_sky.svg';
  a.click();
}}
</script>
</body>
</html>
"""


def render_pass_html(record: PassRecord) -> str:
    """Wrap a rendered PassRecord SVG in a standalone interactive HTML document."""
    return _wrap_html(f"LunaCross Pass — {record.flight} ({record.classification})", render_pass_svg(record))


def render_live_snapshot_html(
    moon: MoonPosition,
    sightings: Iterable[AircraftSighting],
    tracks: dict[str, list[PassPoint]],
    cfg: Config,
    *,
    weather: WeatherSnapshot | None = None,
    predict_s: float | None = None,
) -> str:
    """Wrap a rendered live sky SVG in a standalone HTML document."""
    return _wrap_html(
        "LunaCross Live Sky View",
        render_live_snapshot_svg(moon, sightings, tracks, cfg, weather=weather, predict_s=predict_s),
    )
