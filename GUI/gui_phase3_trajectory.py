"""
gui_phase3_trajectory.py
GUI for Phase 3 Inertial Dead Reckoning.
Ported EXACTLY from Phase 3 monolith (Inertial_Dead_Reckoning.py) lines 698-1228.
No hardcoded values; matches original exact behaviour.
"""

import time
import os
from datetime import datetime

import numpy as np
from core.math_utils import wrap

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

# ── Styling Constants ───────────────────────────────────────────────────────
FIG_SIZE = (16, 9)
COLOR_ESKF = "#1f77b4"
UI_PAUSE = 0.001

TABLE_ROW_LABELS = [
    "Initial position (m)", "Final position (m)",
    "Initial velocity (m/s)", "Final velocity (m/s)",
]
TABLE_HEADER_COLOR = "#B4C6E7"
TABLE_ROW_COLOR = "#D9E1F2"
TS_LINE_COLORS = ["#ca1a0e", "#38ec50", "#3274d8", "#8614f0"]
TS_HISTORY_MAXLEN = 300

BOLD_LW = 2.4
BOLD_LINE_LW = 1.5
TABLE_EDGE_LW = 2.0

PURE_ROLL_HOLD_ROLL_DEG = 45.0
PURE_ROLL_HOLD_PITCH_DEG = 20.0
PURE_ROLL_HOLD_SECONDS = 3.0
PURE_ROLL_YAW_DAMPING_RATE = 0.3

def _bold_box(ax, lw=BOLD_LW):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color("black")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def _bold_table(table, lw=TABLE_EDGE_LW):
    for cell in table.get_celld().values():
        cell.set_edgecolor("black")
        cell.set_linewidth(lw)
    return table


# ── Setup GUI ──────────────────────────────────────────────────────────────
def setup_gui():
    plt.ion()
    fig = plt.figure(figsize=FIG_SIZE)
    fig.canvas.manager.set_window_title("LIVE STRAPDOWN INS (MERGED)")

    outer = fig.add_gridspec(1, 2, width_ratios=[0.57, 0.43], wspace=0.12)

    left = outer[0, 0].subgridspec(
        2, 1, height_ratios=[0.65, 0.35], hspace=0.22)

    # MAP
    ax_map = fig.add_subplot(left[0, 0])
    ax_map.set_title("TOP-VIEW TRAJECTORY (NAV FRAME)", fontweight="bold")
    ax_map.set_xlabel("X (m, East/local)")
    ax_map.set_ylabel("Y (m, North/local)")
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect("equal", adjustable="datalim")
    for spine in ax_map.spines.values():
        spine.set_linewidth(BOLD_LW)

    traj_line, = ax_map.plot([], [], color=COLOR_ESKF, lw=BOLD_LINE_LW, alpha=0.9, label="Sensor")
    ax_map.plot(0, 0, marker="o", color="black", markersize=6, label="Origin")
    body_dot, = ax_map.plot([], [], 'ro', markersize=10, zorder=5, label="Body")

    map_coord_text = ax_map.text(
        0.02, 0.02, "", transform=ax_map.transAxes,
        fontsize=10, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black",
                  linewidth=1.4, alpha=0.9))
    ax_map.legend(loc="upper right", fontsize=8)

    bot_left = left[1, 0].subgridspec(
        2, 1, height_ratios=[0.18, 0.82], hspace=0.18)

    rpy_row = bot_left[0, 0].subgridspec(1, 3, wspace=0.15)
    ax_roll_box  = fig.add_subplot(rpy_row[0, 0])
    ax_pitch_box = fig.add_subplot(rpy_row[0, 1])
    ax_yaw_box   = fig.add_subplot(rpy_row[0, 2])

    roll_box_text = ax_roll_box.text(
        0.5, 0.5, "Roll:\n--°", transform=ax_roll_box.transAxes,
        fontsize=11, fontweight="bold", ha="center", va="center")
    pitch_box_text = ax_pitch_box.text(
        0.5, 0.5, "Pitch:\n--°", transform=ax_pitch_box.transAxes,
        fontsize=11, fontweight="bold", ha="center", va="center")
    yaw_box_text = ax_yaw_box.text(
        0.5, 0.5, "Yaw:\n--°", transform=ax_yaw_box.transAxes,
        fontsize=11, fontweight="bold", ha="center", va="center")

    for ax in (ax_roll_box, ax_pitch_box, ax_yaw_box):
        _bold_box(ax)

    ax_table = fig.add_subplot(bot_left[1, 0])
    ax_table.axis("off")
    cell_text = [[label, "--"] for label in TABLE_ROW_LABELS]
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Parameter", "Value"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.05, 0.05, 0.9, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width([0, 1])
    for (r, c), cell in table.get_celld().items():
        cell.get_text().set_ha("center")
        cell.get_text().set_va("center")
        if r == 0:
            cell.set_facecolor(TABLE_HEADER_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TABLE_ROW_COLOR)
    _bold_table(table)

    right = outer[0, 1].subgridspec(
        4, 1, height_ratios=[0.20, 0.20, 0.30, 0.30], hspace=0.30)

    ts_top = right[0, 0].subgridspec(1, 2, wspace=0.30)
    ax_bx = fig.add_subplot(ts_top[0, 0])
    ax_bx.set_title("Gyro Bias X (°/s)", fontsize=9, fontweight="bold", color="black")
    ln_bx, = ax_bx.plot([], [], color=TS_LINE_COLORS[0], lw=BOLD_LINE_LW)

    ax_by = fig.add_subplot(ts_top[0, 1])
    ax_by.set_title("Gyro Bias Y (°/s)", fontsize=9, fontweight="bold", color="black")
    ln_by, = ax_by.plot([], [], color=TS_LINE_COLORS[1], lw=BOLD_LINE_LW)

    ts_bot = right[1, 0].subgridspec(1, 2, wspace=0.30)
    ax_bz = fig.add_subplot(ts_bot[0, 0])
    ax_bz.set_title("Gyro Bias Z (°/s)", fontsize=9, fontweight="bold", color="black")
    ln_bz, = ax_bz.plot([], [], color=TS_LINE_COLORS[2], lw=BOLD_LINE_LW)

    ax_vel = fig.add_subplot(ts_bot[0, 1])
    ax_vel.set_title("Velocity (m/s)", fontsize=9, fontweight="bold", color="black")
    ln_vel, = ax_vel.plot([], [], color=TS_LINE_COLORS[3], lw=BOLD_LINE_LW)
    ax_vel.set_xlabel("t (s)", fontsize=7)

    for ax in (ax_bx, ax_by, ax_bz, ax_vel):
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        for spine in ax.spines.values():
            spine.set_linewidth(BOLD_LW)

    hc = right[2, 0].subgridspec(1, 2, width_ratios=[0.50, 0.50], wspace=0.18)

    ax_horizon = fig.add_subplot(hc[0, 0])
    ax_horizon.set_title("ATTITUDE HORIZON", fontsize=9, fontweight="bold")
    ax_horizon.set_xlim(-1.3, 1.3)
    ax_horizon.set_ylim(-1.3, 1.3)
    ax_horizon.set_aspect("equal")
    ax_horizon.axis("off")

    horizon_radius = 1.0
    horizon_clip = mpatches.Circle((0, 0), horizon_radius, transform=ax_horizon.transData)
    big = horizon_radius * 6.0
    sky_patch = mpatches.Rectangle((-big, 0.0), 2 * big, big,
                                    facecolor="#87CEEB", edgecolor=None, zorder=1)
    ground_patch = mpatches.Rectangle((-big, -big), 2 * big, big,
                                       facecolor="#8B5A2B", edgecolor=None, zorder=1)
    sky_patch.set_clip_path(horizon_clip)
    ground_patch.set_clip_path(horizon_clip)
    ax_horizon.add_patch(sky_patch)
    ax_horizon.add_patch(ground_patch)

    horizon_line, = ax_horizon.plot([-big, big], [0, 0], color="white", lw=2.0, zorder=2)
    outline = mpatches.Circle((0, 0), horizon_radius, fill=False,
                               edgecolor="black", lw=BOLD_LW, zorder=3)
    ax_horizon.add_patch(outline)
    ax_horizon.plot([-0.5, -0.15], [0, 0], color="yellow", lw=3.5, zorder=4)
    ax_horizon.plot([0.15, 0.5], [0, 0], color="yellow", lw=3.5, zorder=4)
    ax_horizon.plot(0, 0, marker="o", color="yellow", markersize=5, zorder=4)

    ax_rp_table = fig.add_subplot(hc[0, 1])
    ax_rp_table.axis("off")
    rp_cell_text = [["Roll", "--"], ["Pitch", "--"]]
    rp_table = ax_rp_table.table(
        cellText=rp_cell_text, colLabels=["Parameter", "Value"],
        cellLoc="center", loc="center")
    rp_table.auto_set_font_size(False)
    rp_table.set_fontsize(9)
    rp_table.scale(1.0, 1.9)
    for (r, c), cell in rp_table.get_celld().items():
        if r == 0:
            cell.set_facecolor(TABLE_HEADER_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TABLE_ROW_COLOR)
    _bold_table(rp_table)

    cc = right[3, 0].subgridspec(1, 2, width_ratios=[0.50, 0.50], wspace=0.18)

    ax_compass = fig.add_subplot(cc[0, 0])
    ax_compass.set_title("COMPASS (TRUE)", fontsize=9, fontweight="bold")
    ax_compass.set_xlim(-1.3, 1.3)
    ax_compass.set_ylim(-1.3, 1.3)
    ax_compass.set_aspect("equal")
    ax_compass.axis("off")

    bezel_outer = mpatches.Circle((0, 0), 1.20, fill=False, edgecolor="#333333", lw=2.5, zorder=1)
    bezel_inner = mpatches.Circle((0, 0), 1.15, fill=False, edgecolor="#555555", lw=1.0, zorder=1)
    compass_face = mpatches.Circle((0, 0), 1.15, facecolor="#F5F5F0", edgecolor="none", zorder=0)
    ax_compass.add_patch(compass_face)
    ax_compass.add_patch(bezel_outer)
    ax_compass.add_patch(bezel_inner)

    for cdeg in range(0, 360, 10):
        mpl_rad = np.deg2rad(90.0 - cdeg)
        is_major = (cdeg % 30 == 0)
        r_inner = 0.95 if is_major else 1.00
        r_outer = 1.13
        tick_lw = 1.5 if is_major else 0.7
        x0, y0 = r_inner * np.cos(mpl_rad), r_inner * np.sin(mpl_rad)
        x1, y1 = r_outer * np.cos(mpl_rad), r_outer * np.sin(mpl_rad)
        ax_compass.plot([x0, x1], [y0, y1], color="#333333", lw=tick_lw, zorder=2)

    for cdeg in range(0, 360, 30):
        mpl_rad = np.deg2rad(90.0 - cdeg)
        rx, ry = 0.84 * np.cos(mpl_rad), 0.84 * np.sin(mpl_rad)
        if cdeg not in (0, 90, 180, 270):
            ax_compass.text(rx, ry, f"{cdeg}", fontsize=6, ha="center", va="center",
                             color="#555555", zorder=3)

    cardinal_data = [("N", 0, "#d62728"), ("E", 90, "#333333"),
                     ("S", 180, "#333333"), ("W", 270, "#333333")]
    for label, cdeg, color in cardinal_data:
        mpl_rad = np.deg2rad(90.0 - cdeg)
        rx, ry = 0.82 * np.cos(mpl_rad), 0.82 * np.sin(mpl_rad)
        ax_compass.text(rx, ry, label, fontsize=11, ha="center", va="center",
                         fontweight="bold", color=color, zorder=3)

    for label, cdeg in (("NE", 45), ("SE", 135), ("SW", 225), ("NW", 315)):
        mpl_rad = np.deg2rad(90.0 - cdeg)
        rx, ry = 0.70 * np.cos(mpl_rad), 0.70 * np.sin(mpl_rad)
        ax_compass.text(rx, ry, label, fontsize=6, ha="center", va="center",
                         color="#888888", zorder=3)

    needle_half_w = 0.06
    needle_n_verts = np.array([[0.0, 0.55], [-needle_half_w, 0.0], [needle_half_w, 0.0]])
    needle_s_verts = np.array([[0.0, -0.55], [-needle_half_w, 0.0], [needle_half_w, 0.0]])
    needle_north = mpatches.Polygon(needle_n_verts, closed=True,
                                     facecolor="#d62728", edgecolor="#8B0000",
                                     lw=0.8, zorder=6)
    needle_south = mpatches.Polygon(needle_s_verts, closed=True,
                                     facecolor="white", edgecolor="#555555",
                                     lw=0.8, zorder=6)
    ax_compass.add_patch(needle_north)
    ax_compass.add_patch(needle_south)

    hub = mpatches.Circle((0, 0), 0.045, facecolor="#333333", edgecolor="black",
                           lw=1, zorder=7)
    ax_compass.add_patch(hub)

    compass_text = ax_compass.text(
        0.5, -0.12, "", transform=ax_compass.transAxes,
        fontsize=9, va="top", ha="center")

    ax_hd_table = fig.add_subplot(cc[0, 1])
    ax_hd_table.axis("off")
    hd_cell_text = [["Heading", "--"]]
    hd_table = ax_hd_table.table(
        cellText=hd_cell_text, colLabels=["Parameter", "Value"],
        cellLoc="center", loc="center")
    hd_table.auto_set_font_size(False)
    hd_table.set_fontsize(9)
    hd_table.scale(1.0, 1.9)
    for (r, c), cell in hd_table.get_celld().items():
        if r == 0:
            cell.set_facecolor(TABLE_HEADER_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(TABLE_ROW_COLOR)
    _bold_table(hd_table)

    return {
        "fig": fig,
        "ax_map": ax_map, "traj_line": traj_line,
        "body_dot": body_dot,
        "map_coord_text": map_coord_text,
        "roll_box_text": roll_box_text, "pitch_box_text": pitch_box_text,
        "yaw_box_text": yaw_box_text,
        "ax_horizon": ax_horizon, "sky_patch": sky_patch,
        "ground_patch": ground_patch, "horizon_line": horizon_line,
        "horizon_clip": horizon_clip, "horizon_radius": horizon_radius,
        "rp_table": rp_table,
        "ax_compass": ax_compass,
        "needle_north": needle_north, "needle_south": needle_south,
        "needle_n_verts": needle_n_verts, "needle_s_verts": needle_s_verts,
        "compass_text": compass_text,
        "hd_table": hd_table,
        "ax_bx": ax_bx, "ln_bx": ln_bx,
        "ax_by": ax_by, "ln_by": ln_by,
        "ax_bz": ax_bz, "ln_bz": ln_bz,
        "ax_vel": ax_vel, "ln_vel": ln_vel,
        "table": table,
    }

# ── Updates ────────────────────────────────────────────────────────────────
def update_rpy_text(gui, roll_deg, pitch_deg, yaw_deg):
    gui["roll_box_text"].set_text(f"Roll:\n{roll_deg:+.2f}°")
    gui["pitch_box_text"].set_text(f"Pitch:\n{pitch_deg:+.2f}°")
    gui["yaw_box_text"].set_text(f"Yaw:\n{yaw_deg:+.2f}°")

def update_map(gui, pos, p_hist, yaw_rad):
    xs = [p[0] for p in p_hist]
    ys = [p[1] for p in p_hist]
    gui["traj_line"].set_data(xs, ys)

    x, y = pos[0], pos[1]
    gui["body_dot"].set_data([x], [y])

    gui["map_coord_text"].set_text(
        f"Current: X={x:+.2f} m, Y={y:+.2f} m\nHeading: {np.rad2deg(yaw_rad):+.1f} deg")

    ax = gui["ax_map"]
    if len(xs) > 1:
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        span = max(span_x, span_y, 0.01)
        margin = max(0.05, 0.1 * span)
    else:
        margin = 0.05

    x_min = min(xs + [x]) - margin
    x_max = max(xs + [x]) + margin
    y_min = min(ys + [y]) - margin
    y_max = max(ys + [y]) + margin

    pos_bbox = ax.get_position()
    fig_w, fig_h = gui["fig"].get_size_inches()
    ax_w = pos_bbox.width * fig_w
    ax_h = pos_bbox.height * fig_h
    aspect = ax_w / ax_h if ax_h != 0 else 1.0

    w = x_max - x_min
    h = y_max - y_min
    if h > 0 and w / h < aspect:
        w = h * aspect
    elif aspect > 0:
        h = w / aspect

    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0

    ax.set_xlim(x_mid - w / 2.0, x_mid + w / 2.0)
    ax.set_ylim(y_mid - h / 2.0, y_mid + h / 2.0)

def update_horizon(gui, roll_deg, pitch_deg):
    r = gui["horizon_radius"]
    pitch_offset = np.clip(pitch_deg / 90.0, -1.0, 1.0) * r * 0.9
    trans = (mtransforms.Affine2D()
             .translate(0, pitch_offset)
             .rotate_deg_around(0, 0, roll_deg)
             + gui["ax_horizon"].transData)
    gui["sky_patch"].set_transform(trans)
    gui["ground_patch"].set_transform(trans)
    gui["horizon_line"].set_transform(trans)
    gui["sky_patch"].set_clip_path(gui["horizon_clip"])
    gui["ground_patch"].set_clip_path(gui["horizon_clip"])

    cells = gui["rp_table"].get_celld()
    cells[(1, 1)].get_text().set_text(f"{roll_deg:+.2f}°")
    cells[(2, 1)].get_text().set_text(f"{pitch_deg:+.2f}°")

def update_compass(gui, heading_deg):
    angle_rad = np.deg2rad(-heading_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    gui["needle_north"].set_xy((rot @ gui["needle_n_verts"].T).T)
    gui["needle_south"].set_xy((rot @ gui["needle_s_verts"].T).T)

    cells = gui['hd_table'].get_celld()
    cells[(1, 1)].get_text().set_text(f"{heading_deg:+.2f}")

def true_heading_deg(eskf, yaw_rad):
    return np.rad2deg(wrap(yaw_rad + eskf.yaw_initial))

def update_timeseries(gui, t_hist, bx_hist, by_hist, bz_hist, vel_hist):
    for key, y_hist in (("bx", bx_hist), ("by", by_hist), ("bz", bz_hist), ("vel", vel_hist)):
        ln = gui[f"ln_{key}"]
        ax = gui[f"ax_{key}"]
        ln.set_data(t_hist, y_hist)
        ax.relim()
        ax.autoscale_view()

def update_table(gui, pos, vel):
    if gui.get("init_pos") is None:
        gui["init_pos"] = pos.copy()
        gui["init_vel"] = vel.copy()
    init_pos = gui["init_pos"]
    init_vel = gui["init_vel"]

    values = [
        f"({init_pos[0]:+.2f}, {init_pos[1]:+.2f}, {init_pos[2]:+.2f})",
        f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})",
        f"({init_vel[0]:+.2f}, {init_vel[1]:+.2f}, {init_vel[2]:+.2f})",
        f"({vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f})",
    ]
    cells = gui["table"].get_celld()
    for i, v in enumerate(values):
        cells[(i + 1, 1)].get_text().set_text(v)

def update_yaw_display_damped(gui, roll_deg, pitch_deg, yaw_deg):
    if abs(pitch_deg) > 75.0:
        roll_disp = roll_deg * np.exp(-6)
        yaw_disp = yaw_deg * np.exp(-6)
    elif abs(pitch_deg) > 15.0 or (abs(pitch_deg) > 60.0 and abs(pitch_deg) < 75.0):
        roll_disp = roll_deg * np.exp(-3)
        yaw_disp = yaw_deg * np.exp(-3)
    else:
        roll_disp = roll_deg
        yaw_disp = yaw_deg

    now = time.time()
    is_pure_roll = (abs(roll_deg) > PURE_ROLL_HOLD_ROLL_DEG
                     and abs(pitch_deg) < PURE_ROLL_HOLD_PITCH_DEG)

    if is_pure_roll:
        if gui.get("_pure_roll_start") is None:
            gui["_pure_roll_start"] = now
            gui["_pure_roll_yaw_anchor"] = None
        hold_time = now - gui["_pure_roll_start"]
        if hold_time >= PURE_ROLL_HOLD_SECONDS:
            if gui.get("_pure_roll_yaw_anchor") is None:
                gui["_pure_roll_yaw_anchor"] = yaw_disp
            anchor = gui["_pure_roll_yaw_anchor"]
            decay = np.exp(-PURE_ROLL_YAW_DAMPING_RATE * (hold_time - PURE_ROLL_HOLD_SECONDS))
            yaw_disp = anchor + (yaw_disp - anchor) * decay
    else:
        gui["_pure_roll_start"] = None
        gui["_pure_roll_yaw_anchor"] = None

    update_rpy_text(gui, roll_disp, pitch_deg, yaw_disp)
    update_horizon(gui, roll_disp, pitch_deg)

def keep_windows_responsive():
    plt.pause(UI_PAUSE)

def save_gui_plot(gui, out_dir="Trajectory Plots"):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"GUI_screen_{timestamp_str}.png")
    gui["fig"].savefig(out_file, dpi=200, facecolor="white", bbox_inches='tight')
    print(f"Plot saved to {out_file}")
