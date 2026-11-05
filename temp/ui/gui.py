"""
NASA Coding Guidelines Level Architecture: GUI
Handles Matplotlib-based graphical user interface setup and updates.
"""

import time
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from config.config import (
    FIG_SIZE, COLOR_ESKF, TS_LINE_COLORS, BOLD_LW, BOLD_LINE_LW,
    TL_RED_ON, TL_RED_DIM, TL_YELLOW_ON, TL_YELLOW_DIM, TL_GREEN_ON, TL_GREEN_DIM, TL_OFF,
    DISPLAY_MIRROR_X, UI_PAUSE, PITCH_HIGH_DEG, PITCH_MID_DEG,
    PURE_ROLL_HOLD_ROLL_DEG, PURE_ROLL_HOLD_PITCH_DEG, PURE_ROLL_HOLD_SECONDS, PURE_ROLL_YAW_DAMPING_RATE,
    MAG_THRESH_LOW_UT, MAG_THRESH_MID_UT, MAG_THRESH_HIGH_UT
)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#1e293b'
plt.rcParams['figure.facecolor'] = '#1e293b'
plt.rcParams['savefig.facecolor'] = '#1e293b'

def _bold_box(ax, lw=BOLD_LW):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color("white")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def setup_gui():
    plt.ion()
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor("#1e293b")
    fig.canvas.manager.set_window_title("DUAL ESKF PEDESTRIAN DEAD RECKONING")

    outer = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58], wspace=0.20)

    # Trajectory Map
    ax_map = fig.add_subplot(outer[0, 0])
    ax_map.set_title("Trajectory Map", fontsize=14, fontweight="bold")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect("equal", adjustable="datalim")
    for sp in ax_map.spines.values():
        sp.set_linewidth(BOLD_LW)

    traj_line, = ax_map.plot([], [], color=COLOR_ESKF, lw=BOLD_LINE_LW,
                             alpha=0.9, label="Sensor")
    ax_map.plot(0, 0, marker="o", color="white", markersize=6, label="Origin")
    body_dot, = ax_map.plot([], [], 'ro', markersize=10, zorder=5, label="Body")
    map_coord_text = ax_map.text(
        0.02, 0.02, "", transform=ax_map.transAxes, fontsize=9,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="#1e293b", edgecolor="white",
                  linewidth=1.2, alpha=0.9))
    ax_map.legend(loc="upper right", fontsize=8.5)

    # RIGHT HALF
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.22)

    # Right Top: Compass, Horizon, RPY Panel
    right_top = right[0].subgridspec(1, 3, width_ratios=[0.35, 0.35, 0.30], wspace=0.10)

    COMMON_RADIUS = 1.0
    R = COMMON_RADIUS

    # Compass Dial
    ax_compass = fig.add_subplot(right_top[0, 0])
    ax_compass.set_xlim(-1.25, 1.25)
    ax_compass.set_ylim(-1.4, 1.25)
    ax_compass.set_aspect("equal")
    ax_compass.set_title("Heading", fontsize=11, fontweight="bold", color="white")
    ax_compass.axis("off")

    # Compass background and bezels
    bezel_outer = mpatches.Circle((0, 0), R + 0.12, fill=True, facecolor="#9b7e30", edgecolor="#6b5113", lw=1.5, zorder=0)
    bezel_mid = mpatches.Circle((0, 0), R + 0.08, fill=True, facecolor="#cfa44a", edgecolor="#5c4309", lw=1, zorder=0)
    compass_face = mpatches.Circle((0, 0), R, facecolor="#FFFFFF", edgecolor="#2E2519", lw=1.5, zorder=1)
    
    ax_compass.add_patch(bezel_outer)
    ax_compass.add_patch(bezel_mid)
    ax_compass.add_patch(compass_face)

    dark_brown = "#2E2519"
    light_brown = "#a38656"

    # Compass Rose (Star)
    for i in range(8):
        angle = i * 45
        is_cardinal = (i % 2 == 0)
        tip_r = 0.65 * R if is_cardinal else 0.45 * R
        corner_r = 0.12 * R
        
        rad = np.deg2rad(90 - angle)
        rad_cw = np.deg2rad(90 - (angle + 45))
        rad_ccw = np.deg2rad(90 - (angle - 45))
        
        tip = (tip_r * np.cos(rad), tip_r * np.sin(rad))
        corner_cw = (corner_r * np.cos(rad_cw), corner_r * np.sin(rad_cw))
        corner_ccw = (corner_r * np.cos(rad_ccw), corner_r * np.sin(rad_ccw))
        center = (0, 0)
        
        poly_light = mpatches.Polygon([center, tip, corner_ccw], closed=True, facecolor=light_brown, edgecolor='none', zorder=2)
        poly_dark = mpatches.Polygon([center, tip, corner_cw], closed=True, facecolor=dark_brown, edgecolor='none', zorder=2)
        
        ax_compass.add_patch(poly_light)
        ax_compass.add_patch(poly_dark)
        
    # Dots between points
    for i in range(8):
        angle = 22.5 + i * 45
        rad = np.deg2rad(90 - angle)
        ax_compass.plot(0.55 * R * np.cos(rad), 0.55 * R * np.sin(rad), marker='o', markersize=2.5, color=dark_brown, zorder=3)

    # Ticks
    for cdeg in range(0, 360, 2):
        mpl_rad = np.deg2rad(90.0 - cdeg)
        if cdeg % 10 == 0:
            r_in = 0.88 * R
            lw = 1.2
        elif cdeg % 5 == 0:
            r_in = 0.92 * R
            lw = 1.0
        else:
            r_in = 0.95 * R
            lw = 0.6
        r_out = R
        ax_compass.plot(
            [r_in * np.cos(mpl_rad), r_out * np.cos(mpl_rad)],
            [r_in * np.sin(mpl_rad), r_out * np.sin(mpl_rad)],
            color=dark_brown, lw=lw, zorder=3)
            
    # Inner circle for ticks
    ax_compass.add_patch(mpatches.Circle((0, 0), 0.95 * R, fill=False, edgecolor=dark_brown, lw=0.6, zorder=3))

    # Cardinal Labels
    for lbl, cdeg, clr in [("N", 0, "#d62728"), ("E", 90, dark_brown),
                            ("S", 180, dark_brown), ("W", 270, dark_brown)]:
        rad = np.deg2rad(90.0 - cdeg)
        ax_compass.text(0.72 * R * np.cos(rad), 0.72 * R * np.sin(rad), lbl,
                         fontsize=14, ha="center", va="center",
                         fontweight="bold", color=clr, zorder=4)

    # Ordinal Labels
    for lbl, cdeg in [("NE", 45), ("SE", 135), ("SW", 225), ("NW", 315)]:
        rad = np.deg2rad(90.0 - cdeg)
        ax_compass.text(0.66 * R * np.cos(rad), 0.66 * R * np.sin(rad), lbl,
                         fontsize=8.5, ha="center", va="center",
                         fontweight="bold", color=dark_brown, rotation=-cdeg, zorder=4)

    # Needle
    needle_n_verts = np.array([
        [0.0, 0.72 * R], [0.05 * R, 0.38 * R], [0.02 * R, 0.34 * R],
        [0.02 * R, 0.0], [-0.02 * R, 0.0], [-0.02 * R, 0.34 * R], [-0.05 * R, 0.38 * R]
    ])
    needle_s_verts = np.array([
        [0.0, -0.5 * R], [0.05 * R, -0.5 * R], [0.02 * R, -0.42 * R],
        [0.02 * R, 0.0], [-0.02 * R, 0.0], [-0.02 * R, -0.42 * R], [-0.05 * R, -0.5 * R]
    ])
    
    needle_north = mpatches.Polygon(needle_n_verts, closed=True,
                                     facecolor="#d62728", edgecolor="#6b1212",
                                     lw=1.0, zorder=6)
    needle_south = mpatches.Polygon(needle_s_verts, closed=True,
                                     facecolor=dark_brown, edgecolor="#1a140e",
                                     lw=1.0, zorder=6)
                                     
    ax_compass.add_patch(needle_north)
    ax_compass.add_patch(needle_south)
    
    # Center cap
    ax_compass.add_patch(mpatches.Circle((0, 0), 0.08 * R, facecolor="#D4AF37", edgecolor=dark_brown, lw=1.5, zorder=7))

    compass_heading_text = ax_compass.text(
        0.0, -1.3, "Heading\n--", fontsize=11,
        ha="center", va="center", fontweight="bold", color="white")

    # Attitude Horizon
    ax_horizon = fig.add_subplot(right_top[0, 1])
    ax_horizon.set_title("Orientation", fontsize=11, fontweight="bold", color="white")
    ax_horizon.set_xlim(-1.25, 1.25)
    ax_horizon.set_ylim(-1.4, 1.25)
    ax_horizon.set_aspect("equal")
    ax_horizon.axis("off")

    horizon_radius = R

    # Outer bezel
    ax_horizon.add_patch(mpatches.Circle((0, 0), R + 0.12, facecolor="#111111", zorder=0.6))
    
    horizon_clip = mpatches.Circle((0, 0), R, transform=ax_horizon.transData)

    big = horizon_radius * 6.0
    sky_color = "#3db2ff"
    ground_color = "#6b3c1a"
    
    sky_patch = mpatches.Rectangle((-big, 0.0), 2 * big, big, facecolor=sky_color, edgecolor=None, zorder=1)
    ground_patch = mpatches.Rectangle((-big, -big), 2 * big, big, facecolor=ground_color, edgecolor=None, zorder=1)
    sky_patch.set_clip_path(horizon_clip)
    ground_patch.set_clip_path(horizon_clip)
    ax_horizon.add_patch(sky_patch)
    ax_horizon.add_patch(ground_patch)

    horizon_line, = ax_horizon.plot([-big, big], [0, 0], color="white", lw=2.5, zorder=2)
    horizon_line.set_clip_path(horizon_clip)

    roll_ticks = []
    for angle in [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]:
        rad = np.deg2rad(90 - angle)
        length = 0.15 * R if angle in [0, -30, 30, -60, 60] else 0.08 * R
        r_in = R - length
        r_out = R
        line, = ax_horizon.plot(
            [r_in * np.cos(rad), r_out * np.cos(rad)],
            [r_in * np.sin(rad), r_out * np.sin(rad)],
            color="white", lw=2.5, zorder=2
        )
        line.set_clip_path(horizon_clip)
        roll_ticks.append(line)

    # Translucent inner circle
    glass_overlay = mpatches.Circle((0, 0), R, facecolor="black", alpha=0.1, zorder=3)
    ax_horizon.add_patch(glass_overlay)

    # Top white triangle (Roll pointer)
    ptr_w = 0.08 * R
    ptr_h = 0.15 * R
    top_pointer = mpatches.Polygon(
        [[0, R - 0.02*R], [-ptr_w, R + ptr_h], [ptr_w, R + ptr_h]],
        closed=True, facecolor="white", zorder=4
    )
    ax_horizon.add_patch(top_pointer)

    # Reticle / Joystick
    reticle_color = "#4a2a11" # Rich brown, fully opaque
    
    # Bottom shroud (grey cover)
    shroud = mpatches.Polygon(
        [[-R, -R], [R, -R], [0.35*R, -0.65*R], [-0.35*R, -0.65*R]],
        closed=True, facecolor="#3a3e42", edgecolor="#222222", lw=1.5, zorder=5
    )
    ax_horizon.add_patch(shroud)
    
    # Vertical stick
    stick_w = 0.08 * R
    ax_horizon.add_patch(mpatches.Rectangle((-stick_w/2, -0.7*R), stick_w, 0.7*R, facecolor=reticle_color, zorder=6))
    
    # U-shaped arc
    arc_radius = 0.45 * R
    arc = mpatches.Arc((0, -0.05*R), arc_radius*2, arc_radius*2, angle=0, theta1=180, theta2=360, color=reticle_color, lw=4.5, zorder=6)
    ax_horizon.add_patch(arc)
    
    # Diagonal supports
    ax_horizon.plot([0, -0.15*R], [-0.4*R, -0.1*R], color=reticle_color, lw=4, zorder=6)
    ax_horizon.plot([0, 0.15*R], [-0.4*R, -0.1*R], color=reticle_color, lw=4, zorder=6)
    
    # Center orange dot
    ax_horizon.add_patch(mpatches.Circle((0, 0), 0.02 * R, facecolor="#FFB300", zorder=6))
    
    # Orange wings
    wing_w = 0.35 * R
    wing_h = 0.025 * R
    gap = 0.05 * R
    ax_horizon.add_patch(mpatches.Rectangle((-gap - wing_w, 0), wing_w, wing_h, facecolor="#FFB300", edgecolor="black", lw=0.5, zorder=6))
    ax_horizon.add_patch(mpatches.Rectangle((gap, 0), wing_w, wing_h, facecolor="#FFB300", edgecolor="black", lw=0.5, zorder=6))

    # Silver knob in the shroud
    knob_y = -0.75 * R
    ax_horizon.add_patch(mpatches.Circle((0, knob_y), 0.13*R, facecolor="#cccccc", edgecolor="#666666", lw=2, zorder=7))
    ax_horizon.add_patch(mpatches.Circle((0, knob_y), 0.1*R, facecolor="#eeeeee", edgecolor="none", zorder=7))
    ax_horizon.add_patch(mpatches.Circle((0, knob_y), 0.05*R, facecolor="#ffffff", edgecolor="none", zorder=7))

    # Outer ring
    ax_horizon.add_patch(mpatches.Circle((0, 0), R, fill=False, edgecolor="#111111", lw=8, zorder=8))

    horizon_rp_text = ax_horizon.text(
        0.0, -1.3, "Roll  --\nPitch --", fontsize=10,
        ha="center", va="center", fontweight="bold", color="white")

    # R/P/Y Publish Side Panel
    ax_rpy = fig.add_subplot(right_top[0, 2])
    ax_rpy.set_xlim(0, 10)
    ax_rpy.set_ylim(0, 10)
    ax_rpy.axis("off")
    _bold_box(ax_rpy, lw=1.8)

    roll_text = ax_rpy.text(5, 7.8, "Roll:   --", fontsize=14,
                             fontweight="bold", ha="center", va="center",
                             color="white")
    pitch_text = ax_rpy.text(5, 5.0, "Pitch:  --", fontsize=14,
                              fontweight="bold", ha="center", va="center",
                              color="white")
    yaw_text = ax_rpy.text(5, 2.2, "Yaw:    --", fontsize=14,
                            fontweight="bold", ha="center", va="center",
                            color="white")

    # Bottom Right
    right_bot = right[1].subgridspec(1, 2, width_ratios=[0.40, 0.60], wspace=0.20)

    # Gyro Bias and Velocity 4x1 Grid
    gbias_gs = right_bot[0, 0].subgridspec(4, 1, hspace=0.40)
    ax_gbx = fig.add_subplot(gbias_gs[0])
    ax_gby = fig.add_subplot(gbias_gs[1])
    ax_gbz = fig.add_subplot(gbias_gs[2])
    ax_vel  = fig.add_subplot(gbias_gs[3])

    ax_gbx.set_title("Gyro Bias (deg/s) & Velocity", fontsize=11, fontweight="bold", pad=6)
    ax_gbx.set_ylabel("Gbx", fontsize=8.5, fontweight="bold")
    ax_gby.set_ylabel("Gby", fontsize=8.5, fontweight="bold")
    ax_gbz.set_ylabel("Gbz", fontsize=8.5, fontweight="bold")
    ax_vel.set_ylabel("V (m/s)", fontsize=8.5, fontweight="bold")
    ax_vel.set_xlabel("t (s)", fontsize=8.5)

    ln_gbx, = ax_gbx.plot([], [], color=TS_LINE_COLORS[0], lw=BOLD_LINE_LW)
    ln_gby, = ax_gby.plot([], [], color=TS_LINE_COLORS[1], lw=BOLD_LINE_LW)
    ln_gbz, = ax_gbz.plot([], [], color=TS_LINE_COLORS[2], lw=BOLD_LINE_LW)
    ln_vel,  = ax_vel.plot([], [], color=TS_LINE_COLORS[3], lw=BOLD_LINE_LW)

    for ax in (ax_gbx, ax_gby, ax_gbz, ax_vel):
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
        for sp in ax.spines.values():
            sp.set_linewidth(BOLD_LW)

    # New Circular Console
    ax_console = fig.add_subplot(right_bot[0, 1])
    ax_console.set_xlim(-1.05, 1.05)
    ax_console.set_ylim(-1.05, 1.05)
    ax_console.axis("off")
    ax_console.set_aspect("equal")

    # Background of the console must be cream white
    console_bg = mpatches.Circle((0, 0), 1.05, facecolor="#F5F5DC", edgecolor="#CCCCCC", lw=3, zorder=0)
    ax_console.add_patch(console_bg)

    # Magnetic Intensity Signals (High, Medium, Low) in upper half
    mag_high_circle = mpatches.Circle((-0.45, 0.45), 0.18, facecolor=TL_OFF, edgecolor="black", lw=1.5, zorder=1)
    mag_medium_circle = mpatches.Circle((0.0, 0.45), 0.18, facecolor=TL_OFF, edgecolor="black", lw=1.5, zorder=1)
    mag_low_circle = mpatches.Circle((0.45, 0.45), 0.18, facecolor=TL_OFF, edgecolor="black", lw=1.5, zorder=1)
    
    ax_console.add_patch(mag_high_circle)
    ax_console.add_patch(mag_medium_circle)
    ax_console.add_patch(mag_low_circle)

    ax_console.text(-0.45, 0.20, "High", fontsize=11, fontweight="bold", color="black", ha="center", va="center")
    ax_console.text(0.0, 0.20, "Medium", fontsize=11, fontweight="bold", color="black", ha="center", va="center")
    ax_console.text(0.45, 0.20, "Low", fontsize=11, fontweight="bold", color="black", ha="center", va="center")

    def draw_shoe(ax, x, y, angle_deg, facecolor):
        rad = np.deg2rad(angle_deg)
        dist = 0.28
        dx = dist * np.sin(rad)
        dy = -dist * np.cos(rad)
        sole = mpatches.Ellipse((x, y), 0.20, 0.35, angle=angle_deg, facecolor=facecolor, edgecolor="black", lw=1.5, zorder=2)
        heel = mpatches.Ellipse((x + dx, y + dy), 0.15, 0.20, angle=angle_deg, facecolor=facecolor, edgecolor="black", lw=1.5, zorder=2)
        ax.add_patch(sole)
        ax.add_patch(heel)
        return sole, heel

    brown_shoe_color = "#4a2a11"
    idle_shoe_sole, idle_shoe_heel = draw_shoe(ax_console, -0.15, -0.45, 0, brown_shoe_color)
    active_shoe_sole, active_shoe_heel = draw_shoe(ax_console, 0.15, -0.20, 0, brown_shoe_color)

    return {
        "fig": fig,
        "ax_map": ax_map, "traj_line": traj_line,
        "body_dot": body_dot, "map_coord_text": map_coord_text,
        "ax_compass": ax_compass,
        "needle_north": needle_north, "needle_south": needle_south,
        "needle_n_verts": needle_n_verts, "needle_s_verts": needle_s_verts,
        "compass_heading_text": compass_heading_text,
        "ax_horizon": ax_horizon, "sky_patch": sky_patch,
        "ground_patch": ground_patch, "horizon_line": horizon_line,
        "horizon_clip": horizon_clip, "horizon_radius": horizon_radius,
        "horizon_rp_text": horizon_rp_text, "roll_ticks": roll_ticks,
        "roll_text": roll_text, "pitch_text": pitch_text, "yaw_text": yaw_text,
        "ax_gbx": ax_gbx, "ax_gby": ax_gby, "ax_gbz": ax_gbz, "ax_vel": ax_vel,
        "ln_gbx": ln_gbx, "ln_gby": ln_gby, "ln_gbz": ln_gbz, "ln_vel": ln_vel,
        "mag_high_circle": mag_high_circle,
        "mag_minor_circle": mag_medium_circle,
        "mag_low_circle": mag_low_circle,
        "idle_shoe_sole": idle_shoe_sole,
        "idle_shoe_heel": idle_shoe_heel,
        "active_shoe_sole": active_shoe_sole,
        "active_shoe_heel": active_shoe_heel,
    }

def compute_damped_display_angles(roll_deg, pitch_deg, yaw_deg, gui, now):
    roll_disp = roll_deg
    yaw_disp = yaw_deg

    if abs(pitch_deg) > 75.0:
        roll_disp = roll_deg * np.exp(-6)
        yaw_disp = yaw_deg * np.exp(-6)
    elif abs(pitch_deg) > 75.0 and abs(roll_deg) > 75.0:
        yaw_disp = yaw_deg * np.exp(-3)
    elif abs(pitch_deg) > 15.0:
        roll_disp = roll_deg * np.exp(-3)
        yaw_disp = yaw_deg * np.exp(-3)

    is_pure_roll = (abs(roll_deg) > PURE_ROLL_HOLD_ROLL_DEG and
                    abs(pitch_deg) < PURE_ROLL_HOLD_PITCH_DEG)
    if is_pure_roll:
        if gui.get("_pure_roll_start") is None:
            gui["_pure_roll_start"] = now
            gui["_pure_roll_yaw_anchor"] = None
        hold_time = now - gui["_pure_roll_start"]
        if hold_time >= PURE_ROLL_HOLD_SECONDS:
            if gui.get("_pure_roll_yaw_anchor") is None:
                gui["_pure_roll_yaw_anchor"] = yaw_disp
            anchor = gui["_pure_roll_yaw_anchor"]
            decay = 3 * np.exp(-PURE_ROLL_YAW_DAMPING_RATE * (hold_time - PURE_ROLL_HOLD_SECONDS))
            yaw_disp = anchor * decay
    else:
        gui["_pure_roll_start"] = None
        gui["_pure_roll_yaw_anchor"] = None

    return roll_disp, yaw_disp

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
    
    if "roll_ticks" in gui:
        for tick in gui["roll_ticks"]:
            tick.set_transform(trans)
            
    gui["sky_patch"].set_clip_path(gui["horizon_clip"])
    gui["ground_patch"].set_clip_path(gui["horizon_clip"])
    
    if "roll_ticks" in gui:
        for tick in gui["roll_ticks"]:
            tick.set_clip_path(gui["horizon_clip"])
            
    gui["horizon_line"].set_clip_path(gui["horizon_clip"])
    gui["horizon_rp_text"].set_text(f"Roll  {roll_deg:+.1f}°\nPitch {pitch_deg:+.1f}°")

def update_rpy_panel(gui, roll_deg, pitch_deg, yaw_deg):
    gui["roll_text"].set_text(f"Roll:   {roll_deg:+.2f}°")
    gui["pitch_text"].set_text(f"Pitch:  {pitch_deg:+.2f}°")
    gui["yaw_text"].set_text(f"Yaw:    {yaw_deg:+.2f}°")

def update_compass(gui, heading_deg):
    angle_rad = np.deg2rad(-heading_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    gui["needle_north"].set_xy((rot @ gui["needle_n_verts"].T).T)
    gui["needle_south"].set_xy((rot @ gui["needle_s_verts"].T).T)
    gui["compass_heading_text"].set_text(f"Heading\n{heading_deg:+.1f}°")

def update_map(gui, pos, p_hist, yaw_rad):
    xs = [DISPLAY_MIRROR_X * p[0] for p in p_hist]
    ys = [p[1] for p in p_hist]
    gui["traj_line"].set_data(xs, ys)
    x, y = DISPLAY_MIRROR_X * pos[0], pos[1]
    gui["body_dot"].set_data([x], [y])
    gui["map_coord_text"].set_text(f"X={x:+.2f} m  Y={y:+.2f} m")

    ax = gui["ax_map"]
    if len(xs) > 1:
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.01)
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

def update_pedestrian(gui, step_detector, is_idle):
    c = TL_RED_ON if is_idle else TL_GREEN_ON
    gui["idle_shoe_sole"].set_facecolor(c)
    gui["idle_shoe_heel"].set_facecolor(c)
    gui["active_shoe_sole"].set_facecolor(c)
    gui["active_shoe_heel"].set_facecolor(c)

def update_gyro_and_vel(gui, t_hist, bx_hist, by_hist, bz_hist, vel_hist):
    for key, y_hist, ax_key in [("gbx", bx_hist, "ax_gbx"),
                                 ("gby", by_hist, "ax_gby"),
                                 ("gbz", bz_hist, "ax_gbz")]:
        gui[f"ln_{key}"].set_data(t_hist, y_hist)
        gui[ax_key].relim()
        gui[ax_key].autoscale_view()
    gui["ln_vel"].set_data(t_hist, vel_hist)
    gui["ax_vel"].relim()
    gui["ax_vel"].autoscale_view()

def update_magnetic(gui, mag_intensity_ut):
    flash_on = (int(time.time() * 3) % 2 == 0)
    if mag_intensity_ut > MAG_THRESH_MID_UT:
        gui["mag_high_circle"].set_facecolor(
            TL_RED_ON if flash_on else TL_RED_DIM)
        gui["mag_minor_circle"].set_facecolor(TL_OFF)
        gui["mag_low_circle"].set_facecolor(TL_OFF)
    elif mag_intensity_ut > MAG_THRESH_LOW_UT:
        gui["mag_high_circle"].set_facecolor(TL_OFF)
        gui["mag_minor_circle"].set_facecolor(
            TL_YELLOW_ON if flash_on else TL_YELLOW_DIM)
        gui["mag_low_circle"].set_facecolor(TL_OFF)
    else:
        gui["mag_high_circle"].set_facecolor(TL_OFF)
        gui["mag_minor_circle"].set_facecolor(TL_OFF)
        gui["mag_low_circle"].set_facecolor(
            TL_GREEN_ON if flash_on else TL_GREEN_DIM)

def keep_windows_responsive():
    plt.pause(UI_PAUSE)

def save_gui_plot(gui):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(current_dir, "Trajectory Plots")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"GUI_screen_{timestamp_str}.png")
    
    gui["fig"].savefig(out_file, dpi=200, facecolor="#1e293b", bbox_inches='tight')
    print(f"Plot saved to {out_file}")
