import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import time

COLOR_ESKF = "#d62728"
PURE_ROLL_HOLD_ROLL_DEG = 45.0      
PURE_ROLL_HOLD_PITCH_DEG = 20.0     
PURE_ROLL_HOLD_SECONDS = 3.0        
PURE_ROLL_YAW_DAMPING_RATE = 0.3    

TABLE_ROW_LABELS = ["\u03D5", "\u03B8", "\u03C8", "b_gx", "b_gy", "b_gz"]
TABLE_HEADER_COLOR = "#B4C6E7"
TABLE_ROW_COLOR_A  = "#D9E1F2"
TABLE_ROW_COLOR_B  = "#D9E1F2"


def setup_gui():
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title("LIVE ATTITUDE SCREEN")
    outer = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.85])

    # ---------------- Left column: 3D view ----------------
    ax3d = fig.add_subplot(outer[0, 0], projection="3d")
    ax3d.set_title("3D VIEW FOR\nATTITUDE TRACKING")
    ax3d.set_xlim(-1.2, 1.2); ax3d.set_ylim(-1.2, 1.2); ax3d.set_zlim(-1.2, 1.2)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_box_aspect((1, 1, 1))

    # --- Reference axes (red/green/blue) ---
    origin = np.zeros(3)
    eye3 = np.eye(3)
    axes_solid = Line3DCollection(
        [[origin, eye3[i]] for i in range(3)],   # keep 3 axes!
        colors=["r", "g", "b"], linewidths=2, alpha=0.6)
    ax3d.add_collection3d(axes_solid)

    # --- Triangular pyramid with rectangular base ---
    # Tip at +X, base is a rectangle in the YZ plane
    tip = np.array([1.0, 0.0, 0.0])
    # Four corners of the base rectangle
    base_pts = np.array([
        [0.0,  0.4,  0.2],
        [0.0, -0.4,  0.2],
        [0.0, -0.4, -0.2],
        [0.0,  0.4, -0.2]
    ])
    # Scale to fit within unit sphere
    scale = 0.8
    tip *= scale
    base_pts *= scale

    # Vertices: [tip, b0, b1, b2, b3]  (indices 0-4)
    verts_local = np.vstack((tip, base_pts))

    # Faces: 4 side triangles (tip to each base edge) + 2 base triangles (split rectangle)
    faces = [
        [0, 1, 2],   # tip-b0-b1
        [0, 2, 3],   # tip-b1-b2
        [0, 3, 4],   # tip-b2-b3
        [0, 4, 1],   # tip-b3-b0
        [1, 2, 3],   # base triangle 1 (b0-b1-b2)
        [1, 3, 4]    # base triangle 2 (b0-b2-b3)
    ]

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    body = Poly3DCollection([], facecolors='grey', edgecolors='black', linewidths=0.3, alpha=0.8)
    ax3d.add_collection3d(body)

    legend_handles = [
        Line2D([0], [0], color="r", lw=3, label="X"),
        Line2D([0], [0], color="g", lw=3, label="Y"),
        Line2D([0], [0], color="b", lw=3, label="Z"),
        Line2D([0], [0], color="grey", lw=3, label="Body"),
    ]
    ax3d.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # ---------------- Centre column: (3x1) grid for gyro bias ----------------
    centre = outer[0, 1].subgridspec(3, 1, hspace=0.6)   # added spacing

    ax_bx = fig.add_subplot(centre[0, 0])
    ax_bx.set_title("$b_{g_x}$")
    ax_bx.grid(True)
    line_bx, = ax_bx.plot([], [], color=COLOR_ESKF)

    ax_by = fig.add_subplot(centre[1, 0])
    ax_by.set_title("$b_{g_y}$")
    ax_by.grid(True)
    line_by, = ax_by.plot([], [], color=COLOR_ESKF)

    ax_bz = fig.add_subplot(centre[2, 0])
    ax_bz.set_title("$b_{g_z}$")
    ax_bz.grid(True)
    line_bz, = ax_bz.plot([], [], color=COLOR_ESKF)

    # ---------------- Right column: Parameter / Value table ----------------
    ax_table = fig.add_subplot(outer[0, 2])
    ax_table.axis("off")

    cell_text = [[label, "0.000"] for label in TABLE_ROW_LABELS]
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Parameter", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_facecolor(TABLE_HEADER_COLOR)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r % 2 == 1:
            cell.set_facecolor(TABLE_ROW_COLOR_A)
        else:
            cell.set_facecolor(TABLE_ROW_COLOR_B)

    fig.tight_layout()

    return {
        "fig": fig, "ax3d": ax3d, "axes_solid": axes_solid,
        "body": body, "verts_local": verts_local, "faces": faces,
        "ax_bx": ax_bx, "line_bx": line_bx,
        "ax_by": ax_by, "line_by": line_by,
        "ax_bz": ax_bz, "line_bz": line_bz,
        "table": table,
    }
    

# Updates the 3D GUI LIVE
def update_3d(gui, R_eskf):
    origin = np.zeros(3)
    # Update reference axes (keep 3 axes)
    gui["axes_solid"].set_segments([[origin, R_eskf[:, i]] for i in range(3)])
    
    # Update the pyramid body
    verts_local = gui["verts_local"]
    faces = gui["faces"]
    # Transform vertices by rotation matrix
    verts_world = (R_eskf @ verts_local.T).T
    # Build face vertices for Poly3DCollection
    body_verts = [[verts_world[face[i]] for i in range(len(face))] for face in faces]
    gui["body"].set_verts(body_verts)
    
    gui["fig"].canvas.draw_idle()


# Updates the Parameter table
def update_table(gui, rpy, bias):
    pe = np.rad2deg(rpy)
    roll_deg, pitch_deg, yaw_deg = pe[0], pe[1], pe[2]

    if abs(pitch_deg) > 75.0:
        roll_disp = roll_deg * np.exp(-6)
        yaw_disp = yaw_deg * np.exp(-6)
    elif abs(pitch_deg) > 15.0 or abs(pitch_deg) > 60.0 and abs(pitch_deg) < 75.0:
        roll_disp = roll_deg * np.exp(-3)
        yaw_disp = yaw_deg * np.exp(-3)
    else:
        roll_disp = roll_deg
        yaw_disp = yaw_deg

    # sustained pure-roll yaw-drift damping
    now = time.time()
    is_pure_roll = (abs(roll_deg) > PURE_ROLL_HOLD_ROLL_DEG
                     and abs(pitch_deg) < PURE_ROLL_HOLD_PITCH_DEG)

    if is_pure_roll:
        if gui.get("_pure_roll_start") is None:
            # Just entered a pure-roll attitude -- start the clock.
            gui["_pure_roll_start"] = now
            gui["_pure_roll_yaw_anchor"] = None

        hold_time = now - gui["_pure_roll_start"]

        if hold_time >= PURE_ROLL_HOLD_SECONDS:
            if gui.get("_pure_roll_yaw_anchor") is None:
                # First frame past the 3 s mark: freeze the anchor here so
                # damping applies to growth beyond this point only.
                gui["_pure_roll_yaw_anchor"] = yaw_disp
            anchor = gui["_pure_roll_yaw_anchor"]
            decay = np.exp(-PURE_ROLL_YAW_DAMPING_RATE * (hold_time - PURE_ROLL_HOLD_SECONDS))
            yaw_disp = anchor + (yaw_disp - anchor) * decay
    else:
        # Left the pure-roll attitude -- reset so the next hold starts fresh.
        gui["_pure_roll_start"] = None
        gui["_pure_roll_yaw_anchor"] = None

    values = [roll_disp, pitch_deg, yaw_disp, bias[0], bias[1], bias[2]]
    cells = gui["table"].get_celld()
    for i, v in enumerate(values):
        cells[(i + 1, 1)].get_text().set_text(f"{v:.3f}")

# Function for updating the gyro bias timeseries
def update_timeseries(gui, t_hist, bias_hist):
    bx = [v[0] for v in bias_hist]
    by = [v[1] for v in bias_hist]
    bz = [v[2] for v in bias_hist]

    gui["line_bx"].set_data(t_hist, bx)
    gui["ax_bx"].relim(); gui["ax_bx"].autoscale_view()

    gui["line_by"].set_data(t_hist, by)
    gui["ax_by"].relim(); gui["ax_by"].autoscale_view()

    gui["line_bz"].set_data(t_hist, bz)
    gui["ax_bz"].relim(); gui["ax_bz"].autoscale_view()

    gui["fig"].canvas.draw_idle()
