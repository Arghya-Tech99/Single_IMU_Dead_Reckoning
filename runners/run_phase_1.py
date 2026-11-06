
import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
"""
IMU_calibration_Phase01.py

---------------------------------------------------------------------
PURPOSE: 

To verify whether or not the sensors are calibrated or not and to model the gyroscope noise.

---------------------------------------------------------------------
IMPORTANT: 

    uint32 packet_id
    uint32 accel_timestamp_ms;  float ax, ay, az;                       // JY901S accel  (g)
    uint32 gyro_timestamp_ms;   float gx, gy, gz;                       // JY901S gyro   (deg/s)
    uint32 jy_mag_timestamp_ms; float jmx, jmy, jmz;                    // JY901S mag  (IGNORED)  
    uint32 rm_mag_timestamp_ms; float rmx, rmy, rmz;                    // RM3100 mag    (nT, USED)

    PACKET_FORMAT : IIfffIfffIfffIfff
    PACKET SIZE   : 68 bytes

"""


from GUI.gui_phase1_calibration import plot_allan_deviation, plot_mag_calibration, plot_accel_calibration
from core.network import UDPReceiver
from core.config_loader import load_config
from core.calibration import CalibrationManager
from pathlib import Path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici
from scipy.optimize import least_squares
from scipy.linalg import sqrtm

# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

config = load_config("phase_1")
calib_manager = CalibrationManager()

def collect_samples(duration_s, prompt):
    """
    Drains `latest_data` (populated by the untouched udp_receiver() thread)
    for `duration_s` seconds, de-duplicating on packet_id exactly like the
    main loop in ahrs_dual_filter_live.py, and returns parallel lists of
    raw/calibrated accel, gyro, mag samples plus per-sample dt.
    """
    print("\n" + "=" * 70)
    print(prompt)
    print(f"Collecting for {duration_s:.0f} seconds...")
    print("=" * 70)

    t_start = time.time()
    prev_packet_id = None
    prev_imu_ts = None

    acc_raw_list, gyro_raw_list, mag_raw_list = [], [], []
    acc_cal_list, gyro_cal_list, mag_cal_list = [], [], []
    dt_list = []

    while time.time() - t_start < duration_s:
        data = network.get_latest_data()

        if data is None:
            plt.pause(config.get("ui_pause", 0.001))
            time.sleep(0.001)
            continue

        (packet_id,
         gyro_ts, ax_, ay_, az_,
         gx, gy, gz,
         rm_mag_ts, rmx, rmy, rmz) = data

        if packet_id == prev_packet_id:
            plt.pause(config.get("ui_pause", 0.001))
            time.sleep(0.001)
            continue
        prev_packet_id = packet_id

        imu_ts = gyro_ts
        if prev_imu_ts is None:
            prev_imu_ts = imu_ts
            continue

        dt = (imu_ts - prev_imu_ts) / 1000.0
        prev_imu_ts = imu_ts
        if dt <= 0 or dt > config.get("max_dt", 0.2):
            continue

        acc_raw = np.array([ax_, ay_, az_])
        gyro_raw = np.array([gx, gy, gz])       # deg/s, straight off the wire
        mag_raw = np.array([rmx, rmy, rmz])     # RM3100 ONLY -- nT

        acc_cal, gyro_cal, mag_cal = calib_manager.preprocess_sensor_data(acc_raw, gyro_raw, mag_raw, calibrate=config.get("calibrate", True))

        acc_raw_list.append(acc_raw); gyro_raw_list.append(gyro_raw); mag_raw_list.append(mag_raw)
        acc_cal_list.append(acc_cal); gyro_cal_list.append(gyro_cal); mag_cal_list.append(mag_cal)
        dt_list.append(dt)

        remaining = duration_s - (time.time() - t_start)
        print(f"  samples: {len(acc_raw_list):5d}   time left: {remaining:5.1f}s", end="\r")

        plt.pause(config.get("ui_pause", 0.001))

    print(f"\nDone. Collected {len(acc_raw_list)} samples.")

    if len(acc_raw_list) == 0:
        print(
            "\n*** WARNING: 0 samples were collected during this window. ***\n"
            "This usually means no UDP packets ever arrived, most likely because:\n"
            f"  - nothing is sending to {network.ip}:{network.port} (check the sender's target IP/port)\n"
            "  - a firewall is blocking inbound UDP on this port\n"
            "  - the sender's packet layout doesn't match PACKET_FORMAT "
            f"(expected {network.packet_size} bytes per packet)\n"
            "Downstream plotting will be skipped for this phase."
        )

    return {
        "acc_raw": np.array(acc_raw_list), "gyro_raw": np.array(gyro_raw_list), "mag_raw": np.array(mag_raw_list),
        "acc_cal": np.array(acc_cal_list), "gyro_cal": np.array(gyro_cal_list), "mag_cal": np.array(mag_cal_list),
        "dt": np.array(dt_list),
    }


# ============================================================
# AZIMUTH-COVERAGE HELPERS 
# ============================================================

def azimuth_bin_index(x, y, n_bins=config.get("accel_azimuth_bins", 12)):
    """Bin atan2(y, x) into n_bins equal wedges covering the full circle."""
    angle_deg = np.degrees(np.arctan2(y, x)) % 360.0
    bin_width = 360.0 / n_bins
    idx = int(angle_deg // bin_width)
    return min(idx, n_bins - 1)  # guard against the angle_deg == 360.0 edge case


def format_coverage_bar(bin_counts, min_count=config.get("accel_min_samples_per_bin", 15)):
    """'Azimuth coverage: [X X . . X X . . X X . .]' -- 'X' once a wedge has
    reached min_count samples, '.' while it's still empty/short."""
    cells = ["X" if c >= min_count else "." for c in bin_counts]
    return "[" + " ".join(cells) + "]"


def empty_sample_dict():
    """A zero-length version of the dict collect_samples()/friends return,
    so guided holds, the free sweep, and empty edge cases can all be
    concatenated the same way."""
    keys = ["acc_raw", "gyro_raw", "mag_raw", "acc_cal", "gyro_cal", "mag_cal", "dt"]
    return {k: np.zeros((0, 3)) if k != "dt" else np.zeros((0,)) for k in keys}


def concat_sample_dicts(*sample_dicts):
    """Concatenate several collect_samples()-shaped dicts (guided holds +
    free sweep, in any combination) into one, in the order given."""
    keys = ["acc_raw", "gyro_raw", "mag_raw", "acc_cal", "gyro_cal", "mag_cal", "dt"]
    out = {}
    for k in keys:
        arrs = [d[k] for d in sample_dicts if len(d[k]) > 0]
        out[k] = np.concatenate(arrs, axis=0) if arrs else (
            np.zeros((0, 3)) if k != "dt" else np.zeros((0,))
        )
    return out


def bin_counts_from_samples(samples, n_bins=config.get("accel_azimuth_bins", 12)):
    """Recompute the azimuth-bin histogram (on CALIBRATED accel X/Y) from a
    collect_samples()-shaped dict, e.g. to seed the free-sweep's coverage
    gate with whatever the guided holds already covered."""
    counts = np.zeros(n_bins, dtype=int)
    acc_cal = samples["acc_cal"]
    for x, y in acc_cal[:, :2]:
        counts[azimuth_bin_index(x, y, n_bins)] += 1
    return counts


# ============================================================
# PHASE A ACCELEROMETER
# ============================================================

def guided_diagonal_holds(poses, hold_duration=config.get("accel_hold_duration", 3.0),
                           settle_time=config.get("accel_hold_settle", 1.0)):
    
    print("\n" + "=" * 70)
    print("GUIDED DIAGONAL HOLDS (deterministic azimuth-coverage backstop)")
    print("=" * 70)
    print(f"Hold each pose steady for {hold_duration:.0f}s "
          f"(first {settle_time:.0f}s discarded as settle time).\n")

    pose_samples = []
    for label, desc in poses:
        input(f"Pose: {label}\n{desc}\nPress Enter when ready to start collecting...")

        acc_raw_list, gyro_raw_list, mag_raw_list = [], [], []
        acc_cal_list, gyro_cal_list, mag_cal_list = [], [], []
        dt_list = []

        t_start = time.time()
        prev_packet_id = None
        prev_imu_ts = None
        while time.time() - t_start < hold_duration:
            data = network.get_latest_data()
            if data is None:
                plt.pause(config.get("ui_pause", 0.001))
                time.sleep(0.001)
                continue

            (packet_id,
             gyro_ts, ax_, ay_, az_,
             gx, gy, gz,
             rm_mag_ts, rmx, rmy, rmz) = data

            if packet_id == prev_packet_id:
                plt.pause(config.get("ui_pause", 0.001))
                time.sleep(0.001)
                continue
            prev_packet_id = packet_id

            if time.time() - t_start < settle_time:
                continue  # let hand-wobble die out before trusting samples

            imu_ts = gyro_ts
            if prev_imu_ts is None:
                prev_imu_ts = imu_ts
                continue
            dt = (imu_ts - prev_imu_ts) / 1000.0
            prev_imu_ts = imu_ts
            if dt <= 0 or dt > config.get("max_dt", 0.2):
                continue

            acc_raw = np.array([ax_, ay_, az_])
            gyro_raw = np.array([gx, gy, gz])
            mag_raw = np.array([rmx, rmy, rmz])
            acc_cal, gyro_cal, mag_cal = calib_manager.preprocess_sensor_data(acc_raw, gyro_raw, mag_raw, calibrate=config.get("calibrate", True))

            acc_raw_list.append(acc_raw); gyro_raw_list.append(gyro_raw); mag_raw_list.append(mag_raw)
            acc_cal_list.append(acc_cal); gyro_cal_list.append(gyro_cal); mag_cal_list.append(mag_cal)
            dt_list.append(dt)

            plt.pause(config.get("ui_pause", 0.001))

        print(f"  {label}: collected {len(acc_raw_list)} samples.")
        pose_samples.append({
            "acc_raw": np.array(acc_raw_list) if acc_raw_list else np.zeros((0, 3)),
            "gyro_raw": np.array(gyro_raw_list) if gyro_raw_list else np.zeros((0, 3)),
            "mag_raw": np.array(mag_raw_list) if mag_raw_list else np.zeros((0, 3)),
            "acc_cal": np.array(acc_cal_list) if acc_cal_list else np.zeros((0, 3)),
            "gyro_cal": np.array(gyro_cal_list) if gyro_cal_list else np.zeros((0, 3)),
            "mag_cal": np.array(mag_cal_list) if mag_cal_list else np.zeros((0, 3)),
            "dt": np.array(dt_list) if dt_list else np.zeros((0,)),
        })

    return concat_sample_dicts(*pose_samples)


ACCEL_DIAGONAL_POSES_PRIMARY = [
    ("+X+Y", "Tilt the device so it leans toward BOTH +X and +Y at once\n"
             "(a corner-down diagonal tilt, azimuth ~45 deg). Hold steady."),
    ("+X-Y", "Tilt the device so it leans toward +X and -Y at once\n"
             "(diagonal tilt, azimuth ~315 deg / -45 deg). Hold steady."),
    ("-X+Y", "Tilt the device so it leans toward -X and +Y at once\n"
             "(diagonal tilt, azimuth ~135 deg). Hold steady."),
    ("-X-Y", "Tilt the device so it leans toward BOTH -X and -Y at once\n"
             "(diagonal tilt, azimuth ~225 deg). Hold steady."),
]

ACCEL_DIAGONAL_POSES_YAWED = [
    ("+X (yawed diagonal)", "Tilt the device mostly toward +X, but yawed about\n"
                            "45 deg from the previous +X+Y pose (azimuth ~0 deg,\n"
                            "still tilted, not flat-on-axis). Hold steady."),
    ("+Y (yawed diagonal)", "Tilt the device mostly toward +Y, yawed ~45 deg from\n"
                            "the previous +X+Y pose (azimuth ~90 deg, still tilted).\n"
                            "Hold steady."),
    ("-X (yawed diagonal)", "Tilt the device mostly toward -X, yawed ~45 deg from\n"
                            "the previous -X-Y pose (azimuth ~180 deg, still tilted).\n"
                            "Hold steady."),
    ("-Y (yawed diagonal)", "Tilt the device mostly toward -Y, yawed ~45 deg from\n"
                            "the previous -X-Y pose (azimuth ~270 deg, still tilted).\n"
                            "Hold steady."),
]


# ============================================================
# PHASE A-Accel, PART 2: coverage-gated free-rotation sweep
# ============================================================

def collect_samples_coverage_gated(prompt, n_bins=config.get("accel_azimuth_bins", 12),
                                    min_per_bin=config.get("accel_min_samples_per_bin", 15),
                                    max_duration_s=config.get("accel_sweep_max_time", 120.0),
                                    seed_samples=None):
    """
    Replaces a flat "collect for N seconds and hope" sweep with a
    COVERAGE-GATED loop: each incoming sample's azimuth
    atan2(Y, X) (of the CALIBRATED accel reading) is binned into
    `n_bins` wedges (default 12 x 30 deg), live coverage is printed, and
    collection does not stop until every wedge has at least
    `min_per_bin` samples -- so finishing with a cross-shaped gap is
    structurally impossible, rather than merely unlikely.

    `seed_samples`, if given (e.g. the guided diagonal holds run right
    before this), is credited toward the same bin counts before free
    rotation starts, so a wedge the guided holds already nailed doesn't
    need to be independently re-covered by free motion.

    `max_duration_s` is a hard safety cap only -- it exists so a single
    physically-unreachable wedge (bad mount, cable snag, etc.) can't hang
    the script forever. If it's hit, the still-empty wedges are reported
    and collection stops anyway; the post-collection diagnostic
    (compute_azimuth_gap) will catch and flag whatever gap remains.
    """
    print("\n" + "=" * 70)
    print(prompt)
    print(f"Collecting until every azimuth wedge has >= {min_per_bin} samples "
          f"(safety cap: {max_duration_s:.0f}s)...")
    print("=" * 70)

    bin_counts = np.zeros(n_bins, dtype=int)
    if seed_samples is not None:
        bin_counts += bin_counts_from_samples(seed_samples, n_bins)

    t_start = time.time()
    prev_packet_id = None
    prev_imu_ts = None

    acc_raw_list, gyro_raw_list, mag_raw_list = [], [], []
    acc_cal_list, gyro_cal_list, mag_cal_list = [], [], []
    dt_list = []

    while True:
        elapsed = time.time() - t_start
        if bin_counts.min() >= min_per_bin:
            break
        if elapsed >= max_duration_s:
            print(
                f"\n*** WARNING: hit the {max_duration_s:.0f}s safety cap before every "
                "azimuth wedge reached coverage. ***\n"
                f"Still-short wedges: {format_coverage_bar(bin_counts, min_per_bin)}\n"
                "Continuing anyway -- the post-collection diagnostic below will flag "
                "the remaining gap explicitly."
            )
            break
        data = network.get_latest_data()
        if data is None:
            plt.pause(config.get("ui_pause", 0.001))
            time.sleep(0.001)
            continue

        (packet_id,
         gyro_ts, ax_, ay_, az_,
         gx, gy, gz,
         rm_mag_ts, rmx, rmy, rmz) = data

        if packet_id == prev_packet_id:
            plt.pause(config.get("ui_pause", 0.001))
            time.sleep(0.001)
            continue
        prev_packet_id = packet_id

        imu_ts = gyro_ts
        if prev_imu_ts is None:
            prev_imu_ts = imu_ts
            continue
        dt = (imu_ts - prev_imu_ts) / 1000.0
        prev_imu_ts = imu_ts
        if dt <= 0 or dt > config.get("max_dt", 0.2):
            continue

        acc_raw = np.array([ax_, ay_, az_])
        gyro_raw = np.array([gx, gy, gz])
        mag_raw = np.array([rmx, rmy, rmz])
        acc_cal, gyro_cal, mag_cal = calib_manager.preprocess_sensor_data(acc_raw, gyro_raw, mag_raw, calibrate=config.get("calibrate", True))

        acc_raw_list.append(acc_raw); gyro_raw_list.append(gyro_raw); mag_raw_list.append(mag_raw)
        acc_cal_list.append(acc_cal); gyro_cal_list.append(gyro_cal); mag_cal_list.append(mag_cal)
        dt_list.append(dt)

        bin_counts[azimuth_bin_index(acc_cal[0], acc_cal[1], n_bins)] += 1

        print(f"  samples: {len(acc_raw_list):5d}   elapsed: {elapsed:5.1f}s   "
              f"Azimuth coverage: {format_coverage_bar(bin_counts, min_per_bin)}", end="\r")

        plt.pause(config.get("ui_pause", 0.001))

    print(f"\nDone. Collected {len(acc_raw_list)} samples. "
          f"Final azimuth coverage: {format_coverage_bar(bin_counts, min_per_bin)}")

    if len(acc_raw_list) == 0:
        print(
            "\n*** WARNING: 0 samples were collected during this window. ***\n"
            "This usually means no UDP packets ever arrived -- see the notes in "
            "collect_samples() for likely causes.\nDownstream plotting will be skipped "
            "for this phase."
        )

    return {
        "acc_raw": np.array(acc_raw_list) if acc_raw_list else np.zeros((0, 3)),
        "gyro_raw": np.array(gyro_raw_list) if gyro_raw_list else np.zeros((0, 3)),
        "mag_raw": np.array(mag_raw_list) if mag_raw_list else np.zeros((0, 3)),
        "acc_cal": np.array(acc_cal_list) if acc_cal_list else np.zeros((0, 3)),
        "gyro_cal": np.array(gyro_cal_list) if gyro_cal_list else np.zeros((0, 3)),
        "mag_cal": np.array(mag_cal_list) if mag_cal_list else np.zeros((0, 3)),
        "dt": np.array(dt_list) if dt_list else np.zeros((0,)),
    }


def compute_azimuth_gap(xy_points):
    xy_points = np.asarray(xy_points)
    if len(xy_points) == 0:
        return None

    angles = np.degrees(np.arctan2(xy_points[:, 1], xy_points[:, 0])) % 360.0
    angles_sorted = np.sort(angles)

    diffs = np.diff(angles_sorted)
    wrap_gap = (angles_sorted[0] + 360.0) - angles_sorted[-1]
    all_gaps = np.append(diffs, wrap_gap)

    i_max = int(np.argmax(all_gaps))
    gap_deg = float(all_gaps[i_max])
    gap_start = float(angles_sorted[i_max])
    gap_end = float(angles_sorted[(i_max + 1) % len(angles_sorted)])

    return {
        "gap_deg": gap_deg,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "angles": angles_sorted,
    }


def report_azimuth_coverage(samples, warn_deg=config.get("accel_gap_warn_deg", 45.0), phase_label="Phase A-Accel"):
    acc_cal = samples["acc_cal"]
    if len(acc_cal) == 0:
        print(f"\n{phase_label} coverage diagnostic: no samples to check.")
        return None

    gap_info = compute_azimuth_gap(acc_cal[:, :2])
    gap_deg = gap_info["gap_deg"]

    print(f"\n--- {phase_label} azimuthal coverage diagnostic ---")
    print(f"  Largest gap: {gap_deg:.0f} deg, between {gap_info['gap_start']:.1f} deg "
          f"and {gap_info['gap_end']:.1f} deg.")

    if gap_deg > warn_deg:
        print(
            f"*** WARNING: XY azimuthal coverage gap: {gap_deg:.0f} deg span uncovered "
            f"between {gap_info['gap_start']:.0f} deg and {gap_info['gap_end']:.0f} deg "
            f"-- recommend re-running {phase_label}. ***"
        )
    else:
        print(f"  Coverage OK (largest gap {gap_deg:.0f} deg <= {warn_deg:.0f} deg threshold).")

    return gap_info


# ============================================================
# PHASE A MAGNETOMETER
# ============================================================

def spherical_bin_index(x, y, z, n_az=config.get("mag_azimuth_bins", 8), n_el=config.get("mag_elevation_bins", 5)):
    """Bin a 3D vector into (azimuth wedge, elevation band). Azimuth is
    atan2(y, x) same as the accelerometer case; elevation is
    asin(z / ||v||), i.e. angle up from the XY plane, range [-90, 90]."""
    azimuth_deg = np.degrees(np.arctan2(y, x)) % 360.0
    az_width = 360.0 / n_az
    az_idx = int(azimuth_deg // az_width)
    az_idx = min(az_idx, n_az - 1)

    norm = np.sqrt(x * x + y * y + z * z)
    if norm < 1e-12:
        el_idx = n_el // 2  # degenerate near-zero vector; park it in the middle band
    else:
        elevation_deg = np.degrees(np.arcsin(np.clip(z / norm, -1.0, 1.0)))
        el_width = 180.0 / n_el
        el_idx = int((elevation_deg + 90.0) // el_width)
        el_idx = min(max(el_idx, 0), n_el - 1)

    return az_idx, el_idx


def az_bin_range_deg(idx, n_az=config.get("mag_azimuth_bins", 8)):
    width = 360.0 / n_az
    return idx * width, (idx + 1) * width


def el_bin_range_deg(idx, n_el=config.get("mag_elevation_bins", 5)):
    width = 180.0 / n_el
    return -90.0 + idx * width, -90.0 + (idx + 1) * width


def format_mag_coverage_grid(bin_counts_2d, min_count=config.get("mag_min_samples_per_bin", 10),
                              n_az=config.get("mag_azimuth_bins", 8), n_el=config.get("mag_elevation_bins", 5)):
    lines = []
    for el_idx in reversed(range(n_el)):
        lo, hi = el_bin_range_deg(el_idx, n_el)
        row = " ".join(
            "X" if bin_counts_2d[az_idx, el_idx] >= min_count else "."
            for az_idx in range(n_az)
        )
        lines.append(f"  el[{lo:+5.0f},{hi:+5.0f}): [{row}]")
    return "\n".join(lines)


def list_missing_wedges(bin_counts_2d, min_count=config.get("mag_min_samples_per_bin", 10),
                         n_az=config.get("mag_azimuth_bins", 8), n_el=config.get("mag_elevation_bins", 5)):
    missing = []
    for az_idx in range(n_az):
        for el_idx in range(n_el):
            if bin_counts_2d[az_idx, el_idx] < min_count:
                az_lo, az_hi = az_bin_range_deg(az_idx, n_az)
                el_lo, el_hi = el_bin_range_deg(el_idx, n_el)
                missing.append(f"az[{az_lo:.0f},{az_hi:.0f}) el[{el_lo:.0f},{el_hi:.0f})")
    return missing


def mag_bin_counts_from_samples(samples, n_az=config.get("mag_azimuth_bins", 8), n_el=config.get("mag_elevation_bins", 5)):
    counts = np.zeros((n_az, n_el), dtype=int)
    mag_cal = samples["mag_cal"]
    for x, y, z in mag_cal:
        az_idx, el_idx = spherical_bin_index(x, y, z, n_az, n_el)
        counts[az_idx, el_idx] += 1
    return counts


def collect_samples_coverage_gated_mag(prompt, n_az=config.get("mag_azimuth_bins", 8), n_el=config.get("mag_elevation_bins", 5),
                                        min_per_bin=config.get("mag_min_samples_per_bin", 10),
                                        max_duration_s=config.get("mag_sweep_max_time", 180.0),
                                        seed_samples=None):
    print("\n" + "=" * 70)
    print(prompt)
    print(f"Collecting until every (azimuth, elevation) cell has >= {min_per_bin} samples "
          f"(safety cap: {max_duration_s:.0f}s)...")
    print("=" * 70)

    bin_counts = np.zeros((n_az, n_el), dtype=int)
    if seed_samples is not None:
        bin_counts += mag_bin_counts_from_samples(seed_samples, n_az, n_el)

    total_cells = n_az * n_el
    prev_missing_count = None

    t_start = time.time()
    prev_packet_id = None
    prev_imu_ts = None

    acc_raw_list, gyro_raw_list, mag_raw_list = [], [], []
    acc_cal_list, gyro_cal_list, mag_cal_list = [], [], []
    dt_list = []

    while True:
        elapsed = time.time() - t_start
        covered = int(np.sum(bin_counts >= min_per_bin))
        missing_count = total_cells - covered

        if prev_missing_count is None or missing_count != prev_missing_count:
            missing = list_missing_wedges(bin_counts, min_per_bin, n_az, n_el)
            print(f"\nSphere coverage: {covered}/{total_cells} cells covered.")
            print(format_mag_coverage_grid(bin_counts, min_per_bin, n_az, n_el))
            if missing:
                print("Still missing: " + "; ".join(missing))
            prev_missing_count = missing_count

        if missing_count == 0:
            break
        if elapsed >= max_duration_s:
            print(
                f"\n*** WARNING: hit the {max_duration_s:.0f}s safety cap before every "
                "sphere cell reached coverage. ***\n"
                f"Still missing {missing_count}/{total_cells} cells -- continuing anyway; "
                "the resulting ellipsoid fit may be under-constrained in those directions."
            )
            break
        data = network.get_latest_data()
        if data is None:
            plt.pause(config.get("ui_pause", 0.001))
            time.sleep(0.001)
            continue

        (packet_id,
         gyro_ts, ax_, ay_, az_,
         gx, gy, gz,
         rm_mag_ts, rmx, rmy, rmz) = data

        if packet_id == prev_packet_id:
            plt.pause(config.get("ui_pause", 0.001))
            time.sleep(0.001)
            continue
        prev_packet_id = packet_id

        imu_ts = gyro_ts
        if prev_imu_ts is None:
            prev_imu_ts = imu_ts
            continue
        dt = (imu_ts - prev_imu_ts) / 1000.0
        prev_imu_ts = imu_ts
        if dt <= 0 or dt > config.get("max_dt", 0.2):
            continue

        acc_raw = np.array([ax_, ay_, az_])
        gyro_raw = np.array([gx, gy, gz])
        mag_raw = np.array([rmx, rmy, rmz])
        acc_cal, gyro_cal, mag_cal = calib_manager.preprocess_sensor_data(acc_raw, gyro_raw, mag_raw, calibrate=config.get("calibrate", True))

        acc_raw_list.append(acc_raw); gyro_raw_list.append(gyro_raw); mag_raw_list.append(mag_raw)
        acc_cal_list.append(acc_cal); gyro_cal_list.append(gyro_cal); mag_cal_list.append(mag_cal)
        dt_list.append(dt)

        az_idx, el_idx = spherical_bin_index(mag_cal[0], mag_cal[1], mag_cal[2], n_az, n_el)
        bin_counts[az_idx, el_idx] += 1

        print(f"  samples: {len(mag_raw_list):5d}   elapsed: {elapsed:5.1f}s   "
              f"covered: {covered}/{total_cells} cells", end="\r")

        plt.pause(config.get("ui_pause", 0.001))

    print(f"\nDone. Collected {len(mag_raw_list)} samples. "
          f"Final sphere coverage: {int(np.sum(bin_counts >= min_per_bin))}/{total_cells} cells.")

    if len(mag_raw_list) == 0:
        print(
            "\n*** WARNING: 0 samples were collected during this window. ***\n"
            "This usually means no UDP packets ever arrived -- see the notes in "
            "collect_samples() for likely causes.\nDownstream plotting will be skipped "
            "for this phase."
        )

    return {
        "acc_raw": np.array(acc_raw_list) if acc_raw_list else np.zeros((0, 3)),
        "gyro_raw": np.array(gyro_raw_list) if gyro_raw_list else np.zeros((0, 3)),
        "mag_raw": np.array(mag_raw_list) if mag_raw_list else np.zeros((0, 3)),
        "acc_cal": np.array(acc_cal_list) if acc_cal_list else np.zeros((0, 3)),
        "gyro_cal": np.array(gyro_cal_list) if gyro_cal_list else np.zeros((0, 3)),
        "mag_cal": np.array(mag_cal_list) if mag_cal_list else np.zeros((0, 3)),
        "dt": np.array(dt_list) if dt_list else np.zeros((0,)),
    }


# Guided poses for magnetometer sphere coverage. Physically identical
# motion vocabulary to the accelerometer's guided holds (any fixed
# orientation gives a fixed reading against a fixed external vector,
# whether that vector is gravity or the local magnetic field) -- run
# BEFORE the free-rotation sweep to seed the pole/face directions that
# free rotation most often shorts.
MAG_POSES_PRIMARY = [
    ("Z-Up (mag)", "Place the device flat, Z axis pointing UP, and hold still.\n"
                   "(Same physical pose as the accelerometer Z-Up position.)"),
    ("Z-Down (mag)", "Place the device upside down, Z axis pointing DOWN, and hold still."),
    ("X-Up (mag)", "Stand the device on its edge, X axis pointing UP, and hold still."),
    ("X-Down (mag)", "Stand the device on its edge, X axis pointing DOWN, and hold still."),
    ("Y-Up (mag)", "Stand the device sideways, Y axis pointing UP, and hold still."),
    ("Y-Down (mag)", "Stand the device sideways, Y axis pointing DOWN, and hold still."),
]

# 8 cube-corner ("diagonal") poses -- fills the octants BETWEEN the six
# face directions above, run AFTER the free-rotation sweep so the guided
# holds bracket it on both sides, same pattern as the accelerometer phase.
MAG_POSES_DIAGONAL = [
    ("+X+Y+Z (mag)", "Tilt the device toward +X, +Y, AND +Z all at once (a corner\n"
                     "tilt, elevation ~+34 deg). Hold steady."),
    ("+X+Y-Z (mag)", "Tilt the device toward +X, +Y, and -Z all at once (elevation\n"
                     "~-34 deg). Hold steady."),
    ("+X-Y+Z (mag)", "Tilt the device toward +X, -Y, and +Z all at once. Hold steady."),
    ("+X-Y-Z (mag)", "Tilt the device toward +X, -Y, and -Z all at once. Hold steady."),
    ("-X+Y+Z (mag)", "Tilt the device toward -X, +Y, and +Z all at once. Hold steady."),
    ("-X+Y-Z (mag)", "Tilt the device toward -X, +Y, and -Z all at once. Hold steady."),
    ("-X-Y+Z (mag)", "Tilt the device toward -X, -Y, and +Z all at once. Hold steady."),
    ("-X-Y-Z (mag)", "Tilt the device toward -X, -Y, and -Z all at once. Hold steady."),
]


# ============================================================
# PHASE A (MAGNETOMETER): DRIFT / INTERFERENCE CHECK
# ============================================================

def check_mag_drift(samples, warn_frac=config.get("mag_drift_warn_frac", 0.03)):
    mag_raw = samples["mag_raw"]
    n = len(mag_raw)
    if n < 20:
        return {"drifted": False, "reason": "not enough samples to check", "n": n}

    mid = n // 2
    first_half, second_half = mag_raw[:mid], mag_raw[mid:]
    centroid1 = np.mean(first_half, axis=0)
    centroid2 = np.mean(second_half, axis=0)

    field_magnitude = float(np.mean(np.linalg.norm(mag_raw, axis=1)))
    centroid_diff = float(np.linalg.norm(centroid2 - centroid1))
    frac = centroid_diff / field_magnitude if field_magnitude > 1e-9 else float("inf")

    drifted = frac > warn_frac

    print(f"\n--- Magnetometer drift/interference check "
          f"(first-half vs. second-half centroid) ---")
    print(f"  First-half centroid:  {centroid1}")
    print(f"  Second-half centroid: {centroid2}")
    print(f"  Centroid shift: {centroid_diff:.2f} (raw units) "
          f"= {frac * 100:.1f}% of mean field magnitude ({field_magnitude:.2f}).")

    if drifted:
        print(
            f"*** WARNING: magnetometer field centroid shifted by {frac * 100:.1f}% "
            f"(> {warn_frac * 100:.0f}% threshold) between the first and second half "
            "of the sweep. ***\n"
            "This usually means something magnetic moved relative to the sensor during\n"
            "collection (a laptop, phone, monitor, or a desk/chair with embedded metal).\n"
            "Recommend re-running this sweep in a magnetically quiet spot, with anything\n"
            "like that either removed or held perfectly still relative to the sensor for\n"
            "the whole window, before trusting a re-derived MAG_BIAS/MAG_AINV."
        )
    else:
        print(f"  Drift OK (centroid shift {frac * 100:.1f}% <= {warn_frac * 100:.0f}% threshold).")

    return {
        "drifted": drifted,
        "centroid1": centroid1,
        "centroid2": centroid2,
        "centroid_diff": centroid_diff,
        "frac": frac,
        "field_magnitude": field_magnitude,
    }


# ============================================================
# PHASE A (MAGNETOMETER): ELLIPSOID FITTING
# ============================================================

def fit_mag_ellipsoid_calibration(samples):
    pts = np.asarray(samples["mag_raw"], dtype=float)
    if len(pts) < 20:
        raise ValueError(
            f"Not enough magnetometer samples ({len(pts)}) to fit an ellipsoid; "
            "need at least 20 (coverage gating should normally give far more)."
        )

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    D_design = np.column_stack([
        x ** 2, y ** 2, z ** 2,
        2 * x * y, 2 * x * z, 2 * y * z,
        2 * x, 2 * y, 2 * z,
    ])
    coeffs, *_ = np.linalg.lstsq(D_design, np.ones_like(x), rcond=None)
    A, B, C, Dc, E, F, G, H, I = coeffs
    J = -1.0

    M = np.array([
        [A, Dc, E],
        [Dc, B, F],
        [E, F, C],
    ])
    v = np.array([G, H, I])

    bias = -np.linalg.solve(M, v)
    k = -(bias @ M @ bias + 2.0 * v @ bias + J)
    if not np.isfinite(k) or k <= 0:
        raise ValueError(
            "Ellipsoid fit degenerate (k <= 0) -- this usually means the coverage "
            "and/or drift checks should have blocked this fit; re-run the sweep."
        )

    target_radius = float(np.median(np.linalg.norm(pts, axis=1)))
    T = np.real(sqrtm(M / k))  # numerical noise can leave a negligible imaginary part
    Ainv = target_radius * T

    residual_norms = np.linalg.norm(Ainv @ (pts - bias).T, axis=0)
    rms_residual = float(np.sqrt(np.mean((residual_norms - target_radius) ** 2)))

    print("\nComputed magnetometer calibration (algebraic ellipsoid fit over "
          f"{len(pts)} coverage-gated samples):")
    print(f"  Target sphere radius (median raw field magnitude): {target_radius:.2f}")
    print(f"  Residual RMS (calibrated norm vs. target radius): {rms_residual:.4f}")
    print(f"  bias: {bias}")
    print(f"  Ainv:\n{Ainv}")

    return bias, Ainv


def recompute_mag_cal(samples):
    mag_raw = samples["mag_raw"]
    if len(mag_raw) == 0:
        samples["mag_cal"] = np.zeros((0, 3))
        return samples
    samples["mag_cal"] = np.array([calib_manager.R_mag @ calib_manager.apply_mag_calibration(m, calibrate=config.get("calibrate", True)) for m in mag_raw])
    return samples


# ============================================================
# ACCELEROMETER 6-POSITION CALIBRATION WITH ROBUST REFITTING
# ============================================================

def accelerometer_calibration_6pos():
    """
    Returns: bias (3-vector), Ainv (3x3 matrix)
    """
    # Fixed order: Z-up, Z-down, X-up, X-down, Y-up, Y-down
    orientations = [
        ("Z-Up", "Place device flat on table, Z axis pointing UP (top side up).",
         np.array([0.0, 0.0, 1.0])),
        ("Z-Down", "Place device upside down, Z axis pointing DOWN (top side down).",
         np.array([0.0, 0.0, -1.0])),
        ("X-Up", "Stand device on its edge, X axis pointing UP.",
         np.array([1.0, 0.0, 0.0])),
        ("X-Down", "Stand device on its edge, X axis pointing DOWN.",
         np.array([-1.0, 0.0, 0.0])),
        ("Y-Up", "Stand device sideways, Y axis pointing UP.",
         np.array([0.0, 1.0, 0.0])),
        ("Y-Down", "Stand device sideways, Y axis pointing DOWN.",
         np.array([0.0, -1.0, 0.0])),
    ]
    duration = 5.0       # seconds per position (up from 4 s -- more samples to filter/median over)
    settle_time = 1.0    # seconds discarded at the start of each hold (let hand-wobble die out)
    mad_k = 5.0           # MAD outlier threshold (robust-sigma multiples); generous but real

    position_medians = []    # robust per-position representative vector, in fixed order
    retained_raw_all = []    # every retained individual sample, pooled across all positions
    retained_expected_all = []  # matching expected direction, repeated per retained sample

    print("\n" + "=" * 70)
    print("ACCELEROMETER 6-POSITION CALIBRATION (robust full 3x3 + bias refit)")
    print("=" * 70)
    print(f"Hold the sensor still in each position for {duration:.0f} seconds")
    print(f"(first {settle_time:.0f}s is discarded as settle time).")
    print("Samples are MAD-filtered per position, then Ainv/bias are refined\n"
          "with a robust nonlinear least-squares fit over every retained sample.\n")

    for label, desc, expected_dir in orientations:
        input(f"Position: {label}\n{desc}\nPress Enter when ready to start collecting...")

        acc_raw_list = []
        t_start = time.time()
        prev_packet_id = None
        while time.time() - t_start < duration:
            data = network.get_latest_data()
            if data is None:
                time.sleep(0.001)
                continue
            (packet_id,
             gyro_ts, ax_, ay_, az_,
             gx, gy, gz,
             rm_mag_ts, rmx, rmy, rmz) = data
            if packet_id == prev_packet_id:
                time.sleep(0.001)
                continue
            prev_packet_id = packet_id

            if time.time() - t_start < settle_time:
                continue  # settle window

            acc_raw = np.array([ax_, ay_, az_])   # raw in g's
            acc_raw_list.append(acc_raw)

        acc_raw_arr = np.array(acc_raw_list) if acc_raw_list else np.zeros((0, 3))

        # --- MAD-based outlier rejection on sample norm ---
        if len(acc_raw_arr) >= 5:
            norms = np.linalg.norm(acc_raw_arr, axis=1)
            med_norm = np.median(norms)
            mad = np.median(np.abs(norms - med_norm))
            robust_sigma = 1.4826 * mad  # MAD -> std-equivalent for Gaussian-ish noise
            if robust_sigma > 1e-9:
                keep = np.abs(norms - med_norm) <= mad_k * robust_sigma
            else:
                keep = np.ones(len(norms), dtype=bool)
            filtered = acc_raw_arr[keep]
        else:
            filtered = acc_raw_arr

        if len(filtered) == 0:
            filtered = acc_raw_arr if len(acc_raw_arr) else np.zeros((1, 3))

        pos_median = np.median(filtered, axis=0)
        position_medians.append(pos_median)

        retained_raw_all.append(filtered)
        retained_expected_all.append(np.tile(expected_dir, (len(filtered), 1)))

        print(f"  Collected {len(acc_raw_arr)} samples, kept {len(filtered)} after "
              f"settle-skip + MAD filtering, median = {pos_median}")

    raw_all = np.vstack(retained_raw_all)            # (N, 3) every retained sample, all positions
    expected_all = np.vstack(retained_expected_all)  # (N, 3) matching expected direction

    # --- Stage 1: classic 6-point linear solve (on robust per-position medians) ---
    # Used ONLY to produce a well-conditioned initial guess for the nonlinear
    expected_positions = np.array([o[2] for o in orientations])
    raw_positions = np.array(position_medians)

    M, y = [], []
    for i in range(6):
        r, e = raw_positions[i], expected_positions[i]
        for comp in range(3):
            row = np.zeros(12)
            row[3 * comp: 3 * comp + 3] = r
            row[9 + comp] = 1.0
            M.append(row)
            y.append(e[comp])
    M, y = np.array(M), np.array(y)

    x0_lin, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
    A0 = x0_lin[:9].reshape(3, 3)
    b0 = x0_lin[9:]
    Ainv0 = A0.copy()
    bias0 = -np.linalg.inv(A0) @ b0

    def residuals(x, raw_pts, expected_pts):
        Ainv_ = x[:9].reshape(3, 3)
        bias_ = x[9:]
        cal = (raw_pts - bias_) @ Ainv_.T           # (N,3), acc_cal = Ainv @ (raw - bias)
        return (cal - expected_pts).ravel()

    x0 = np.concatenate([Ainv0.ravel(), bias0])
    rms_before = np.sqrt(np.mean(residuals(x0, raw_all, expected_all) ** 2))

    # --- Stage 2: robust nonlinear refit over EVERY retained sample ---
    result = least_squares(
        residuals, x0, args=(raw_all, expected_all),
        loss="soft_l1", f_scale=0.15, method="trf",
    )
    Ainv = result.x[:9].reshape(3, 3)
    bias = result.x[9:]
    rms_after = np.sqrt(np.mean(residuals(result.x, raw_all, expected_all) ** 2))

    print("\nComputed accelerometer calibration (robust refit over "
          f"{len(raw_all)} pooled samples from {len(orientations)} positions):")
    print(f"  Initial 6-point linear solve:      residual RMS = {rms_before:.4f} g")
    print(f"  After robust nonlinear refit:      residual RMS = {rms_after:.4f} g")
    print(f"  Ainv:\n{Ainv}")
    print(f"  bias: {bias}")

    return bias, Ainv


# ============================================================
# SPHERE-FORCING FUNCTION 
# ============================================================

def force_sphere(points, radius=None):
    """
    Project each point onto a sphere of given radius centered at origin.
    If radius is None, use the average norm of the points.
    """
    points = np.asarray(points)
    norms = np.linalg.norm(points, axis=1)
    if radius is None:
        radius = np.mean(norms)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-12)
    return points / norms[:, np.newaxis] * radius


# ============================================================
# PLOTTING -- SCHEME 1: Allan Deviation of Total Noise
# ============================================================

def main():

    OUTPUT_DIR = Path(config.get("plots_directory", "plots/phase_1"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.ion()

    global network
    network = UDPReceiver(port=config.get("udp_port", 5005), packet_format=config.get("packet_format", "<II6fI3f"))
    network.start()
    time.sleep(0.5)

    # ---------------- Phase 0: Accelerometer 6‑position calibration ------------
    new_bias, new_Ainv = accelerometer_calibration_6pos()
    calib_manager.calib_data["accelerometer"]["bias"] = new_bias.tolist()
    calib_manager.calib_data["accelerometer"]["scale_factor"] = new_Ainv.tolist()
    calib_manager.save_calibration(calib_manager.calib_data)
    print("\nUpdated accelerometer calibration constants in memory (full 3x3 matrix).")
    print("All subsequent accelerometer processing will use these new values.")

    # ---------------- Phase A: RM3100 hard-iron/soft-iron check ------------

    print(
        "\nPHASE A - RM3100 magnetometer calibration check.\n"
        "(JY901S internal magnetometer is ignored -- RM3100 only.)\n"
        "*** Before starting: move to a magnetically quiet spot, away from laptops,\n"
        "phones, monitors, and desks/chairs with embedded metal, and keep anything like\n"
        "that stationary relative to the sensor for the ENTIRE window below. Moving\n"
        "electronics or ferrous objects mid-sweep is the direct cause of the doubled-ring\n"
        "symptom this phase's drift check is designed to catch. ***"
    )

    mag_holds_before = guided_diagonal_holds(MAG_POSES_PRIMARY,
                                              hold_duration=config.get("mag_hold_duration", 3.0),
                                              settle_time=config.get("mag_hold_settle", 1.0))

    mag_free_sweep = collect_samples_coverage_gated_mag(
        "PHASE A - Free-rotation sweep.\n"
        "Slowly rotate the sensor through as many orientations as you can (roll it,\n"
        "pitch it, spin it in yaw) to sweep out a full sphere of headings. Watch the\n"
        "sphere map below: any '.' cell is a direction you haven't pointed toward yet.",
        seed_samples=mag_holds_before,
    )

    mag_holds_after = guided_diagonal_holds(MAG_POSES_DIAGONAL,
                                             hold_duration=config.get("mag_hold_duration", 3.0),
                                             settle_time=config.get("mag_hold_settle", 1.0))

    mag_samples = concat_sample_dicts(mag_holds_before, mag_free_sweep, mag_holds_after)

    drift_info = check_mag_drift(mag_samples)

    if len(mag_samples["mag_raw"]) > 0 and not drift_info.get("drifted", False):
        try:
            new_mag_bias, new_mag_ainv = fit_mag_ellipsoid_calibration(mag_samples)
            MAG_BIAS = new_mag_bias
            MAG_AINV = new_mag_ainv
            print("\nUpdated magnetometer calibration constants in memory "
                  "(ellipsoid fit, coverage-gated + drift-checked sweep).")
            recompute_mag_cal(mag_samples)
        except ValueError as exc:
            print(f"\n*** Skipping magnetometer calibration re-derivation: {exc} ***")
    else:
        print(
            "\n*** Skipping magnetometer calibration re-derivation -- keeping the "
            "existing MAG_BIAS/MAG_AINV placeholders. ***\n"
            "Re-run Phase A in a magnetically quiet spot per the drift warning above, "
            "then re-derive the calibration from that clean sweep."
        )

    plot_mag_calibration(mag_samples, OUTPUT_DIR / "rm3100_calibration_raw_vs_cal.png")

    # ---------------- Phase A-Accel: DEDICATED accelerometer sweep ---------
    print(
        "\nPHASE A-Accel - Dedicated accelerometer calibration check.\n"
        "This phase brackets a free-rotation sweep with guided diagonal holds\n"
        "so azimuthal (XY) coverage can't end up with a cross-shaped gap."
    )

    holds_before = guided_diagonal_holds(ACCEL_DIAGONAL_POSES_PRIMARY)

    free_sweep_samples = collect_samples_coverage_gated(
        "PHASE A-Accel - Free-rotation sweep.\n"
        "Slowly tip and roll the sensor through a wide range of tilts in every\n"
        "direction -- roll it side to side, pitch it forward/back, and rock it\n"
        "through diagonals too. Watch the live coverage readout below: any '.'\n"
        "wedge is a direction you haven't tilted toward yet.",
        seed_samples=holds_before,
    )

    holds_after = guided_diagonal_holds(ACCEL_DIAGONAL_POSES_YAWED)

    accel_sweep_samples = concat_sample_dicts(holds_before, free_sweep_samples, holds_after)

    gap_info = report_azimuth_coverage(accel_sweep_samples, phase_label="Phase A-Accel")


    plot_accel_calibration(
        accel_sweep_samples, OUTPUT_DIR / "accel_calibration_raw_vs_cal.png",
        gap_info=gap_info,
    )

    # ---------------- Phase B: gyro noise / Allan deviation check ----------

    print("\nSkipping gyro Allan deviation analysis (commented out).")
    gyro_samples = collect_samples(
        config.get("allan_collection_time", 60.0),
        "PHASE B - Gyro noise characterization (Allan Deviation).\n"
        "Set the sensor down and DO NOT MOVE IT for the entire window.\n"
        "(Longer, stiller windows -> cleaner bias-instability / rate-random-walk fits.)",
    )
    fit_report = plot_allan_deviation(gyro_samples, OUTPUT_DIR / "allan_deviation_gyro.png")

    # ---------------- Summary ----------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1 CHARACTERIZATION SUMMARY")
    print("=" * 70)
    print("Accelerometer calibration updated via 6‑position least‑squares (full 3x3 matrix).")
    print("Both accelerometer and magnetometer calibrated points are forced onto a sphere.")
    print("Accelerometer XY azimuth coverage was coverage-gated (guided diagonal holds +")
    print("free-rotation sweep bracketed by holds) and diagnosed for gaps before plotting.")
    print("Magnetometer sweep was coverage-gated on FULL sphere (azimuth + elevation),")
    print("bracketed by guided holds, checked for mid-sweep drift/interference, and used")
    print("to re-derive MAG_BIAS/MAG_AINV via an algebraic ellipsoid fit (when clean).")
    print("Magnetometer calibration plot and accelerometer plot generated.")
    print("Gyro Allan deviation analysis is temporarily disabled.")
    print(f"\nPNG copies of the plots were saved under: {OUTPUT_DIR.resolve()}")
    print("(No CSV files are written by this script.)")
    print("Close the plot windows (or Ctrl+C) to exit.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()