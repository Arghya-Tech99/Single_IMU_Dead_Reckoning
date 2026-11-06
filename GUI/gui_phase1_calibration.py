
import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


import numpy as np
from core.math_utils import force_sphere
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sici

COLOR_RAW = "b"
COLOR_CAL = "r"

def overlapping_allan_deviation(data, dt, n_taus=90):

    data = np.asarray(data, dtype=float)
    n = len(data)
    theta = np.cumsum(data) * dt

    max_m = max(2, n // 2 - 1)
    m_candidates = np.unique(
        np.logspace(0, np.log10(max_m), n_taus).astype(int)
    )
    m_candidates = m_candidates[m_candidates >= 1]

    taus, sigmas = [], []
    for m in m_candidates:
        if 2 * m >= n:
            continue
        second_diff = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        var = np.mean(second_diff ** 2) / (2.0 * (m * dt) ** 2)
        taus.append(m * dt)
        sigmas.append(np.sqrt(var))

    return np.array(taus), np.array(sigmas)


def estimate_noise_coefficients(taus, sigmas):
   # Read the noise coefficients for modelling the total noise
    log_tau = np.log10(taus)
    log_sig = np.log10(np.clip(sigmas, 1e-300, None))
    slopes = np.gradient(log_sig, log_tau)

    mid = len(taus) // 2

    # N: closest-to -1/2 slope, searched in the short-tau half of the curve
    short_idx = np.arange(0, max(mid, 1))
    i_n = short_idx[np.argmin(np.abs(slopes[short_idx] + 0.5))]
    N = sigmas[i_n] * np.sqrt(taus[i_n])              # sigma_N(tau)=N/sqrt(tau)

    # K: closest-to +1/2 slope, searched in the long-tau half of the curve
    long_idx = np.arange(mid, len(taus))
    i_k = long_idx[np.argmin(np.abs(slopes[long_idx] - 0.5))]
    K = sigmas[i_k] * np.sqrt(3.0 / taus[i_k])         # sigma_K(tau)=K*sqrt(tau/3)

    # B: minimum of the curve, divided by 0.664
    i_b = int(np.argmin(sigmas))
    B = sigmas[i_b] / 0.664
    tau_at_min = taus[i_b]

    return {"N": N, "K": K, "B": B, "tau_at_min": tau_at_min}


def theoretical_sigma_N(taus, N):
    return N / np.sqrt(taus)


def theoretical_sigma_K(taus, K):
    return K * np.sqrt(taus / 3.0)


def theoretical_sigma_B(taus, B, f0):
    x = np.pi * f0 * taus
    x = np.clip(x, 1e-9, None)
    _, ci_2x = sici(2 * x)
    _, ci_4x = sici(4 * x)
    val = (2.0 * B ** 2 / np.pi) * (
        np.log(2.0)
        - (np.sin(x) ** 3 / (2.0 * x ** 2)) * (np.sin(x) + 4 * x * np.cos(x))
        + ci_2x - ci_4x
    )
    return np.sqrt(np.clip(val, 0.0, None))


# ============================================================
# DATA COLLECTION
# ============================================================


def plot_allan_deviation(samples, out_path):
    # This function is kept but not called; it remains for reference.
    if len(samples["gyro_cal"]) == 0:
        print(
            f"\nSkipping Allan Deviation plot ({out_path}): no gyro samples were "
            "collected -- see the WARNING above from collect_samples()."
        )
        return {}

    dt = float(np.median(samples["dt"])) if len(samples["dt"]) else 0.01
    gyro_cal = samples["gyro_cal"]
    axis_labels = ["x", "y", "z"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    fit_report = {}

    for i, ax in enumerate(axes):
        signal = gyro_cal[:, i]
        taus, sigmas = overlapping_allan_deviation(signal, dt)
        if len(taus) < 4:
            ax.set_title(f"gyro {axis_labels[i]} (not enough samples)")
            continue

        coeffs = estimate_noise_coefficients(taus, sigmas)
        N, K, B = coeffs["N"], coeffs["K"], coeffs["B"]
        f0 = 1.0 / (2.0 * np.pi * max(coeffs["tau_at_min"], dt))
        fit_report[axis_labels[i]] = coeffs

        sigma_N = theoretical_sigma_N(taus, N)
        sigma_K = theoretical_sigma_K(taus, K)
        sigma_B = theoretical_sigma_B(taus, B, f0)

        ax.loglog(taus, sigma_N, "--", color="#1f77b4", label=r"$\sigma_N(\tau)$")
        ax.loglog(taus, sigma_K, "--", color="#ff7f0e", label=r"$\sigma_K(\tau)$")
        ax.loglog(taus, sigma_B, "--", color="#2ca02c", label=r"$\sigma_B(\tau)$")
        ax.loglog(taus, sigmas, "-", color="#d62728", linewidth=2, label=r"$\sigma(\tau)$ measured")

        ax.set_title(f"Gyro {axis_labels[i]}-axis  (N={N:.2e}, B={B:.2e}, K={K:.2e})", fontsize=10)
        ax.set_xlabel(r"Averaging Time $\tau$ (s)")
        if i == 0:
            ax.set_ylabel(r"Allan Deviation $\sigma(\tau)$ (rad/s)")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("Allan Deviation of Total Gyro Noise (stationary window)\n"
                , fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.canvas.manager.set_window_title("Allan Deviation of Gyro Noise")
    fig.savefig(out_path, dpi=150)
    print(f"Saved Allan Deviation plot to {out_path} (window shown live)")
    plt.show(block=False)
    plt.pause(0.1)
    return fit_report


# ============================================================
# PLOTTING -- SCHEME 2: Raw vs. Calibrated Magnetometer scatter 
# ============================================================

def plot_mag_calibration(samples, out_path):
    mag_raw = samples["mag_raw"]
    mag_cal = samples["mag_cal"]

    if len(mag_raw) == 0:
        print(
            f"\nSkipping magnetometer calibration plot ({out_path}): no mag samples "
            "were collected -- see the WARNING above from collect_samples()."
        )
        return

    mag_cal_forced = force_sphere(mag_cal, radius=None)  # use average norm

    planes = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, (lab_a, lab_b, ia, ib) in zip(axes, planes):
        ax.scatter(mag_raw[:, ia], mag_raw[:, ib], marker="*", color=COLOR_RAW,
                   s=25, label="Raw Meas.")
        ax.scatter(mag_cal_forced[:, ia], mag_cal_forced[:, ib], marker="*", color=COLOR_CAL,
                   s=25, label="Calibrated (forced sphere)")
        ax.set_title(f"{lab_a}{lab_b} RM3100 Magnetometer Data")
        ax.set_xlabel(f"{lab_a} [nT]")
        ax.set_ylabel(f"{lab_b} [nT]")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("RM3100 Magnetometer: Raw vs. Calibrated\n",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.canvas.manager.set_window_title("RM3100 Magnetometer: Raw vs. Calibrated")
    fig.savefig(out_path, dpi=150)
    print(f"Saved magnetometer calibration verification plot to {out_path} (window shown live)")
    plt.show(block=False)
    plt.pause(0.1)

    # Quantitative check on forced sphere
    cal_norms = np.linalg.norm(mag_cal_forced, axis=1)
    print(f"\n--- Magnetometer forced sphere: all norms = {cal_norms.mean():.3f} (constant)")


# ============================================================
# PLOTTING -- SCHEME 3: Raw vs. Calibrated Accelerometer scatter
# ============================================================

def plot_accel_calibration(samples, out_path, gap_info=None, gap_warn_deg=45.0):
    acc_raw = samples["acc_raw"]
    acc_cal = samples["acc_cal"]

    if len(acc_raw) == 0:
        print(
            f"\nSkipping accelerometer calibration plot ({out_path}): no accel samples "
            "were collected -- see the WARNING above from collect_samples()."
        )
        return

    acc_cal_forced = force_sphere(acc_cal, radius=1.0)

    planes = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, (lab_a, lab_b, ia, ib) in zip(axes, planes):
        ax.scatter(acc_raw[:, ia], acc_raw[:, ib], marker="*", color=COLOR_RAW,
                   s=25, label="Raw Meas.")
        ax.scatter(acc_cal_forced[:, ia], acc_cal_forced[:, ib], marker="*", color=COLOR_CAL,
                   s=25, label="Calibrated (forced sphere, R=1g)")
        ax.set_title(f"{lab_a}{lab_b} Accelerometer Data")
        ax.set_xlabel(f"{lab_a} [g]")
        ax.set_ylabel(f"{lab_b} [g]")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=8)

        # Annotatation
        if lab_a == "X" and lab_b == "Y" and gap_info is not None:
            gap_deg = gap_info["gap_deg"]
            theta1, theta2 = gap_info["gap_start"], gap_info["gap_end"]
            if gap_deg > gap_warn_deg:
                theta_range = np.linspace(theta1, theta1 + gap_deg, 100)
                r = 1.15
                wedge_x = np.concatenate([[0.0], r * np.cos(np.radians(theta_range)), [0.0]])
                wedge_y = np.concatenate([[0.0], r * np.sin(np.radians(theta_range)), [0.0]])
                ax.fill(wedge_x, wedge_y, color="orange", alpha=0.2, zorder=0,
                        label=f"Uncovered gap ({gap_deg:.0f} deg)")
                ax.text(0.02, 0.02,
                        f"Coverage gap: {gap_deg:.0f} deg span uncovered\n"
                        f"between {theta1:.0f} deg and {theta2:.0f} deg",
                        transform=ax.transAxes, fontsize=8, color="darkorange",
                        va="bottom", ha="left")
                ax.legend(loc="upper right", fontsize=8)
            else:
                ax.text(0.02, 0.02,
                        f"Coverage OK (largest gap {gap_deg:.0f} deg)",
                        transform=ax.transAxes, fontsize=8, color="green",
                        va="bottom", ha="left")

    fig.suptitle("Accelerometer: Raw vs. Calibrated (full 3x3 calibration)\n",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.canvas.manager.set_window_title("Accelerometer: Raw vs. Calibrated")
    fig.savefig(out_path, dpi=150)
    print(f"Saved accelerometer calibration verification plot to {out_path} (window shown live)")
    plt.show(block=False)
    plt.pause(0.1)

    # Quick check: all norms should be 1
    cal_norms = np.linalg.norm(acc_cal_forced, axis=1)
    print(f"\n--- Accelerometer forced sphere: all norms = {cal_norms.mean():.3f} (constant)")






