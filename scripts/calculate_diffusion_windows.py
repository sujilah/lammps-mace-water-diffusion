#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_msd(filename):
    data = np.loadtxt(filename, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError("MSD file must have columns: step MSD_all MSD_H MSD_O")
    return data

def linear_fit_with_bootstrap(time_ps, msd, tmin, tmax, n_bootstrap=2000, seed=123):
    mask = (time_ps >= tmin) & (time_ps <= tmax)
    x = time_ps[mask]
    y = msd[mask]

    if len(x) < 3:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    D_cm2_s = (slope / 6.0) * 1.0e-4

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rng = np.random.default_rng(seed)
    residuals = y - y_fit
    D_boot = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        y_boot = y_fit + rng.choice(residuals, size=len(residuals), replace=True)
        slope_boot, _ = np.polyfit(x, y_boot, 1)
        D_boot[i] = (slope_boot / 6.0) * 1.0e-4

    err_std = np.std(D_boot, ddof=1)
    ci_low, ci_high = np.percentile(D_boot, [2.5, 97.5])

    return {
        "tmin": tmin,
        "tmax": tmax,
        "npoints": len(x),
        "slope": slope,
        "D_cm2_s": D_cm2_s,
        "err_std": err_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r2": r2,
        "x": x,
        "y_fit": y_fit,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="msd_bulk_continue.dat")
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--species", choices=["all", "H", "O"], default="O")
parser.add_argument(
    "--windows",
    nargs="+",
    default=[
        "0:1", "0:2", "0:3", "0:4", "0:5",
        "0.5:5", "1:5", "2:5", "3:5",
        "0.5:2", "0.8:1.8", "2.5:4", "3:4.5",
    ],
)
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()

data = load_msd(args.file)

step = data[:, 0]
col_map = {"all": 1, "H": 2, "O": 3}
msd = data[:, col_map[args.species]]
time_ps = step * args.dt

print("=== Diffusion coefficient from MSD ===")
print(f"Input file      : {args.file}")
print(f"Species         : {args.species}")
print(f"Timestep        : {args.dt} ps")
print(f"Number of points: {len(time_ps)}")
print()
print("Window(ps)  N  Slope(Å²/ps)      D(cm²/s)        Error(std)      95% CI low      95% CI high     R²")

results = []

for window in args.windows:
    tmin, tmax = map(float, window.split(":"))
    res = linear_fit_with_bootstrap(time_ps, msd, tmin, tmax, args.bootstrap)
    if res is None:
        print(f"{window:10s} skipped")
        continue

    results.append(res)

    print(
        f"{tmin:4.1f}-{tmax:<4.1f} "
        f"{res['npoints']:3d} "
        f"{res['slope']:14.6e} "
        f"{res['D_cm2_s']:14.6e} "
        f"{res['err_std']:14.6e} "
        f"{res['ci_low']:14.6e} "
        f"{res['ci_high']:14.6e} "
        f"{res['r2']:8.4f}"
    )

summary = np.array([
    [
        r["tmin"], r["tmax"], r["npoints"], r["slope"],
        r["D_cm2_s"], r["err_std"], r["ci_low"], r["ci_high"], r["r2"],
    ]
    for r in results
])

header = "tmin_ps tmax_ps npoints slope_A2_ps D_cm2_s err_std_cm2_s ci95_low ci95_high R2"

np.savetxt("diffusion_window_summary.dat", summary, header=header)
np.savetxt("diffusion_window_summary.csv", summary, delimiter=",", header=header, comments="")

plt.figure(figsize=(7, 5))
plt.plot(time_ps, msd, "ko-", markersize=3, linewidth=1, label=f"MSD_{args.species}")

for r in results:
    label = f"{r['tmin']:.1f}-{r['tmax']:.1f} ps, D={r['D_cm2_s']:.2e}, R²={r['r2']:.2f}"
    plt.plot(r["x"], r["y_fit"], "--", linewidth=1.5, label=label)

plt.xlabel("Time (ps)")
plt.ylabel(f"MSD_{args.species} (Å²)")
plt.title(f"MSD_{args.species} diffusion fits with different time windows")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("diffusion_windows.png", dpi=300)

print()
print("Saved diffusion_window_summary.dat")
print("Saved diffusion_window_summary.csv")
print("Saved diffusion_windows.png")
print()
print("Scientific note:")
print("Use only windows with positive D, high R², and positive 95% CI.")
