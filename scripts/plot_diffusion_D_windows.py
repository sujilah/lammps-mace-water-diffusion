import numpy as np
import matplotlib.pyplot as plt

infile = "diffusion_window_summary.dat"
out_png = "diffusion_D_windows.png"
out_txt = "diffusion_D_stats.txt"

data = np.loadtxt(infile, comments="#")

if data.ndim == 1:
    data = data.reshape(1, -1)

# columns:
# tmin_ps tmax_ps npoints slope_A2_ps D_cm2_s err_std_cm2_s ci95_low ci95_high R2
tmin = data[:, 0]
tmax = data[:, 1]
D = data[:, 4]
D_err = data[:, 5]
ci_low = data[:, 6]
ci_high = data[:, 7]
r2 = data[:, 8]

labels = [f"{a:g}-{b:g}" for a, b in zip(tmin, tmax)]
x = np.arange(len(D))

# error bar pakai 95% CI
lower_err = D - ci_low
upper_err = ci_high - D

D_mean = np.mean(D)
D_min = np.min(D)
D_max = np.max(D)
D_std = np.std(D, ddof=1) if len(D) > 1 else 0.0

# valid window: D positif, CI bawah positif, dan R2 cukup
valid = (D > 0) & (ci_low > 0) & (r2 >= 0.5)

if np.any(valid):
    D_valid_mean = np.mean(D[valid])
    D_valid_min = np.min(D[valid])
    D_valid_max = np.max(D[valid])
    D_valid_std = np.std(D[valid], ddof=1) if np.sum(valid) > 1 else 0.0
else:
    D_valid_mean = np.nan
    D_valid_min = np.nan
    D_valid_max = np.nan
    D_valid_std = np.nan

plt.figure(figsize=(9, 5))

plt.errorbar(
    x,
    D,
    yerr=[lower_err, upper_err],
    fmt="o",
    capsize=4,
    label="D per fitting window with 95% CI",
)

plt.axhline(D_mean, linestyle="--", label=f"Mean(all) = {D_mean:.2e} cm²/s")
plt.axhline(D_min, linestyle=":", label=f"Min(all) = {D_min:.2e} cm²/s")
plt.axhline(D_max, linestyle=":", label=f"Max(all) = {D_max:.2e} cm²/s")

if np.any(valid):
    plt.axhspan(D_valid_min, D_valid_max, alpha=0.12, label="Valid-window min–max range")
    plt.axhline(D_valid_mean, linestyle="-.", label=f"Mean(valid) = {D_valid_mean:.2e} cm²/s")

plt.xticks(x, labels, rotation=45, ha="right")
plt.xlabel("Fitting window (ps)")
plt.ylabel("Diffusion coefficient D (cm²/s)")
plt.title("Diffusion Coefficient from MSD_O: Window Dependence")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out_png, dpi=300)

with open(out_txt, "w") as f:
    f.write("Diffusion coefficient statistics\n")
    f.write("================================\n")
    f.write(f"Input file: {infile}\n")
    f.write(f"Number of windows: {len(D)}\n\n")

    f.write("All windows:\n")
    f.write(f"  mean D = {D_mean:.8e} cm^2/s\n")
    f.write(f"  std  D = {D_std:.8e} cm^2/s\n")
    f.write(f"  min  D = {D_min:.8e} cm^2/s\n")
    f.write(f"  max  D = {D_max:.8e} cm^2/s\n\n")

    f.write("Valid windows criterion: D > 0, CI_low > 0, R2 >= 0.5\n")
    f.write(f"Number of valid windows: {np.sum(valid)}\n")

    if np.any(valid):
        f.write(f"  mean D valid = {D_valid_mean:.8e} cm^2/s\n")
        f.write(f"  std  D valid = {D_valid_std:.8e} cm^2/s\n")
        f.write(f"  min  D valid = {D_valid_min:.8e} cm^2/s\n")
        f.write(f"  max  D valid = {D_valid_max:.8e} cm^2/s\n")
    else:
        f.write("  No valid windows found. MSD likely not in reliable diffusive regime.\n")

print("Saved", out_png)
print("Saved", out_txt)
print("All windows:")
print(f"  mean D = {D_mean:.8e} cm^2/s")
print(f"  std  D = {D_std:.8e} cm^2/s")
print(f"  min  D = {D_min:.8e} cm^2/s")
print(f"  max  D = {D_max:.8e} cm^2/s")
print("Valid windows criterion: D > 0, CI_low > 0, R2 >= 0.5")
print(f"  valid windows = {np.sum(valid)} / {len(D)}")

if np.any(valid):
    print(f"  mean D valid = {D_valid_mean:.8e} cm^2/s")
    print(f"  min  D valid = {D_valid_min:.8e} cm^2/s")
    print(f"  max  D valid = {D_valid_max:.8e} cm^2/s")
