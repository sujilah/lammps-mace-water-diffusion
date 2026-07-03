import numpy as np
import matplotlib.pyplot as plt

filename = "msd_bulk_continue.dat"
dt_ps = 0.0001      # timestep LAMMPS dalam ps, sesuaikan kalau timestep beda
msd_col = 3         # kolom MSD_O: step MSD_all MSD_H MSD_O
fit_fraction_start = 0.5
n_bootstrap = 1000
random_seed = 123

data = np.loadtxt(filename, comments="#")

step = data[:, 0]
msd = data[:, msd_col]
time_ps = step * dt_ps

# Fit pakai setengah akhir data
start = int(len(time_ps) * fit_fraction_start)
x = time_ps[start:]
y = msd[start:]

if len(x) < 3:
    raise ValueError("Not enough MSD points for fitting. Need at least 3 points.")

slope, intercept = np.polyfit(x, y, 1)

# MSD = 6 D t
D_A2_per_ps = slope / 6.0
D_cm2_per_s = D_A2_per_ps * 1.0e-4

# Residual bootstrap untuk error bar
rng = np.random.default_rng(random_seed)
y_fit = slope * x + intercept
residuals = y - y_fit

D_boot = []
for _ in range(n_bootstrap):
    resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
    y_boot = y_fit + resampled_residuals
    slope_boot, _ = np.polyfit(x, y_boot, 1)
    D_boot.append((slope_boot / 6.0) * 1.0e-4)

D_boot = np.array(D_boot)

D_err_std = np.std(D_boot, ddof=1)
D_ci_low, D_ci_high = np.percentile(D_boot, [2.5, 97.5])

ss_res = np.sum((y - y_fit) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print("=== Diffusion estimate from MSD_O ===")
print(f"File                : {filename}")
print(f"Number of points    : {len(time_ps)}")
print(f"Fit points          : {len(x)}")
print(f"Fit window          : {x[0]:.6f} to {x[-1]:.6f} ps")
print(f"Slope MSD           : {slope:.8f} Å^2/ps")
print(f"Intercept           : {intercept:.8f} Å^2")
print(f"R^2                 : {r2:.6f}")
print(f"D                   : {D_A2_per_ps:.8e} Å^2/ps")
print(f"D                   : {D_cm2_per_s:.8e} cm^2/s")
print(f"Error std           : ±{D_err_std:.8e} cm^2/s")
print(f"95% bootstrap CI    : [{D_ci_low:.8e}, {D_ci_high:.8e}] cm^2/s")

np.savetxt(
    "diffusion_bootstrap_D_cm2_s.dat",
    D_boot,
    header="Bootstrap samples of diffusion coefficient D in cm^2/s",
)

plt.figure(figsize=(6, 4))
plt.plot(time_ps, msd, "o-", label="MSD_O data", markersize=4)
plt.plot(x, y_fit, "r--", label="Linear fit window")
plt.xlabel("Time (ps)")
plt.ylabel("MSD_O (Å²)")
plt.title(f"MSD_O and diffusion fit\nD = {D_cm2_per_s:.3e} ± {D_err_std:.1e} cm²/s")
plt.legend()
plt.tight_layout()
plt.savefig("msd_diffusion_fit.png", dpi=300)

print("Saved msd_diffusion_fit.png")
print("Saved diffusion_bootstrap_D_cm2_s.dat")
