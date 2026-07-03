import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_msd(filename):
    data = np.loadtxt(filename, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError("Expected columns: step MSD_all MSD_H MSD_O")
    return data

def fit_diffusion(step, msd, dt_ps, fit_start_fraction=0.5, n_bootstrap=2000, seed=123):
    time_ps = step * dt_ps
    start = int(len(time_ps) * fit_start_fraction)

    x = time_ps[start:]
    y = msd[start:]

    if len(x) < 3:
        raise ValueError("Not enough points in fit window")

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    D_A2_ps = slope / 6.0
    D_cm2_s = D_A2_ps * 1.0e-4

    residuals = y - y_fit
    rng = np.random.default_rng(seed)

    boot = []
    for _ in range(n_bootstrap):
        yb = y_fit + rng.choice(residuals, size=len(residuals), replace=True)
        slope_b, _ = np.polyfit(x, yb, 1)
        boot.append((slope_b / 6.0) * 1.0e-4)

    boot = np.array(boot)

    err = np.std(boot, ddof=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "dt_ps": dt_ps,
        "fit_start_fraction": fit_start_fraction,
        "t0": x[0],
        "t1": x[-1],
        "slope": slope,
        "intercept": intercept,
        "D_cm2_s": D_cm2_s,
        "err": err,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r2": r2,
        "time_ps": time_ps,
        "msd": msd,
        "x": x,
        "y_fit": y_fit,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="msd_bulk_continue.dat")
    parser.add_argument("--species", choices=["all", "H", "O"], default="O")
    parser.add_argument("--dt", nargs="+", type=float, default=[0.0001, 0.0005, 0.001])
    parser.add_argument("--fit-start", nargs="+", type=float, default=[0.5])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data = load_msd(args.file)

    step = data[:, 0]
    col_map = {"all": 1, "H": 2, "O": 3}
    msd = data[:, col_map[args.species]]

    print(f"File: {args.file}")
    print(f"Species/MSD column: {args.species}")
    print(f"Number of MSD points: {len(step)}")
    print()
    print("dt_ps fit_start t_fit(ps) slope(A^2/ps) D(cm^2/s) err_std 95%_CI_low 95%_CI_high R2")

    results = []

    for dt in args.dt:
        for fs in args.fit_start:
            res = fit_diffusion(step, msd, dt, fs, args.bootstrap, args.seed)
            results.append(res)

            print(
                f"{dt:.7g} {fs:.2f} "
                f"{res['t0']:.4f}-{res['t1']:.4f} "
                f"{res['slope']:.8e} "
                f"{res['D_cm2_s']:.8e} "
                f"{res['err']:.8e} "
                f"{res['ci_low']:.8e} "
                f"{res['ci_high']:.8e} "
                f"{res['r2']:.5f}"
            )

    with open("diffusion_multi_dt_summary.dat", "w") as f:
        f.write("# dt_ps fit_start t0_ps t1_ps slope_A2_ps D_cm2_s err_std ci95_low ci95_high R2\n")
        for res in results:
            f.write(
                f"{res['dt_ps']} {res['fit_start_fraction']} "
                f"{res['t0']} {res['t1']} {res['slope']} "
                f"{res['D_cm2_s']} {res['err']} "
                f"{res['ci_low']} {res['ci_high']} {res['r2']}\n"
            )

    plt.figure(figsize=(7, 5))

    first_fit = args.fit_start[0]
    for res in results:
        if res["fit_start_fraction"] == first_fit:
            label = f"dt={res['dt_ps']} ps, D={res['D_cm2_s']:.2e}±{res['err']:.1e}"
            plt.plot(res["time_ps"], res["msd"], "o-", markersize=3, alpha=0.35)
            plt.plot(res["x"], res["y_fit"], "--", label=label)

    plt.xlabel("Time (ps)")
    plt.ylabel(f"MSD_{args.species} (Å²)")
    plt.title(f"MSD_{args.species} diffusion fit for different timestep assumptions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"msd_diffusion_multi_dt_{args.species}.png", dpi=300)

    print()
    print("Saved diffusion_multi_dt_summary.dat")
    print(f"Saved msd_diffusion_multi_dt_{args.species}.png")
    print("Note: changing dt rescales D but does not fix negative/nonlinear MSD.")

if __name__ == "__main__":
    main()
