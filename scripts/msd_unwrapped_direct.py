import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_unwrapped_lammpstrj(path):
    steps = []
    frames = []
    types_ref = None
    ids_ref = None

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        step = int(lines[i + 1].strip())
        natoms = int(lines[i + 3].strip())

        atom_header_idx = i + 8
        cols = lines[atom_header_idx].split()[2:]

        c_id = cols.index("id")
        c_type = cols.index("type")
        c_xu = cols.index("xu")
        c_yu = cols.index("yu")
        c_zu = cols.index("zu")

        block = np.zeros((natoms, 5), dtype=float)

        for j in range(natoms):
            p = lines[atom_header_idx + 1 + j].split()
            block[j, 0] = int(p[c_id])
            block[j, 1] = int(p[c_type])
            block[j, 2] = float(p[c_xu])
            block[j, 3] = float(p[c_yu])
            block[j, 4] = float(p[c_zu])

        order = np.argsort(block[:, 0])
        block = block[order]

        ids = block[:, 0].astype(int)
        types = block[:, 1].astype(int)
        xyz = block[:, 2:5]

        if ids_ref is None:
            ids_ref = ids
            types_ref = types
        else:
            if not np.array_equal(ids, ids_ref):
                raise RuntimeError("Urutan/id atom berubah antar frame.")

        steps.append(step)
        frames.append(xyz)

        i = atom_header_idx + 1 + natoms

    return np.array(steps), np.array(frames), types_ref


def msd_from_frames(frames, types, com_no=True):
    r0 = frames[0]

    idx_h = np.where(types == 1)[0]
    idx_o = np.where(types == 2)[0]

    msd_all = []
    msd_h = []
    msd_o = []

    for r in frames:
        dr = r - r0

        if not com_no:
            drift = dr.mean(axis=0)
            dr = dr - drift

        sq = np.sum(dr * dr, axis=1)

        msd_all.append(np.mean(sq))
        msd_h.append(np.mean(sq[idx_h]))
        msd_o.append(np.mean(sq[idx_o]))

    return np.array(msd_all), np.array(msd_h), np.array(msd_o)


def fit_D(time_ps, msd, windows, nboot=2000, seed=123):
    rng = np.random.default_rng(seed)
    rows = []

    for t0, t1 in windows:
        mask = (time_ps >= t0) & (time_ps <= t1)
        x = time_ps[mask]
        y = msd[mask]

        if len(x) < 3:
            continue

        slope, intercept = np.polyfit(x, y, 1)
        yfit = slope * x + intercept

        # 3D diffusion: MSD = 6Dt
        D = slope / 6.0 * 1e-4  # cm^2/s

        residuals = y - yfit
        boot = []

        for _ in range(nboot):
            yb = yfit + rng.choice(residuals, len(residuals), replace=True)
            sb, _ = np.polyfit(x, yb, 1)
            boot.append(sb / 6.0 * 1e-4)

        boot = np.array(boot)
        err = boot.std(ddof=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        ss_res = np.sum((y - yfit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append([t0, t1, len(x), slope, D, err, lo, hi, r2])

    return np.array(rows)


parser = argparse.ArgumentParser()
parser.add_argument("--traj", default="bulk_water_mace_medium_unwrapped.lammpstrj")
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--com", choices=["no", "yes"], default="no")
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()

steps, frames, types = read_unwrapped_lammpstrj(args.traj)

time_ps = (steps - steps[0]) * args.dt

msd_all, msd_h, msd_o = msd_from_frames(
    frames,
    types,
    com_no=(args.com == "no")
)

out_msd = "msd_from_unwrapped_com{}.dat".format(args.com)

np.savetxt(
    out_msd,
    np.column_stack([steps, time_ps, msd_all, msd_h, msd_o]),
    header="step time_ps MSD_all_A2 MSD_H_A2 MSD_O_A2"
)

plt.figure(figsize=(6, 4))
plt.plot(time_ps, msd_all, "o-", ms=3, label="all")
plt.plot(time_ps, msd_h, "o-", ms=3, label="H")
plt.plot(time_ps, msd_o, "o-", ms=3, label="O")
plt.xlabel("Time (ps)")
plt.ylabel("MSD (Å²)")
plt.title("MSD from xu yu zu, com {}".format(args.com))
plt.legend()
plt.tight_layout()

out_png = "msd_from_unwrapped_com{}.png".format(args.com)
plt.savefig(out_png, dpi=300)

windows = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0.5, 5),
    (1, 5),
    (2, 5),
    (3, 5),
]

rows = fit_D(time_ps, msd_o, windows, nboot=args.bootstrap)

out_diff = "diffusion_from_unwrapped_com{}.dat".format(args.com)

np.savetxt(
    out_diff,
    rows,
    header="tmin_ps tmax_ps npoints slope_A2_ps D_cm2_s err_std_cm2_s ci95_low ci95_high R2"
)

print("Frames:", len(frames))
print("Atoms:", frames.shape[1])
print("H atoms:", np.sum(types == 1))
print("O atoms:", np.sum(types == 2))
print("Time ps:", time_ps[0], "to", time_ps[-1])
print("Final MSD_all/H/O:", msd_all[-1], msd_h[-1], msd_o[-1])

print("Saved", out_msd)
print("Saved", out_png)
print("Saved", out_diff)

print("\nDiffusion from MSD_O:")
print("window_ps n slope_A2_ps D_cm2_s err_std ci_low ci_high R2")

for r in rows:
    print(
        "{:.1f}-{:.1f} {} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.4f}".format(
            r[0],
            r[1],
            int(r[2]),
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
        )
    )
