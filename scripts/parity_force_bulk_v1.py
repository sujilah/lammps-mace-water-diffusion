import re
import numpy as np
import matplotlib.pyplot as plt

ref_file = "train_bulk_v1.extxyz"
pred_file = "pred_bulk_v1.extxyz"

def read_extxyz_forces(filename, force_key):
    values = []

    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            i += 1
            continue

        header = lines[i + 1]
        m = re.search(r"Properties=([^\s]+)", header)

        if m is None:
            raise ValueError("No Properties field found")

        props = m.group(1).split(":")

        col = 0
        start_col = None

        j = 0
        while j < len(props):
            name = props[j]
            count = int(props[j + 2])

            if name == force_key:
                start_col = col
                break

            col += count
            j += 3

        if start_col is None:
            raise ValueError(f"{force_key} not found in {filename}")

        for k in range(natoms):
            parts = lines[i + 2 + k].split()
            fx, fy, fz = map(float, parts[start_col:start_col + 3])
            values.extend([fx, fy, fz])

        i += natoms + 2

    return np.array(values)

f_ref = read_extxyz_forces(ref_file, "forces")
f_pred = read_extxyz_forces(pred_file, "MACE_forces")

rmse = np.sqrt(np.mean((f_ref - f_pred) ** 2))
mae = np.mean(np.abs(f_ref - f_pred))

rmse_mev = rmse * 1000
mae_mev = mae * 1000

plt.figure(figsize=(5, 5))
plt.scatter(f_ref, f_pred, s=5, alpha=0.5)

fmin = min(f_ref.min(), f_pred.min())
fmax = max(f_ref.max(), f_pred.max())
plt.plot([fmin, fmax], [fmin, fmax], "r--")

plt.xlabel("Reference Force (eV/Å)")
plt.ylabel("Predicted Force (eV/Å)")
plt.title(
    f"Bulk Force Parity v1\n"
    f"RMSE = {rmse_mev:.2f} meV/Å, MAE = {mae_mev:.2f} meV/Å"
)

plt.tight_layout()
plt.savefig("parity_force_bulk_v1.png", dpi=300)

print("Force components:", len(f_ref))
print("RMSE =", rmse_mev, "meV/Å")
print("MAE =", mae_mev, "meV/Å")
print("Saved parity_force_bulk_v1.png")
