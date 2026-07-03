import re
import numpy as np
import matplotlib.pyplot as plt

ref_file = "train_bulk_v1.extxyz"
pred_file = "pred_bulk_v1.extxyz"

def read_energies(filename, key):
    energies = []
    natoms = []

    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        try:
            n = int(lines[i].strip())
        except ValueError:
            i += 1
            continue

        header = lines[i + 1]
        m = re.search(key + r'=([-0-9.eE+]+)', header)

        if m:
            energies.append(float(m.group(1)))
            natoms.append(n)

        i += n + 2

    return np.array(energies), np.array(natoms)

eref, nref = read_energies(ref_file, "energy")
epred, npred = read_energies(pred_file, "MACE_energy")

if len(eref) != len(epred):
    raise ValueError("Jumlah frame reference dan prediction tidak sama")

eref_pa = eref / nref
epred_pa = epred / npred

rmse = np.sqrt(np.mean((eref_pa - epred_pa) ** 2))
rmse_mev_atom = rmse * 1000

plt.figure(figsize=(5, 5))
plt.scatter(eref_pa, epred_pa, s=25)

emin = min(eref_pa.min(), epred_pa.min())
emax = max(eref_pa.max(), epred_pa.max())
plt.plot([emin, emax], [emin, emax], "r--")

plt.xlabel("Reference Energy per atom (eV/atom)")
plt.ylabel("Predicted Energy per atom (eV/atom)")
plt.title(f"Bulk Energy Parity v1\nRMSE = {rmse_mev_atom:.2f} meV/atom")

plt.tight_layout()
plt.savefig("parity_energy_bulk_v1.png", dpi=300)

print("Frames:", len(eref))
print("RMSE =", rmse_mev_atom, "meV/atom")
print("Saved parity_energy_bulk_v1.png")
