import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("msd_bulk_continue.dat", comments="#")

step = data[:, 0]
msd_all = data[:, 1]
msd_H = data[:, 2]
msd_O = data[:, 3]

# timestep LAMMPS kamu
dt_ps = 0.0001
time_ps = step * dt_ps

plt.figure()
plt.plot(time_ps, msd_all, label="All")
plt.plot(time_ps, msd_H, label="H")
plt.plot(time_ps, msd_O, label="O")

plt.xlabel("Time (ps)")
plt.ylabel("MSD (Å²)")
plt.title("MSD Bulk Water")
plt.legend()
plt.tight_layout()
plt.savefig("msd_bulk_continue.png", dpi=300)

print("Saved msd_bulk_continue.png")
