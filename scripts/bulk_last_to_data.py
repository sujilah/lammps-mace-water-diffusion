from ase.io import read, write
import re

xyz_file = "bulk_water_mace.xyz"
old_data = "bulk_water.data"
new_data = "bulk_continue.data"
new_xyz = "bulk_continue_last.xyz"

# Ambil frame terakhir trajectory bulk
atoms = read(xyz_file, index=-1)

# Ambil box dari bulk_water.data
with open(old_data) as f:
    text = f.read()

x = re.search(r"([\-0-9.eE+]+)\s+([\-0-9.eE+]+)\s+xlo\s+xhi", text)
y = re.search(r"([\-0-9.eE+]+)\s+([\-0-9.eE+]+)\s+ylo\s+yhi", text)
z = re.search(r"([\-0-9.eE+]+)\s+([\-0-9.eE+]+)\s+zlo\s+zhi", text)

xlo, xhi = map(float, x.groups())
ylo, yhi = map(float, y.groups())
zlo, zhi = map(float, z.groups())

Lx = xhi - xlo
Ly = yhi - ylo
Lz = zhi - zlo

atoms.set_cell([Lx, Ly, Lz])
atoms.set_pbc([True, True, True])

write(new_xyz, atoms)

write(
    new_data,
    atoms,
    format="lammps-data",
    atom_style="atomic",
    specorder=["H", "O"],
    masses=True,
)

print("Read:", xyz_file)
print("Frame atoms:", len(atoms))
print("Box:", Lx, Ly, Lz)
print("Saved:", new_data)
print("Saved:", new_xyz)
