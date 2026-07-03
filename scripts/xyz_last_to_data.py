from ase.io import read, write

input_xyz = "md_lammps_mace_elements.xyz"
output_data = "h2o_continue.data"
output_xyz = "h2o_continue_last.xyz"

atoms = read(input_xyz, index=-1)

# beri box besar non-periodic/periodic sederhana
atoms.set_cell([10.0, 10.0, 10.0])
atoms.center()
atoms.set_pbc([False, False, False])

write(output_xyz, atoms)

write(
    output_data,
    atoms,
    format="lammps-data",
    atom_style="atomic",
    specorder=["H", "O"],
    masses=True,
)

print("Atoms:", len(atoms))
print("Saved:", output_data)
print("Saved:", output_xyz)
