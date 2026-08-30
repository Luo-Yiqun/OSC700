# Yiqun Luo (luo2@andrew.cmu.edu)

"""
This script is used to plot the histograms of the GW+BSE properties of the materials in the dataset.
"""

import os
import json
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Plot histograms of the GW+BSE properties in the dataset.")
parser.add_argument("root", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "json"),
                    help="Path to the json/ data directory (default: ../json relative to this script).")
root = parser.parse_args().root
absorption = []
sigma = []
epsilon = []
unknown = []
n_materials = []
o_materials = []
s_materials = []

n_atom_per_molecule = {}
n_atom_per_cell = {}
rmsd = {}
gw_static_dielectric_constant = {}
fundamental_gap = {}
optical_gap = {}
triplet_gap = {}
singlet_binding_energy = {}
triplet_binding_energy = {}
df = {}
singlet_triplet_gap = {}


for folder in ("kaiji", "sub1ev", "100-125ev", "125-150ev"):
    print("Current folder:", folder)
    for file in os.listdir(os.path.join(root, folder)):
        id = file.split(".")[0]
        with open(os.path.join(root, folder, file), "r") as f:
            data = json.load(f)            
        if "gwbse" in data:
            if "bse_DF" in data["gwbse"]:
                all_features = True
                for key in ("author", "kgrid_coarse", "kgrid_fine", "epsilon_gw", "bandstructure", "vbm", "cbm", "fundamental_gap", "bse_Es", "bse_Et", "bse_DF", "bse_Delta_st", "absorption", "bse_Es_bind", "bse_Et_bind"):
                    if key not in data["gwbse"]:
                        print(f"Warning: bse_DF in but {key} not in gwbse of {file}")
                        all_features = False
                if data["geometry"]["rmsd"] > 0.4:
                    print("Warning:", file, "RMSD", data["geometry"]["rmsd"], "too large!")
                absorption.append(id)

                optical_gap[id] = data["gwbse"]["bse_Es"]
                triplet_gap[id] = data["gwbse"]["bse_Et"]
                df[id] = data["gwbse"]["bse_DF"]
                singlet_triplet_gap[id] = data["gwbse"]["bse_Delta_st"]
                if all_features:
                    singlet_binding_energy[id] = data["gwbse"]["bse_Es_bind"]
                    triplet_binding_energy[id] = data["gwbse"]["bse_Et_bind"]

            if "fundamental_gap" in data["gwbse"]:
                for key in ("author", "kgrid_coarse", "epsilon_gw", "vbm", "cbm", "fundamental_gap"):
                    if key not in data["gwbse"]:
                        print(f"Warning: fundamental_gap in but {key} not in gwbse of {file}")
                sigma.append(id)

                fundamental_gap[id] = data["gwbse"]["fundamental_gap"]

            if "epsilon_gw" in data["gwbse"]:
                for key in ("author", "kgrid_coarse", "epsilon_gw"):
                    if key not in data["gwbse"]:
                        print(f"Warning: epsilon_gw in but {key} not in gwbse of {file}")
                epsilon.append(id)

                gw_static_dielectric_constant[id] = data["gwbse"]["epsilon_gw"]

            if id not in epsilon and id not in sigma and id not in absorption:
                print("Warning:", file, "has gwbse key but probably no results")
                unknown.append(id)
            else:

                n_atom_per_molecule[id] = len(data["geometry"]["molecule"]["sites"])
                n_atom_per_cell[id] = len(data["geometry"]["relaxed_crystal"]["sites"])
                rmsd[id] = data["geometry"]["rmsd"]

                if 'N' in data["geometry"]["chemical_formula"]:
                    n_materials.append(id)
                if 'O' in data["geometry"]["chemical_formula"]:
                    o_materials.append(id)
                if 'S' in data["geometry"]["chemical_formula"]:
                    s_materials.append(id)
    print()

print(len(absorption), "absorption:", absorption)
print(len(sigma), "sigma:", sigma)
print(len(epsilon), "epsilon:", epsilon)
print(len(unknown), "unknown:", unknown)
print(len(n_materials), "n_materials:", n_materials)
print(len(o_materials), "o_materials:", o_materials)
print(len(s_materials), "s_materials:", s_materials)
print()

fig, ax = plt.subplots()
ax.hist(n_atom_per_molecule.values(), bins = 14, range = (0, 140), label = f"Count = {len(n_atom_per_molecule)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Number of atoms per molecule", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
fig.savefig("n_atom_per_molecule.png")

fig, ax = plt.subplots()
ax.hist(n_atom_per_cell.values(), bins = 32, range = (0, 800), label = f"Count = {len(n_atom_per_cell)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Number of atoms per cell", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
fig.savefig("n_atom_per_cell.png")

fig, ax = plt.subplots()
ax.hist(rmsd.values(), bins = 15, range = (0, 0.5), label = f"Count = {len(rmsd)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("RMSD", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("RMSD > 0.4:")
for k, v in rmsd.items():
    if v > 0.4:
        print(k, v)
print()
fig.savefig("rmsd.png")

fig, ax = plt.subplots()
ax.hist(gw_static_dielectric_constant.values(), bins = 48, range = (1, 13), label = f"Count = {len(gw_static_dielectric_constant)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("GW static dielectric constant", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("GW static dielectric constant outliers:")
for k, v in gw_static_dielectric_constant.items():
    if v < 2.25 or v > 8:
        print(k, v)
print()
fig.savefig("gw_static_dielectric_constant.png")

fig, ax = plt.subplots()
ax.hist(fundamental_gap.values(), bins = 24, range = (0, 6), label = f"Count = {len(fundamental_gap)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Fundamental gap", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("Fundamental gap outliers:")
for k, v in fundamental_gap.items():
    if v < 1.25 or v > 5:
        print(k, v)
print()
fig.savefig("fundamental_gap.png")

fig, ax = plt.subplots()
ax.hist(optical_gap.values(), bins = 25, range = (0, 5), label = f"Count = {len(optical_gap)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Optical gap", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("Small optical gap materials:")
for k, v in optical_gap.items():
    if v < 1.2:
        print(k, v)
print()
fig.savefig("optical_gap.png")

fig, ax = plt.subplots()
ax.hist(triplet_gap.values(), bins = 15, range = (0, 3), label = f"Count = {len(triplet_gap)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Triplet gap", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("Small triplet gap materials:")
for k, v in triplet_gap.items():
    if v < 0.6:
        print(k, v)
print()
fig.savefig("triplet_gap.png")

fig, ax = plt.subplots()
ax.hist(singlet_binding_energy.values(), bins = 30, range = (0, 3), label = f"Count = {len(singlet_binding_energy)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Singlet binding energy", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("Singlet binding energy outliers:")
for k, v in singlet_binding_energy.items():
    if v < 0.2 or v > 2.5:
        print(k, v)
print()
fig.savefig("singlet_binding_energy.png")

fig, ax = plt.subplots()
ax.hist(triplet_binding_energy.values(), bins = 40, range = (0, 4), label = f"Count = {len(triplet_binding_energy)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Triplet binding energy", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("Triplet binding energy outliers:")
for k, v in triplet_binding_energy.items():
    if v < 0.7 or v > 3.1:
        print(k, v)
print()
fig.savefig("triplet_binding_energy.png")

precision = sum(1 for v in df.values() if v > -1) / len(df)
fig, ax = plt.subplots()
ax.hist(df.values(), bins = 50, range = (-3, 2), label = f"Count = {len(df)}\nPrecision = {precision:.2f}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("DF", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("Large DF materials:")
for k, v in df.items():
    if v > -0.62:
        print(k, v)
print()
fig.savefig("df.png")

fig, ax = plt.subplots()
ax.hist(singlet_triplet_gap.values(), bins = 21, range = (0, 2.1), label = f"Count = {len(singlet_triplet_gap)}")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Singlet-triplet gap", fontsize = 18)
ax.set_ylabel("Number of structures", fontsize = 18)
ax.legend(fontsize = 16)
fig.tight_layout()
print("Small singlet-triplet gap materials:")
for k, v in singlet_triplet_gap.items():
    if v < 0.2:
        print(k, v)
print()
fig.savefig("singlet_triplet_gap.png")

plt.show()