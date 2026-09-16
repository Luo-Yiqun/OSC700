# Yiqun Luo (luo2@andrew.cmu.edu)
 
import json
import os
import re
import argparse
import numpy as np

from collections import deque
from numpy.char import isdigit
from scipy.sparse import csgraph
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.local_env import JmolNN



def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def s2formula(s : str) -> dict:
    """
    Convert the string to the chemical formula in dict:
    H6 C6 -> {'C': 6, 'H': 6}
    """
    formula = {}
    for tmp in s.split():
        formula[re.match(r"[A-Z][a-z]*", tmp).group()] = int(re.search(r"\d+$", tmp).group())
    return formula


def extract_molecules(structure: Structure) -> list[Molecule]:
    """
    Return one PBC-unwrapped pymatgen ``Molecule`` per connected molecular fragment.

    Uses the JmolNN bond definition; component membership comes from
    ``scipy.csgraph``, and a BFS over the raw ``nn_info`` (which retains
    per-edge periodic image info) undoes periodic wrapping so every fragment
    becomes a stand-alone molecule with finite-cluster Cartesians.
    """
    nn_info = JmolNN().get_all_nn_info(structure)
    n = len(nn_info)

    adj = np.zeros((n, n), dtype=int)
    for i, neighbours in enumerate(nn_info):
        for nb in neighbours:
            adj[i, nb["site_index"]] = 1
    _, labels = csgraph.connected_components(adj, directed=False)

    lattice = structure.lattice.matrix
    cart_coords = structure.cart_coords
    species = [str(s) for s in structure.species]

    molecules: list[Molecule] = []
    for comp_id in range(int(labels.max()) + 1):
        comp = np.where(labels == comp_id)[0]
        anchor = int(comp[0])
        offsets = {anchor: np.zeros(3, dtype=float)}
        queue, visited = deque([anchor]), {anchor}
        while queue:
            u = queue.popleft()
            for nb in nn_info[u]:
                v = nb["site_index"]
                if v in visited or labels[v] != comp_id:
                    continue
                # nn_info[u] gives v's periodic image relative to u, so adding
                # is unconditionally correct (sign falls out naturally).
                offsets[v] = offsets[u] + np.array(nb["image"], dtype=float)
                visited.add(v)
                queue.append(v)
        positions = np.stack([cart_coords[i] + offsets[i] @ lattice for i in comp])
        molecules.append(Molecule([species[i] for i in comp], positions))
    return molecules


def check_rmsd(rmsd: float, low_threshold : float = 0.01, high_threshold : float = 0.4) -> None:
    if rmsd is None:
        print("RMSD is None.")
    elif rmsd < low_threshold:
        print(f"RMSD for is too low: {rmsd}. Please make sure this structure did be relaxed properly.")
        print()
    elif rmsd > high_threshold:
        print(f"RMSD for is too high: {rmsd}. Please make sure this structure is correct.")
        print()
    return


def get_RMSD(id: str, cif_file: str) -> float:
    try:
        from ccdc.io import CrystalReader
        from ccdc.crystal import PackingSimilarity
    except ImportError:
        print("ccdc package not installed, cannot calculate RMSD. Will return None")
        print()
        return None
    
    csd = CrystalReader('csd')
    exp = csd.crystal(id)
    crystal_reader = CrystalReader(cif_file, format = 'cif')
    relaxed_struct = crystal_reader[0]
    similarity_engine = PackingSimilarity()
    similarity_engine.settings.packing_shell_size = 20

    h = similarity_engine.compare(exp, relaxed_struct)
    if h is None:
        similarity_engine.settings.allow_molecular_differences = True
        h = similarity_engine.compare(exp, relaxed_struct)
    rmsd = h.rmsd
    
    if rmsd is None:
        similarity_engine.settings.allow_molecular_differences = True
        h = similarity_engine.compare(exp, relaxed_struct)
        rmsd = h.rmsd

    check_rmsd(rmsd)
    return rmsd


def write_id(id: str, json_file: dict) -> dict:
    """
    Write down the CSD reference code.
    """
    print("Writing id for", id)
    print()
    if "struct_id" in json_file:
        assert json_file["struct_id"] == id, "struct_id mismatch"
    else:
        json_file["struct_id"] = id
    return json_file


def write_relax(
        id: str,
        json_file: dict,
        cif_file: str = None,
        author: str = None,
        relax_code: str = None,
        overwrite: bool = False
        ) -> dict:
    """
    Write down the relaxed geometry from [CSDcode].cif
    
    Please make sure:
    I) Already install dbaAutomator: https://github.com/xingyu-alfred-liu/dbaAutomator/tree/master
    II) author and relax_code are correct.
        Current options:
        author: Brian, Keya, Vincent, Yiqun, Siyu, Xuyan, Yi, Jiayi
        relax_code: FHI-aims, Quantum Espresso
    III) if cif_file is not inputted, will not write geometry.
    """
    if not cif_file or not os.path.exists(cif_file):
        print(f"CIF file {cif_file} not inputted or does not exist. Will not write geometry.")
        print()
        return json_file
    else:
        print("Writing relaxation geometry for", id)
        print()

    if "geometry" in json_file:
        print("geometry already in the json file, may or may not overwrite it")
    else:
        print(f"No geometry in the json file, Adding geometry")
        json_file["geometry"] = {}
    print()

    unitcell = Structure.from_file(cif_file)

    print("Input author:", author)
    print("Please make sure to have the correct author.")
    if "author" not in json_file["geometry"] or overwrite:
        if author is not None:
            json_file["geometry"]["author"] = author
    print()
    
    print("Input relaxation code:", relax_code)
    print("Please make sure to type the correct relaxation code.")
    if "relax_code" not in json_file["geometry"] or overwrite:
        json_file["geometry"]["relax_code"] = relax_code
    print()

    if "relaxed_crystal" not in json_file["geometry"] or overwrite:
        json_file["geometry"]["relaxed_crystal"] = unitcell.as_dict()

    if "rmsd" not in json_file["geometry"] or json_file["geometry"]["rmsd"] is None or overwrite:
        rmsd = get_RMSD(id, cif_file)
        if rmsd is not None:
            json_file["geometry"]["rmsd"] = rmsd
    
    if "molecule" not in json_file["geometry"] or overwrite:
        molecules = extract_molecules(unitcell)
        formulas = {mol.formula for mol in molecules}
        if len(formulas) > 1:
            print(f"Warning: fragments with differing formulas found: {formulas}. Storing the largest fragment.")
        molecule = max(molecules, key=len)
        json_file["geometry"]["molecule"] = molecule.as_dict()
    else:
        molecule = Molecule.from_dict(json_file["geometry"]["molecule"])
    print()

    if "chemical_formula" not in json_file["geometry"] or overwrite:
        json_file["geometry"]["chemical_formula"] = s2formula(molecule.formula)

    return json_file
    

def write_dft(id: str, json_file: dict) -> dict:
    return json_file


def write_gwbse(
        id: str,
        json_file: dict,
        root_folder: str = None,
        author: str = None,
        overwrite: bool = False
        ) -> dict:
    """
    Write down GW+BSE results from BerkeleyGW

    Please make sure the files you want to write are under the root folder:
    1-mf/1-scf/in
    1-mf/2.1-wfn/kgrid.in
    1-mf/3.1-wfn-fi/kgrid.in
    1-mf/4-bandstructure/kpoints
    2-bgw/1-epsilon/epsilon.out
    2-bgw/2-sigma/sigma.inp
    2-bgw/2-sigma/sigma.out
    2-bgw/4-a-absorption/absorption.inp
    2-bgw/4-b-absorption/absorption.inp
    2-bgw/4-c-absorption/absorption.inp
    2-bgw/4-triplet-absorption/absorption.inp
    2-bgw/4-a-absorption/absorption_eh.dat
    2-bgw/4-b-absorption/absorption_eh.dat
    2-bgw/4-c-absorption/absorption_eh.dat
    2-bgw/4-a-absorption/eigenvalues.dat
    2-bgw/4-b-absorption/eigenvalues.dat
    2-bgw/4-c-absorption/eigenvalues.dat
    2-bgw/4-triplet-absorption/eigenvalues.dat
    2-bgw/5-bandstructure/bandstructure.dat
    """
    if not root_folder or not os.path.exists(root_folder):
        print(f"Root folder {root_folder} not inputted or does not exist. Will not write gwbse.")
        print()
        return json_file
    else:
        print("Writing gwbse for", id)
        print()

    if "gwbse" in json_file:
        print("GW+BSE already in the json file, may or may not overwrite it")
    else:
        print(f"No GW+BSE in the json file, adding GW+BSE")
        json_file["gwbse"] = {}
    print()
    
    print("Input author:", author)
    print("Please make sure to have the correct author.")
    if "author" not in json_file["gwbse"] or overwrite:
        if author is not None:
            json_file["gwbse"]["author"] = author
    print()

    all_collected = True
    for tmp in (
        "1-mf/1-scf/in",
        "1-mf/2.1-wfn/kgrid.in",
        "1-mf/3.1-wfn-fi/kgrid.in",
        "1-mf/4-bandstructure/kpoints",
        "2-bgw/1-epsilon/epsilon.out",
        "2-bgw/2-sigma/sigma.inp",
        "2-bgw/2-sigma/sigma.out",
        "2-bgw/2-sigma/sigma_hp.log",
        "2-bgw/4-a-absorption/absorption.inp",
        "2-bgw/4-b-absorption/absorption.inp",
        "2-bgw/4-c-absorption/absorption.inp",
        "2-bgw/4-triplet-absorption/absorption.inp",
        "2-bgw/4-a-absorption/absorption_eh.dat",
        "2-bgw/4-b-absorption/absorption_eh.dat",
        "2-bgw/4-c-absorption/absorption_eh.dat",
        "2-bgw/4-a-absorption/eigenvalues.dat",
        "2-bgw/4-b-absorption/eigenvalues.dat",
        "2-bgw/4-c-absorption/eigenvalues.dat",
        "2-bgw/4-triplet-absorption/eigenvalues.dat",
        "2-bgw/5-bandstructure/inteqp.inp",
        "2-bgw/5-bandstructure/bandstructure.dat"
        ):
        path = os.path.join(root_folder, tmp)
        if not os.path.exists(path):
            print(f"File {path} does not exist. Corresponding parts will not be written.")
            all_collected = False
    if all_collected:
        print("All GW+BSE files exist.")
    print()

    path = os.path.join(root_folder, "1-mf/1-scf/in")
    if os.path.exists(path):
        if "mf_pseudo_potential" not in json_file["gwbse"] or overwrite:
            with open(path, 'r') as f:
                for line in f:
                    if "pseudo_dir" in line:
                        if "Hartwigesen-Goedecker-Hutter-PBE" in line:
                            json_file["gwbse"]["mf_pseudo_potential"] = "HGH"
                        elif "sg1.2" in line:
                            json_file["gwbse"]["mf_pseudo_potential"] = "ONCV"
                        else:
                            print(f"Unknown pseudo potential: {line}!")
                        break
        if "mf_ecutwfc" not in json_file["gwbse"] or overwrite:
            with open(path, 'r') as f:
                for line in f:
                    if "ecutwfc" in line:
                        json_file["gwbse"]["mf_ecutwfc"] = float(line.split()[-1])
                        break

    if "kgrid_coarse" not in json_file["gwbse"] or overwrite:
        path = os.path.join(root_folder, "1-mf/2.1-wfn/kgrid.in")
        if os.path.exists(path):
            with open(path, 'r') as f:
                json_file["gwbse"]["kgrid_coarse"] = list(map(int, f.readlines()[0].split()))
    
    if "kgrid_fine" not in json_file["gwbse"] or overwrite:
        path = os.path.join(root_folder, "1-mf/3.1-wfn-fi/kgrid.in")
        if os.path.exists(path):
            with open(path, 'r') as f:
                json_file["gwbse"]["kgrid_fine"] = list(map(int, f.readlines()[0].split()))

    if "epsilon_gw" not in json_file["gwbse"] or overwrite:
        path = os.path.join(root_folder, "2-bgw/1-epsilon/epsilon.out")
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    if "q-pt      1: Head of Epsilon Inverse" in line:
                        real, imag = list(map(float, line.split()[-2:]))
                        assert imag < 1e-3, "Imaginary part of head of Epsilon Inverse is not 0"
                        json_file["gwbse"]["epsilon_gw"] = 1 / real
                        break

    path = os.path.join(root_folder, "2-bgw/2-sigma/sigma.inp")
    n_sigma_k_points = 0 # Only for check purposes.
    if os.path.exists(path):
        with open(path, 'r') as f:
            start = False
            for line in f:
                if "begin" in line:
                    start = True
                elif "end" in line:
                    start = False
                    break
                elif start:
                    assert len(line.split()) == 4, "sigma.inp k point format changed"
                    n_sigma_k_points += 1

    path = os.path.join(root_folder, "2-bgw/2-sigma/sigma.out")
    if os.path.exists(path):
        if "screen_Coulomb_cutoff" not in json_file["gwbse"] or overwrite:
            with open(path, 'r') as f:
                for line in f:
                    if "Cutoff of the screened Coulomb interaction (Ry)" in line:
                        json_file["gwbse"]["screen_Coulomb_cutoff"] = float(line.split()[-1])
                        break
        if "number_bands_GW" not in json_file["gwbse"] or overwrite:
            with open(path, 'r') as f:
                for line in f:
                    if "Total number of bands in the calculation" in line:
                        json_file["gwbse"]["number_bands_GW"] = int(line.split()[-1])
                        break

    path = os.path.join(root_folder, "2-bgw/2-sigma/sigma_hp.log")
    if os.path.exists(path):
        if "dos" not in json_file["gwbse"] or overwrite:
            json_file["gwbse"]["dos"] = {"kpoints": [], "val": []}
            with open(path, 'r') as f:
                start = False
                columns = None
                indexes = None
                for line in f:
                    result_re = re.match(r"\s*k\s*=\s*([+-]*\d+\.\d+)\s*([+-]*\d+\.\d+)\s*([+-]*\d+\.\d+)\s*ik\s*=\s*\d+\s*spin\s*=\s*\d+", line)
                    result_split = line.strip().split()
                    if result_re:
                        json_file["gwbse"]["dos"]["kpoints"].append(list(map(float, result_re.groups())))
                    elif len(result_split) > 0 and result_split[0] == 'n':
                        if columns:
                            assert columns == result_split, "Columns in sigma_hp.log changed"
                        else:
                            columns = result_split
                            indexes = list(i for i, col in enumerate(columns) if col in ('n', "Eo", "Eqp1"))
                        start = True
                        json_file["gwbse"]["dos"]["val"].append([])
                    elif re.match(r"=+", line):
                        print("End of sigma_hp.log")
                        break
                    elif start and len(result_split) > 0:
                        assert all([is_number(i) for i in result_split]), f"Non-numeric value in line {line} of sigma_hp.log!"
                        json_file["gwbse"]["dos"]["val"][-1].append(list(map(float, [result_split[index] for index in indexes])))
            if n_sigma_k_points > 0:
                assert n_sigma_k_points == len(json_file["gwbse"]["dos"]["kpoints"]) == len(json_file["gwbse"]["dos"]["val"]), "Number of k points in sigma_hp.log does not match sigma.inp"

    if "bandstructure" not in json_file["gwbse"]:
        json_file["gwbse"]["bandstructure"] = {}
    
    if "kpoints" not in json_file["gwbse"]["bandstructure"] or overwrite:
        path = os.path.join(root_folder, "1-mf/4-bandstructure/kpoints")
        if os.path.exists(path):
            with open(path, 'r') as f:
                json_file["gwbse"]["bandstructure"]["kpoints"] = list(map(lambda x: x.strip(), f.readlines()))

    if "val" not in json_file["gwbse"]["bandstructure"] or overwrite:
        path = os.path.join(root_folder, "2-bgw/5-bandstructure/bandstructure.dat")
        if os.path.exists(path):
            with open(path, 'r') as f:
                json_file["gwbse"]["bandstructure"]["val"] = list(map(lambda x: list(map(float, x.split())), f.readlines()[2:]))

    if "fundamental_gap" in json_file["gwbse"]:
        for key in ("vbm", "cbm", "fundamental_gap"):
            if key not in json_file["gwbse"]:
                print(f"Warning: fundamental_gap in but {key} not in the json file. This typically does not happen. There may be some bugs in previous and/or this run")
    if "fundamental_gap" not in json_file["gwbse"] or overwrite:
        inteqp_in_path = os.path.join(root_folder, "2-bgw/5-bandstructure/inteqp.inp")
        bandstructure_dat_path = os.path.join(root_folder, "2-bgw/5-bandstructure/bandstructure.dat")
        if os.path.exists(inteqp_in_path) and os.path.exists(bandstructure_dat_path):
            with open(inteqp_in_path, 'r') as f:
                for line in f:
                    if "number_val_bands_fine" in line:
                        n_val = int(line.split()[-1])
                    elif "number_cond_bands_fine" in line:
                        n_cond = int(line.split()[-1])
            with open(bandstructure_dat_path, 'r') as f:
                start_idx, end_idx = None, None
                val_structure, cond_structure = [], []
                for line in f.readlines()[2:]:
                    band_idx = int(line.split()[1])
                    if start_idx is None:
                        start_idx = band_idx
                    elif band_idx < start_idx + n_val:
                        val_structure.append(list(map(float, line.split()[1:])))
                    elif band_idx >= start_idx + n_val:
                        cond_structure.append(list(map(float, line.split()[1:])))
                    end_idx = band_idx
                assert end_idx - start_idx + 1 == n_val + n_cond, "Number of bands does not match"
            val_structure = np.array(val_structure)
            cond_structure = np.array(cond_structure)
            vbm = np.argmax(val_structure[:, 5])
            cbm = np.argmin(cond_structure[:, 5])
            json_file["gwbse"]["vbm"] = val_structure[vbm][:4].tolist()
            json_file["gwbse"]["cbm"] = cond_structure[cbm][:4].tolist()
            json_file["gwbse"]["fundamental_gap"] = cond_structure[cbm][5] - val_structure[vbm][5]

    if "number_val_bands_fine" in json_file["gwbse"]:
        for key in ("number_val_bands_coarse", "number_cond_bands_coarse", "number_val_bands_fine", "number_cond_bands_fine"):
            if key not in json_file["gwbse"]:
                print(f"Warning: {key} in but not in the json file. This typically does not happen. There may be some bugs in previous and/or this run")
    if "number_val_bands_fine" not in json_file["gwbse"] or overwrite:
        path = os.path.join(root_folder, "2-bgw/4-a-absorption/absorption.inp")
        properties = ("number_val_bands_coarse", "number_cond_bands_coarse", "number_val_bands_fine", "number_cond_bands_fine")
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    for property in properties:
                        if property in line:
                            json_file["gwbse"][property] = int(line.split()[-1])
        for task in ("a", "b", "c", "triplet"):
            path = os.path.join(root_folder, f"2-bgw/4-{task}-absorption/absorption.inp")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        for property in properties:
                            if property in line and int(line.split()[-1]) != json_file["gwbse"][property]:
                                print(f"{property} in 4-{task}-absorption/absorption.inp does not match the {property} in 4-a-absorption/absorption.inp. This typically does not happen. Please check absorption input files!")

    if "bse_Es" not in json_file["gwbse"] or overwrite:
        path = os.path.join(root_folder, "2-bgw/4-a-absorption/eigenvalues.dat")
        if os.path.exists(path):
            with open(path, 'r') as f:
                json_file["gwbse"]["bse_Es"] = float(f.readlines()[4].split()[0])
    
    if "bse_Et" not in json_file["gwbse"] or overwrite:
        path = os.path.join(root_folder, "2-bgw/4-triplet-absorption/eigenvalues.dat")
        if os.path.exists(path):
            with open(path, 'r') as f:
                json_file["gwbse"]["bse_Et"] = float(f.readlines()[4].split()[0])
    
    if "bse_DF" in json_file["gwbse"]:
        for key in ("bse_DF", "bse_Delta_st"):
            if key not in json_file["gwbse"]:
                print(f"Warning: {key} in but not in the json file. This typically does not happen. There may be some bugs in previous and/or this run")
    if "bse_DF" not in json_file["gwbse"] or overwrite:
        if "bse_Es" in json_file["gwbse"] and "bse_Et" in json_file["gwbse"]:
            json_file["gwbse"]["bse_DF"] = json_file["gwbse"]["bse_Es"] - 2 * json_file["gwbse"]["bse_Et"]
            json_file["gwbse"]["bse_Delta_st"] = json_file["gwbse"]["bse_Es"] - json_file["gwbse"]["bse_Et"]

    if "absorption" not in json_file["gwbse"]:
        json_file["gwbse"]["absorption"] = {}
    for prefix in ("a", "b", "c"):
        if prefix not in json_file["gwbse"]["absorption"] or overwrite:
            path = os.path.join(root_folder, f"2-bgw/4-{prefix}-absorption/absorption_eh.dat")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    json_file["gwbse"]["absorption"][prefix] = list(map(lambda x: list(map(float, x.split())), f.readlines()[4:]))
    
    if "fundamental_gap" in json_file["gwbse"]:
        if "bse_Es_bind" not in json_file["gwbse"] or overwrite:
            if "bse_Es" in json_file["gwbse"]:
                json_file["gwbse"]["bse_Es_bind"] = json_file["gwbse"]["fundamental_gap"] - json_file["gwbse"]["bse_Es"]
        if "bse_Et_bind" not in json_file["gwbse"] or overwrite:
            if "bse_Et" in json_file["gwbse"]:
                json_file["gwbse"]["bse_Et_bind"] = json_file["gwbse"]["fundamental_gap"] - json_file["gwbse"]["bse_Et"]

    return json_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing JSON file")
    parser.add_argument("--json_file", type=str, help="The path to the JSON file")

    parser.add_argument("--geometry_author", type=str, default=None, help="The author of relaxation")
    parser.add_argument("--relax_code", type=str, default=None, help="The relaxation code")
    parser.add_argument("--cif_file", type=str, default=None, help="The path to the CIF file")
    
    parser.add_argument("--gwbse_author", type=str, default=None, help="The author of the GW+BSE results")
    parser.add_argument("--gwbse_root_folder", type=str, default=None, help="The root folder of the GW+BSE results")

    args = parser.parse_args()

    id = os.path.basename(args.json_file).split('.')[0]
    print(f"Please make sure the CSD ID {id} is correct.")

    try:
        with open(args.json_file, 'r') as f:
            json_file = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {args.json_file}: {e}. Starting with an empty JSON file.")
        json_file = {}

    if "geometry" in json_file and "rmsd" in json_file["geometry"]:
        check_rmsd(json_file["geometry"]["rmsd"])

    json_file = write_id(id, json_file)
    json_file = write_relax(id, json_file, args.cif_file, args.geometry_author, args.relax_code, args.overwrite)
    json_file = write_dft(id, json_file)
    json_file = write_gwbse(id, json_file, args.gwbse_root_folder, args.gwbse_author, args.overwrite)

    with open(args.json_file, 'w') as f:
        json.dump(json_file, f, indent = 4)

if __name__ == "__main__":
    main()


