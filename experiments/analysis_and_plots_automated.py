import bacterial_septation.trajectory_analysis as bsa
import matplotlib.pyplot as plt
from pathlib import Path

import json, glob
import pandas as pd
import numpy as np
import random

from lib_littleguys import plotting, parameters, analysis_functions
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Arc

import numpy as np
from collections import defaultdict
from itertools import combinations

plotting.load_pretty_figure_setup()


import gzip
from collections import defaultdict
import numpy as np

import gzip

def read_lammps_timestep(f):
    """
    Reads one complete timestep from a LAMMPS dump file.

    Returns:
        (timestep, atoms) or (None, None) at EOF or on incomplete frame.

    atoms = list of (atom_id, mol, type, x, y, z)
    """
    try:
        # Find the next timestep header
        while True:
            line = f.readline()
            if not line:
                return None, None  # EOF
            if line.startswith("ITEM: TIMESTEP"):
                break

        line = f.readline()
        if not line:
            return None, None
        timestep = int(line.strip())

        f.readline()  # ITEM: NUMBER OF ATOMS
        line = f.readline()
        if not line:
            return None, None
        n_atoms = int(line.strip())

        f.readline()  # ITEM: BOX BOUNDS
        if not f.readline() or not f.readline() or not f.readline():
            return None, None

        f.readline()  # ITEM: ATOMS ...

        atoms = []
        for _ in range(n_atoms):
            line = f.readline()
            if not line:
                return None, None  # Incomplete frame
            parts = line.split()
            if len(parts) < 6:
                return None, None

            atom_id = int(float(parts[0]))
            mol = int(float(parts[1]))
            atype = int(parts[2])
            x, y, z = map(float, parts[3:6])

            if atom_id <= 2:
                continue

            atoms.append((atom_id, mol, atype, x, y, z))

        return timestep, atoms

    except (EOFError, ValueError, IndexError, OSError):
        # Handles incomplete gzip blocks or partial frames
        return None, None



    
    
def molecule_size_distribution(
    filename,
    atom_type_targets = [1,2,3],
    timestep_target=None
):
    """
    Computes molecule size distribution (number of atoms per molecule).

    Parameters
    ----------
    timestep_target : int or None
        If None -> process all timesteps
        If int  -> process only this timestep

    Returns
    -------
    dict:
        If timestep_target is None:
            sizes[timestep][mol_id] = n_atoms
        Else:
            sizes[mol_id] = n_atoms
    """

    sizes = {}
    dump_filename = Path(filename)

    open_func = gzip.open if dump_filename.suffix == ".gz" else open

    with open_func(dump_filename, 'rt') as f:
        while True:
            timestep, atoms = read_lammps_timestep(f)
            if timestep is None:
                break


            if timestep_target is not None and timestep != timestep_target:
                continue

            mol_counts = defaultdict(int)

            for _, mol, atype, _, _, _ in atoms:
                if atype in atom_type_targets:
                    mol_counts[mol] += 1
                    
            # Filter out molecules with 2 or fewer atoms
            mol_counts = {mol: count for mol, count in mol_counts.items() if count > 2}

            if timestep_target is not None:
                return dict(mol_counts)

            sizes[timestep] = dict(mol_counts)

    return sizes


import numpy as np
from itertools import combinations
from collections import defaultdict

def interacting_molecules_vs_time(
    filename,
    cutoff=0.3,
    target_type=6
):
    interacting_vs_time = {}
    cutoff2 = cutoff * cutoff
    
    dump_filename = Path(filename)
    open_func = gzip.open if dump_filename.suffix == ".gz" else open

    with open_func(dump_filename, 'rt') as f:
        while True:
            timestep, atoms = read_lammps_timestep(f)
            print("Processing timestep:", timestep)

            if timestep is None:
                break

            # RESET per timestep
            interacting_mols = set()

            # Group type-6 atoms by molecule
            mol_atoms = defaultdict(list)
            for _, mol, atype, x, y, z in atoms:
                if atype == target_type:
                    mol_atoms[mol].append(np.array([x, y, z]))

            for mol_i, mol_j in combinations(mol_atoms.keys(), 2):

                # Skip if either molecule already interacting
                if mol_i in interacting_mols or mol_j in interacting_mols:
                    continue

                atoms_i = mol_atoms[mol_i]
                atoms_j = mol_atoms[mol_j]

                for pos_i in atoms_i:
                    for pos_j in atoms_j:
                        if np.sum((pos_i - pos_j) ** 2) <= cutoff2:
                            interacting_mols.add(mol_i)
                            interacting_mols.add(mol_j)
                            break
                    else:
                        continue
                    break

            interacting_vs_time[timestep] = interacting_mols

    return interacting_vs_time



import gzip

def save_patch_neighbors_to_txt(dump_filename, output_filename):
    """
    Reads a LAMMPS dump file (plain text or .gz) with c_cPatchNeighbors and writes:
        timestep  mol_id
    for each unique mol where c_cPatchNeighbors == 1.
    """

    # Choose correct open function based on file extension
    dump_filename = Path(dump_filename)

    open_func = gzip.open if dump_filename.suffix == ".gz" else open

    with open_func(dump_filename, 'rt') as f, open(output_filename, 'w') as out:
        out.write("# timestep mol_id\n")
        try:
            while True:
                line = f.readline()
                if not line:
                    break

                if not line.startswith("ITEM: TIMESTEP"):
                    continue

                timestep = int(f.readline().strip())

                f.readline()  # ITEM: NUMBER OF ATOMS
                n_atoms = int(f.readline().strip())

                f.readline()  # ITEM: BOX BOUNDS
                f.readline()
                f.readline()
                f.readline()

                f.readline()  # ITEM: ATOMS ...

                mols_with_patch = set()

                for _ in range(n_atoms):
                    parts = f.readline().split()

                    atom_id = int(parts[0])
                    mol = int(parts[1])
                    atom_type = int(parts[2])
                    patch_neighbor1 = int(float(parts[6]))
                    #patch_neighbor2 = int(float(parts[7]))

                    if mol < 1:
                        continue

                    if patch_neighbor1 == 1 and atom_type == 6: #or patch_neighbor2 == 1:
                        mols_with_patch.add(mol)

                # write unique mol IDs for this timestep
                for mol in sorted(mols_with_patch):
                    out.write(f"{timestep} {mol}\n")
        except (EOFError, ValueError, IndexError, OSError):
        # Handles incomplete gzip blocks and partial frames
            print('File incomplete!')



import gzip

def save_molecule_count_per_timestep(dump_filename, output_filename):
    """
    Reads a LAMMPS dump trajectory (plain text or .gz) and writes:
        timestep  number_of_molecules
    Molecule ID 0 is excluded.
    """

    open_func = gzip.open if dump_filename.endswith('.gz') else open

    with open_func(dump_filename, 'rt') as f, open(output_filename, 'w') as out:
        out.write("# timestep n_molecules\n")
        try:
            while True:
                line = f.readline()
                if not line:
                    break

                if not line.startswith("ITEM: TIMESTEP"):
                    continue

                timestep = int(f.readline().strip())

                f.readline()  # ITEM: NUMBER OF ATOMS
                n_atoms = int(f.readline().strip())

                f.readline()  # ITEM: BOX BOUNDS
                f.readline()
                f.readline()
                f.readline()

                f.readline()  # ITEM: ATOMS ...

                mol_ids = set()

                for _ in range(n_atoms):
                    parts = f.readline().split()

                    atom_id = int(parts[0])
                    mol = int(parts[1])

                    if atom_id <= 2:
                        continue

                    if mol != 0:
                        mol_ids.add(mol)

                out.write(f"{timestep} {len(mol_ids)}\n")
        except (EOFError, ValueError, IndexError, OSError):
        # Handles incomplete gzip blocks and partial frames
            print('File incomplete!')


def save_molecule_size_distribution_txt(
    size_data,
    filename
):
    """
    Save molecule size distribution to a text file.

    Parameters
    ----------
    size_data : dict
        Output of molecule_size_distribution()
        Either:
          {mol_id: n_atoms}
        or:
          {timestep: {mol_id: n_atoms}}

    filename : str
        Output text file name
    """

    with open(filename, "w") as f:
        # Detect whether this is single-timestep or multi-timestep
        first_value = next(iter(size_data.values()))

        if isinstance(first_value, dict):
            # Multiple timesteps
            f.write("# timestep mol_id n_atoms\n")
            for timestep, mol_counts in size_data.items():
                for mol_id, n_atoms in mol_counts.items():
                    f.write(f"{timestep} {mol_id} {n_atoms}\n")
        else:
            # Single timestep
            f.write("# mol_id n_atoms\n")
            for mol_id, n_atoms in size_data.items():
                f.write(f"{mol_id} {n_atoms}\n")


def collect_sizes_from_txt(
    filename,
    start_timestep
):
    """
    Collect molecule sizes from a txt file for timesteps >= start_timestep.

    Returns
    -------
    np.ndarray of molecule sizes
    """
    data = np.loadtxt(filename)

    timesteps = data[:, 0].astype(int)
    sizes = data[:, 2].astype(int)
    
    #new_sizes = sizes[timesteps%1000000==0]
    #new_timesteps = timesteps[timesteps%1000000==0]

    return sizes[timesteps >= start_timestep ]

def pad_stack(arrays):
    max_len = max(len(a) for a in arrays)
    padded = [
        np.pad(np.asarray(a, dtype=float),
               (0, max_len - len(a)),
               constant_values=np.nan)
        for a in arrays
    ]
    X = np.vstack(padded)
    mean = np.nanmean(X, axis=0)
    std  = np.nanstd(X, axis=0, ddof=1)
    return mean, std



def load_condition(condition_dir, stiffness, bonds_filename="bonds.gz", bond_type=1):
    #print("Loading condition from:", condition_dir)
    seed_dirs = sorted(condition_dir.glob("stiffness_"+ str(stiffness)+"_seed_*"))
    #print("Found seed dirs:", seed_dirs)
    s_orders = []

    for seed_dir in seed_dirs:
        bonds_path = seed_dir / bonds_filename
        if bonds_path.exists():
            parser = bsa.BondsParser(bonds_path)
            s_orders.append(parser.nematic_order(bond_type=bond_type))
            #print(s_orders)
        else:
            print(f"⚠️ Missing file: {bonds_path}")

    return pad_stack(s_orders)

def load_condition_with_restart(condition_dir, stiffness, bonds_filename="bonds.gz", 
                   restart_filename="bonds_restart.gz", bond_type=1):
    """Load condition from main and restart bond files, connecting at equal timesteps."""
    seed_dirs = sorted(condition_dir.glob(f"stiffness_{stiffness}_seed_*"))
    s_orders = []

    for seed_dir in seed_dirs:
        bonds_path = seed_dir / bonds_filename
        restart_path = seed_dir / restart_filename
        
        if not bonds_path.exists():
            print(f"⚠️ Missing main file: {bonds_path}")
            continue
        
        # Parse main file
        parser = bsa.BondsParser(bonds_path)
        s_order_main = parser.nematic_order(bond_type=bond_type)
        timesteps_main = parser.get_timesteps()
        
        # Check if restart file exists
        if restart_path.exists():
            # Parse restart file
            parser_restart = bsa.BondsParser(restart_path)
            s_order_restart = parser_restart.nematic_order(bond_type=bond_type)
            timesteps_restart = parser_restart.get_timesteps()
            
            # Find the first index in restart that is after the last main timestep
            last_timestep_main = timesteps_main[-1]
            start_idx = 0
            
            # Find where restart timesteps become greater than last main timestep
            for i, ts in enumerate(timesteps_restart):
                if ts > last_timestep_main:
                    start_idx = i
                    break
            
            # If we found a start index, concatenate
            if start_idx < len(s_order_restart):
                print(f"Connecting main file (last timestep: {last_timestep_main}) "
                      f"to restart file (first new timestep: {timesteps_restart[start_idx]})")
                s_order_combined = np.concatenate([s_order_main, s_order_restart[start_idx:]])
            else:
                # No new data, just use main
                s_order_combined = s_order_main
        else:
            s_order_combined = s_order_main
        
        s_orders.append(s_order_combined)
    
    return pad_stack(s_orders) if s_orders else np.array([])



#main_dir = Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence")

# map folder names to plot labels

#analysis one patch square box
#conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_rigidity_influence/patch_interaction"): "patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_rigidity_influence/no_patch_interaction"): "no patch ",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_rigidity_influence/patch_interaction_modulate_pol_and_depol"): "Modulate kinetics"
#}

#analysis one patch square box
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_rigidity_influence/patch_interaction"): "Patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_rigidity_influence/no_patch_interaction"): "No patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_rigidity_influence/patch_interaction_modulate_pol_and_depol"): "Modulate kinetics",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_square_infinite_growth/no_patch_interaction"): "Infinite growth no patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_square_ferdinandsetup_infinite_growth/patch_interaction"): "Infinite growth + patch"
# }

#analysis one patch treadmilling Chris' profile
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_my_treadmilling_profile_infinite_growth/patch_interaction"): "Patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_my_treadmilling_profile_infinite_growth/no_patch_interaction"): "No patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_my_treadmilling_profile_infinite_growth/patch_interaction_modulate_pol_and_depol"): "Modulate kinetics"
# }

#2 patches
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/patch_attraction"): "Both patches",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/no_patch_attraction"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/patch_attraction_modulate_pol_and_depol"): "Modulate kinetics"
# }

#2 patches with infinite growth
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/patch_attraction"): "Both patches",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/no_patch_attraction"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/patch_attraction_modulate_pol_and_depol"): "Modulate kinetics",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_square_infinite_growth/no_patch_attraction"): "Infinite growth no patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_square_infinite_growth/patch_attraction"): "Infinite growth + patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_square_infinite_growth/patch_attraction_modulate_pol_and_depol"): "Infinite growth + modulated kinetics"
# }

#2 patches vol excl
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/patch_attraction_volume_exclusion_samesizebox"): "Both patches",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/no_patch_attraction_volume_exclusion_samesizebox"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_rigidity_influence/patch_attraction_modulate_pol_and_depol_volume_exclusion_samesizebox"): "Modulate kinetics"
# }

#2 patches vol excl
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_square_infinite_growth/patch_attraction"): "Both patches",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_square_infinite_growth/no_patch_attraction"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_square_infinite_growth/patch_attraction_modulate_pol_and_depol"): "Modulate kinetics",
# }

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NEW RESULTS - HIGHER DIFFUSION + CHRIS PARAM RING

# main_dir = '/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_treadmilling_alignment_chris_params'

# #CHRIS PARAM RING - 1 patch
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_treadmilling_alignment_chris_params/patch_interaction"): "Both patches",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_treadmilling_alignment_chris_params/no_patch_interaction"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_treadmilling_alignment_chris_params/patch_interaction_modulate_pol_depol"): "Modulate kinetics",
# }

#CHRIS PARAM RING - 1 patch, higher diffusion
# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_treadmilling_alignment_chris_params_larger_diffusion/patch_interaction"): "Both patches",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_treadmilling_alignment_chris_params_larger_diffusion/no_patch_interaction"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_treadmilling_alignment_chris_params_larger_diffusion/patch_interaction_modulate_pol_depol"): "Modulate kinetics",
# }

#SQUARE - 1 patch, higher diffusion, higher nucleation - RUN JUST HIGHER NUCLEATION, MAYBE DENSITY IS ENOUGH
# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_squarebox_volexcl_higher_nucleation_rate_D_10')

# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_squarebox_volexcl_higher_nucleation_rate_D_10/patch_attraction"): "Patch",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_squarebox_volexcl_higher_nucleation_rate_D_10/no_patch_attraction"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_squarebox_volexcl_higher_nucleation_rate_D_10/patch_attraction_modulate_pol_and_depol"): "Modulate kinetics",
# }

# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/square_D_10_modulatethyd_faster')

# conditions = {
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/square_D_10_modulatethyd_faster/no_patch_attraction'): "Faster hydrolysis decrease",
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/square_D_10_modulatethyd/no_patch_attraction'): "Slower hydrolysis decrease",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_squarebox_volexcl_higher_nucleation_rate_D_10/no_patch_attraction"): "No change"
# }

##### REPULSION

# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/square_volexcl_D_10_L_400_noalignment_repulsion')

# conditions = {
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/task2_squarebox_volexcl_higher_nucleation_rate_D_10/no_patch_attraction'): "No interaction",
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/square_volexcl_D_10_L_400_noalignment_repulsion/patch_attraction'): "Patch - REPULSIVE",
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/square_volexcl_D_10_L_400_noalignment_repulsion/patch_attraction_modulate_pol_and_depol'): "Modulate kinetics"
    
# }

##1 patch - no pink patch

# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_squarebox_larger_nucleation_rate_D_10')

# conditions = {
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_squarebox_larger_nucleation_rate_D_10/no_patch_interaction'): "No interaction",
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_squarebox_larger_nucleation_rate_D_10/patch_interaction'): "Patch",
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_squarebox_larger_nucleation_rate_D_10/patch_interaction_modulate_pol_and_depol'): "Modulate kinetics"
    
# }

##antiparallel fils D=1, 

main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/antiparallel_filaments')

conditions = {
    Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/antiparallel_filaments/modulation_inf_growth'): "Modulate infinite growth",
    Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/antiparallel_filaments/no_modulation_inf_growth'): "No modulation infinite growth"
}

##crosslinked fils D=1, 

# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/crosslinked_filaments')

# conditions = {
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/crosslinked_filaments/modulation_inf_growth'): "Modulate infinite growth",
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/crosslinked_filaments/no_modulation_inf_growth'): "No modulation infinite growth"
# }

##parallel fils D=1, 

# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/parallel_filaments')

# conditions = {
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/parallel_filaments/modulation_inf_growth'): "Modulate infinite growth",
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_D_1_dist0.9/parallel_filaments/no_modulation_inf_growth'): "No modulation infinite growth"
# }


#CHRIS PARAM RING - 2 patches

# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_volexcl_D_10_L_600_infgrowth')

# conditions = {
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_volexcl_D_10_L_600_infgrowth/no_patch_attraction"): "No interaction",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_volexcl_D_10_L_600_infgrowth/patch_attraction"): "Patches",
#     Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_volexcl_D_10_L_600_infgrowth/patch_attraction_modulate_pol_and_depol"): "Modulate kinetics",
# }

# main_dir = Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_volexcl_D_5_L_600_noinfgrowth')

# conditions = {
#     Path('/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filament_2_patches_higher_volexcl/ring_volexcl_D_5_L_600_noinfgrowth/patch_attraction_modulate_pol_and_depol'): "Modulate kinetics"
# }

def get_first_timestep(bonds_path):
    """Extract the first timestep number from a bonds file."""
    import gzip
    import re
    
    TIMESTEP_RE = r"ITEM: TIMESTEP\s+(\d+)"
    
    try:
        with gzip.open(bonds_path, 'rt') as f:
            # Read only the first few lines to find the first timestep
            content = f.read(10000)  # Read first 10k chars which should contain first timestep
            timesteps = re.findall(TIMESTEP_RE, content)
            if timesteps:
                return int(timesteps[0])
    except Exception as e:
        print(f"Warning: Could not read timestep from {bonds_path}: {e}")
    
    return 0  # Default to 0 if can't read

results = {}
stiffness = 1000


# for folder_name, label in conditions.items():
#     condition_dir = folder_name
#     mean, std = load_condition_with_restart(condition_dir, stiffness, bonds_filename="bonds.gz",restart_filename="bonds_restart.gz",bond_type=1)
#     results[label] = (mean, std)


# plt.figure(figsize=(5.4, 4))

# for label, (mean, std) in results.items():
#     t = np.arange(1, 1 + len(mean))
#     plt.plot(t, mean, lw=3, label=label)
#     plt.fill_between(t, mean - std, mean + std, alpha=0.3)

# plt.legend()
# plt.ylim(0, 1)
# plt.xlabel("Time")
# plt.ylabel("Nematic order")
# plt.tight_layout()
# plt.savefig(main_dir / f"nematic_order_over_time_all_conditions_with_restart_mean_{stiffness}.pdf",
#             transparent=True)
# plt.show()


#plot density 

def load_condition_density(condition_dir,stiffness,volume=None, trajectory_filename="trajectory.gz", target_types=[1,2,3]):
    seed_dirs = sorted(condition_dir.glob("stiffness_"+ str(stiffness)+"_seed_*"))
    print(f"\n🔍 Scanning: {condition_dir}")
    print(f"Found {len(seed_dirs)} seed folders")

    densities = []
    #volume = None

    for seed_dir in seed_dirs:
        traj_path = seed_dir / trajectory_filename
        if traj_path.exists():
            print(f"  ✔ Using {traj_path}")
            parser = bsa.TrajectoryParser(traj_path)

            if volume is None:
                box_bounds = parser.get_box()
                lower_bounds, upper_bounds = box_bounds[0]
                length = upper_bounds - lower_bounds
                volume = length**2   # assuming 2D

            density = parser.get_natoms(target_types=target_types) / volume
            densities.append(density)
        else:
            print(f"  ⚠️ Missing file: {traj_path}")

    if len(densities) == 0:
        raise RuntimeError(f"No trajectory files found in {condition_dir}")

    return pad_stack(densities)
"""
main_dir = Path("/nfs/scistore26/saricgrp/mkrstic/felix_treadmilling/experiments/double_filaments/task1_rigidity_influence")

# map folder names to plot labels
conditions = {
    "patch_interaction": "patch",
    "no_patch_interaction": "no patch ",
    "patch_interaction_modulate_pol_and_depol": "Kinetics modulation",
}

results = {}
stiffness = 1000
"""

#volume = 200*100
#volume = 200*400

# for folder_name, label in conditions.items():
#     condition_dir = folder_name
#     mean, std = load_condition_density(condition_dir, stiffness, volume=None,trajectory_filename="trajectory.gz", target_types=[1,2,3])
#     results[label] = (mean, std)

# plt.figure(figsize=(5.4, 4))

# for label, (mean, std) in results.items():
#     t = np.arange(1, len(mean) + 1)
#     plt.plot(t, mean, lw=3, label=label)
#     plt.fill_between(t, mean - std, mean + std, alpha=0.3)

# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Surface density")
# plt.tight_layout()
# plt.savefig(main_dir / f"density_over_time_all_conditions_mean_{stiffness}.pdf", transparent=True)
#plt.show()

#get number of molecules interacting through the patch

def generate_patch_neighbors(condition_dir,
                             stiffness,
                             traj_filename="trajectory.gz",
                             output_filename="patch_neighbors.txt"):
    seed_dirs = sorted(condition_dir.glob("stiffness_"+str(stiffness)+"_seed_*"))
    print('FOUND DIRECTORIES: ' + str(seed_dirs))
    print(f"\n🔧 Generating patch neighbors in: {condition_dir}")

    for seed_dir in seed_dirs:
        traj_path = seed_dir / traj_filename
        out_path = seed_dir / output_filename

        if traj_path.exists():
            print(f"  ✔ Processing {traj_path}")
            save_patch_neighbors_to_txt(traj_path, out_path)
        else:
            print(f"  ⚠️ Missing trajectory file: {traj_path}")


def load_condition_patch_neighbors(condition_dir,
                                   stiffness,
                                   neighbors_pattern="patch_neighbors*.txt*",
                                   dump_interval=100000):
    seed_dirs = sorted(condition_dir.glob("stiffness_" + str(stiffness) + "_seed_*"))
    print(f"\n📊 Loading patch neighbors from: {condition_dir}")

    counts_per_seed = []
    time_per_seed = []

    for seed_dir in seed_dirs:
        neighbor_files = list(seed_dir.glob(neighbors_pattern))
        if not neighbor_files:
            print(f"  ⚠️ No patch neighbor file in {seed_dir}")
            continue

        neighbors_path = neighbor_files[0]
        print(f"  ✔ Using {neighbors_path}")

        data = np.loadtxt(neighbors_path, comments="#")  # works for .gz too
        timesteps, mols = data[:, 0], data[:, 1]

        unique_ts, counts = np.unique(timesteps, return_counts=True)
        unique_ts = unique_ts / dump_interval

        time_per_seed.append(unique_ts)
        counts_per_seed.append(counts)

    if len(counts_per_seed) == 0:
        raise RuntimeError(f"No patch neighbor files found in {condition_dir}")
    
    mean, std = pad_stack(counts_per_seed)

    # Build a matching time vector (in dump units)
    max_len = max(len(t) for t in time_per_seed)
    t = np.arange(max_len)

    return mean, std, t
"""
main_dir = Path("/path/to/main_folder")

conditions = {
    "slower_depol": "Slower depol",
    "modulate_pol_depol": "Modulate pol and depol",
    # add more if needed
}
"""


# Step 4a: Generate neighbor files
# for folder in conditions.keys():
#     generate_patch_neighbors(folder, stiffness)


# # Step 4b: Load + plot
# plt.figure(figsize=(5.4, 4))

# for folder, label in conditions.items():
#     mean, std, t = load_condition_patch_neighbors(folder, stiffness)
#     plt.plot(t, mean, lw=3, label=label)
#     plt.fill_between(t, mean - std, mean + std, alpha=0.3)

# plt.xlabel("Timestep")
# plt.ylabel("No. of antiprallel filaments")
# plt.legend()
# plt.tight_layout()
# plt.savefig(main_dir / f"patch_neighbors_over_time_all_conditions_mean_{stiffness}.pdf",
#             transparent=True)
#plt.show()

#get percent of interacting molecules

def normalize_counts(unique_ts, counts, mol_count_dict):
    norm_counts = np.zeros_like(counts, dtype=float)
    for i, ts in enumerate(unique_ts):
        nmol = mol_count_dict.get(ts, 0)
        norm_counts[i] = 100 * counts[i] / nmol if nmol > 0 else 0.0
    return norm_counts

def load_seed_percentage(seed_dir,
                         neighbors_pattern="patch_neighbors*.txt*",
                         mol_count_pattern="num_of_filaments*.txt*",
                         dump_interval=100000):
    neighbor_files = list(seed_dir.glob(neighbors_pattern))
    mol_files = list(seed_dir.glob(mol_count_pattern))

    if not neighbor_files or not mol_files:
        return None, None

    neighbors_path = neighbor_files[0]
    mol_path = mol_files[0]

    data = np.loadtxt(neighbors_path, comments="#")
    timesteps, mols = data[:, 0], data[:, 1]
    unique_ts, counts = np.unique(timesteps, return_counts=True)

    mol_data = np.loadtxt(mol_path, comments="#")
    mol_timesteps = mol_data[:, 0]
    n_molecules = mol_data[:, 1]
    mol_count_dict = dict(zip(mol_timesteps, n_molecules))

    norm_counts = normalize_counts(unique_ts, counts, mol_count_dict)
    unique_ts = unique_ts / dump_interval

    return unique_ts, norm_counts

def load_condition_percentage(condition_dir,
                              stiffness,
                              neighbors_pattern="patch_neighbors*.txt*",
                              mol_count_pattern="num_of_filaments*.txt*",
                              dump_interval=100000):
    seed_dirs = sorted(condition_dir.glob("stiffness_"+str(stiffness)+"_seed_*"))
    print(f"\n📊 Processing percentage interactions in: {condition_dir}")

    norm_counts_per_seed = []
    time_per_seed = []

    for seed_dir in seed_dirs:
        t, norm_counts = load_seed_percentage(
            seed_dir,
            neighbors_pattern=neighbors_pattern,
            mol_count_pattern=mol_count_pattern,
            dump_interval=dump_interval
        )

        if t is not None:
            print(f"  ✔ Using {seed_dir}")
            time_per_seed.append(t)
            norm_counts_per_seed.append(norm_counts)
        else:
            print(f"  ⚠️ Missing files in {seed_dir}")

    if len(norm_counts_per_seed) == 0:
        raise RuntimeError(f"No valid percentage data in {condition_dir}")

    mean, std = pad_stack(norm_counts_per_seed)
    # Build a matching time vector (in dump units)
    max_len = max(len(t) for t in time_per_seed)
    t = np.arange(max_len)

    return mean, std, t

"""
results = {}

for folder, label in conditions.items():
    mean, std, t = load_condition_percentage(main_dir / folder,stiffness)
    results[label] = (mean, std, t)

plt.figure(figsize=(5.4, 4))

for label, (mean, std, t) in results.items():
    plt.plot(t, mean, lw=3, label=label)
    plt.fill_between(t, mean - std, mean + std, alpha=0.3)

plt.xlabel("Timestep")
plt.ylabel("Percentage of filaments interacting laterally")
plt.legend()
plt.tight_layout()
plt.savefig(main_dir / f"patch_neighbors_over_time_all_conditions_percentage_mean_{stiffness}.pdf",
            transparent=True)
plt.show()
"""
###########################

#get length distribution

def generate_size_distributions(condition_dir,
                                stiffness,
                                traj_pattern="trajectory.gz",
                                output_pattern="molecule_sizes_vs_time.txt"):
    seed_dirs = sorted(condition_dir.glob("stiffness_" + str(stiffness) + "_seed_*"))
    print(f"\n🔧 Generating molecule size distributions in: {condition_dir}")

    for seed_dir in seed_dirs:
        traj_files = list(seed_dir.glob(traj_pattern))
        if not traj_files:
            print(f"  ⚠️ No trajectory file in {seed_dir}")
            continue

        traj_path = traj_files[0]
        out_path = seed_dir / output_pattern

        print(f"  ✔ Processing {traj_path}")
        sizes_all = molecule_size_distribution(traj_path)
        save_molecule_size_distribution_txt(sizes_all, out_path)


def collect_sizes_from_txt(filename, start_timestep):
    data = np.loadtxt(filename)
    timesteps = data[:, 0].astype(int)
    sizes = data[:, 2].astype(int)
    return sizes[timesteps>= start_timestep]

def load_condition_sizes(condition_dir,
                         stiffness,
                          size_pattern="molecule_sizes_vs_time.txt",
                          start_timestep=0):
    seed_dirs = sorted(condition_dir.glob("stiffness_" + str(stiffness) + "_seed_*"))
    print(f"\n📊 Collecting molecule sizes in: {condition_dir}")

    all_sizes = []

    for seed_dir in seed_dirs:
        size_files = list(seed_dir.glob(size_pattern))
        if not size_files:
            print(f"  ⚠️ No size file in {seed_dir}")
            continue

        size_path = size_files[0]
        print(f"  ✔ Using {size_path}")
        sizes = collect_sizes_from_txt(size_path, start_timestep=start_timestep)
        all_sizes.append(sizes)

    if len(all_sizes) == 0:
        raise RuntimeError(f"No size distributions found in {condition_dir}")

    return np.concatenate(all_sizes)

def plot_size_distribution_comparison(results_dict,
                                      bins=20,
                                      max_size=None,
                                      filename="molecule_size_distribution_comparison.pdf"):
    plt.figure(figsize=(5.4, 4))

    for label, sizes in results_dict.items():
        if max_size is not None:
            sizes = sizes[sizes < max_size]
        plt.hist(sizes, bins=bins, density=True, alpha=0.5, label=label)

    plt.xlabel("Filament length")
    plt.ylabel("Probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(main_dir / filename, transparent=True)
    plt.show()



# Step 1 — Generate size distributions

# for folder in conditions.keys():
#     generate_size_distributions(folder, stiffness)

# Step 2 — Load and collect sizes after a given timestep
start_timestep = 0 * 100000  # e.g. 500 dumps

results = {}
for folder, label in conditions.items():
    sizes = load_condition_sizes(
        folder,
        stiffness,
        size_pattern="molecule_sizes_vs_time*.txt*",
        start_timestep=start_timestep
    )
    results[label] = sizes

# Step 3 — Plot all conditions together
plot_size_distribution_comparison(
    results,
    bins=20,
    filename=main_dir / f"molecule_size_distribution_all_conditions_before_{start_timestep}.pdf"
)


##############################################
#get number of filaments per timestep

def load_condition_filament_counts(condition_dir,
                                   stiffness,
                                   neighbors_pattern="molecule_sizes_vs_time*.txt*",
                                   dump_interval=100000):
    seed_dirs = sorted(condition_dir.glob(f"stiffness_{stiffness}_seed_*"))
    print(f"\n📊 Loading filament counts from: {condition_dir}")

    counts_per_seed = []
    time_per_seed = []

    for seed_dir in seed_dirs:
        neighbor_files = list(seed_dir.glob(neighbors_pattern))
        if not neighbor_files:
            print(f"  ⚠️ No patch neighbor file in {seed_dir}")
            continue

        neighbors_path = neighbor_files[0]
        print(f"  ✔ Using {neighbors_path}")

        data = np.loadtxt(neighbors_path, comments="#")
        timesteps = data[:, 0]

        unique_ts, counts = np.unique(timesteps, return_counts=True)
        unique_ts = unique_ts / dump_interval

        time_per_seed.append(unique_ts)
        counts_per_seed.append(counts)

    if not counts_per_seed:
        raise RuntimeError(f"No patch neighbor files found in {condition_dir}")

    mean, std = pad_stack(counts_per_seed)

    max_len = max(len(t) for t in time_per_seed)
    t = np.arange(max_len)  # or * dump_interval if you prefer physical time

    return mean, std, t


def ensemble_mean_filament_counts(condition_dir,
                                   stiffness,
                                   neighbors_pattern="molecule_sizes_vs_time*.txt*",
                                   dump_interval=100000):
    seed_dirs = sorted(condition_dir.glob(f"stiffness_{stiffness}_seed_*"))

    counts_per_seed = []
    time_per_seed = []

    for seed_dir in seed_dirs:
        try:
            t, counts = load_filament_counts_per_seed(
                seed_dir,
                pattern=neighbors_pattern,
                dump_interval=dump_interval
            )
            counts_per_seed.append(counts)
            time_per_seed.append(t)
        except FileNotFoundError:
            print(f"⚠️ No file in {seed_dir}")

    if not counts_per_seed:
        raise RuntimeError("No valid seed files found.")

    mean, std = pad_stack(counts_per_seed)

    max_len = max(len(t) for t in time_per_seed)
    t = np.arange(max_len) #* dump_interval

    return mean, std, t



# results = {}

# for folder, label in conditions.items():
#     mean, std, t = load_condition_filament_counts(folder, stiffness)
#     results[label] = (mean, std, t)

# plt.figure(figsize=(5.4, 4))

# for label, (mean, std, t) in results.items():
#     plt.plot(t, mean, lw=3, label=label)
#     plt.fill_between(t, mean - std, mean + std, alpha=0.3)

# plt.xlabel("Timestep")
# plt.ylabel("Number of filaments")
# plt.legend()
# plt.tight_layout()
# plt.savefig(main_dir / f"filament_count_over_time_all_conditions_mean_{stiffness}.pdf",
#             transparent=True)
#plt.show()



