from pathlib import Path
from itertools import product
import json
import math

import re
import glob
import os

import jinja2
import numpy as np
import shutil
import subprocess

from bacterial_septation.structure_init import StructureInitialiser, BADUtils, DoubleBADUtils, _ESCRTUtils, DoubleBADTwoPatchesUtils
from bacterial_septation.chemistry_writer import ChemistryWriter
from bacterial_septation.lammps_blocks import (
    TreadmillingBlock, SystemDumpBlock, BADFilamentBlock, TreadmillingBlockVolumeExclusion,
    DoubleBADFilamentBlock, BADFilamentDumpBlock, ESCRTFilamentBlock, DoubleBADDoubleFilamentBlock, DoubleBADDoubleFilamentTwoPatchesBlock, BADFilamentDumpBlockConstriction
)
from bacterial_septation.parameters import (
    SimulationParameters, TreadmillingParameters,
    MembraneParameters, DoubleBADFilamentParameters, ESCRTFilamentParameters, DoubleBADTwoPatchesFilamentParameters
)

def get_max_restart_number(directory='.'):
    """
    Find the maximum integer number from restart files matching pattern:
    restart.Integer_number.bin.gz
    """
    pattern = os.path.join(directory, 'restart.*.bin.gz')
    files = glob.glob(pattern)
    
    max_number = None
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract integer from filename using regex
        match = re.search(r'restart\.(\d+)\.bin\.gz', filename)
        if match:
            number = int(match.group(1))
            if max_number is None or number > max_number:
                max_number = number
    
    return max_number

def main(dirpath, real_system_size, r_on, tau_hyd, r_off, r_off_modulation, seed, patch_distance,new_patch_distance, patch1_angle_rigidity, patch2_angle_rigidity, filament_stiffness, are_patches_interacting, add_pol_modulation, add_depol_modulation, r_on_modulation, tau_hyd_modulation, infinite_growth, add_min_treadmilling_profile,add_volume_exclusion,add_alignment_force,parallel_filament, parallel_and_antiparallel, antiparallel_filament,antiparallel_filaments_modulate_both_patches,repulsive_interaction_antiparallel_filaments, runtime_mins, constriction, hydrolysis_mode, reservoir, secnuc, prob_secnuc,prob_changetype, r_secnuc, max_restart_number, restart, stacking, prob_stacking, prob_changetype_stacking, r_stacking, vary_nucleation, max_atoms):
    """
    dirpath = Path(__file__).parent.resolve()
    
    real_system_size = 2000 # nm #was 2000 for other sims
    r_on = 8
    tau_hyd = 5 #was 1
    r_off = 10 #was 3
    seed = 1234
    patch_distance = 0.5
    new_patch_distance = 1.1
    patch1_angle_rigidity = 5e5
    patch2_angle_rigidity = 5e5
    filament_stiffness = 1e3

    # write templates for modulated polymerization/depolymerization
    are_patches_interacting = True
    add_pol_modulation = False
    add_depol_modulation = False
    r_on_modulation = 9.5
    tau_hyd_modulation = 10
    add_min_treadmilling_profile = False
    
    infinite_growth = False
    if add_pol_modulation:
        if r_on_modulation > r_off:
            infinite_growth = True
    """
    # define units based ftsz size
    # sizes based on Shi et al. (KC Huang). “Chiral Twisting in a Bacterial Cytoskeletal Polymer Affects Filament Size and Orientation.”
    # https://doi.org/10.1038/s41467-020-14752-9
    real_ftsz_size = 5 # nm
    simulated_ftsz_diameter = 1 # ftsz diameters

    # define a cylinder that is approximately the size of a 
    system_size = real_system_size / real_ftsz_size * simulated_ftsz_diameter # in ftsz units

    #50% of the original system length for intinite growth box size
    infinite_growth_box_size = system_size * 0.8
    
    treadmilling_profile_width = system_size * 0.5 #20% of the system size 
    
    # calculate nucleation rate to conform to desired rate per area - this worked for mean-field
    if hydrolysis_mode == 'mean-field':
        nucleation_rate_per_area = 1/40000 #1 for a box of L=200
        nucleation_area = system_size * system_size
        nucleation_rate = nucleation_rate_per_area * nucleation_area
    elif hydrolysis_mode == 'explicit':
        nucleation_rate = 100 # this is for explicit hydrolysis, higher nuc was 15, std is 5
    
    tau_per_sec = 20 #with this setup the max roff=10, was 10, works with 20 for D=1, 50 for D=5, 80 for D=10
    simulation_parameters = SimulationParameters(
        timestep=1e-4, total_time=runtime_mins*60*tau_per_sec, # 20 minutes
        reaction_time=0.05*tau_per_sec, print_time=1*tau_per_sec, # print every second - changed reaction time so i can use rates >10
        nucleation_time=(1./nucleation_rate)*tau_per_sec, 
        sigma = real_ftsz_size/simulated_ftsz_diameter,
        diffusivity=2, #was 1
        tau_per_sec=tau_per_sec,
        seed=int(seed),
        hydrolysis_mode=hydrolysis_mode, #or 'mean_field'
        max_restart_number=max_restart_number
    )
    
    simulation_parameters.set_diffusivity_nms(1000)  # WAS 500 # nm^2/s, this should be what chris calls 100nm^2/s, set to 2500 cause tau_per_sec is increased! increase diffusivity to be 5-> to 5000

    treadmilling_parameters = TreadmillingParameters(
        r_on=r_on/tau_per_sec,
        tau_hyd=tau_hyd*tau_per_sec,
        r_off=r_off/tau_per_sec,
        r_off_modulation = r_off_modulation/tau_per_sec,
        r_nuc=nucleation_rate/tau_per_sec,
        template_path=dirpath / 'chemistry_templates',
        add_pol_modulation=add_pol_modulation,
        add_depol_modulation=add_depol_modulation,
        add_sec_nuc_modulation=secnuc,
        prob_secnuc=prob_secnuc, #probability of secondary nucleation, 0.1 is a good value
        prob_changetype=prob_changetype, #probability of changing type of filament, 0.1 is a good value
        r_sec_nuc=r_secnuc/tau_per_sec, #rate of secondary nucleation
        add_stacking_modulation=stacking,
        prob_stacking=prob_stacking, #probability of stacking, 0.1 is a good value
        prob_changetype_stacking=prob_changetype_stacking, #probability of changing type of filament, 0.1 is a good value
        r_stacking=r_stacking/tau_per_sec, #rate of stacking
        r_on_modulation= r_on_modulation/tau_per_sec, # increase kon by 3 units
        tau_hyd_modulation= tau_hyd_modulation*tau_per_sec, # increase tau_hyd by
        infinite_growth=infinite_growth,
        infinite_growth_box_size = infinite_growth_box_size,
        add_min_treadmilling_profile=add_min_treadmilling_profile,
        treadmilling_profile_width=treadmilling_profile_width,
        are_patches_interacting=are_patches_interacting,
        add_two_oligomer_templates = True,
        add_alignment_force = add_alignment_force,
        reservoir=reservoir,
        vary_nucleation=vary_nucleation,
        max_atoms=max_atoms
    )

    structure_initialiser = StructureInitialiser()
    structure_initialiser.box = np.array([
        [-system_size/2, system_size/2],
        [-system_size/2, system_size/2],
        [-system_size/2, system_size/2]
    ])
    
    #making a stright filament with a patch that is patch_distance away in the y axis
    main_beads = BADUtils.get_centered_line(20, r_zero=simulated_ftsz_diameter)
    main_beads[:, 1] += 4
    connecting_patches = main_beads.copy()
    connecting_patches[:, 1] += patch_distance
    new_patches = main_beads.copy()
    new_patches[:, 1] -= new_patch_distance

    equilibrium_helix = np.zeros((len(main_beads)*3, 3))
    equilibrium_helix[0::3, :] = main_beads
    equilibrium_helix[1::3, :] = connecting_patches
    equilibrium_helix[2::3, :] = new_patches

    #this adds a filament of length 20 to the initial state file, so commenting out to have an initial file with only the seed
    #structure_initialiser.add_as_double_filament_with_two_patches(equilibrium_helix)
    structure_initialiser.add_seed_dimer(seed_type=12) #adding patches of types 9,10,11

    # get equilibrium properties of this helix
    bad_helix = equilibrium_helix[::3].copy()

    distance_dict, angle = DoubleBADTwoPatchesUtils.get_helix_properties(equilibrium_helix, 1, verbose=True)
    print("Double Filament 2 Patches data: ", distance_dict, angle)
    
    filament_parameters = DoubleBADTwoPatchesFilamentParameters(
        mass=1,
        bond_distance=simulated_ftsz_diameter, bond_rigidity=5e4,
        equilibrium_angle=180, angle_rigidity=0.,
        equilibrium_dihedral=0, dihedral_rigidity=0,
        nn_angle_rigidity=filament_stiffness, nn_equilibrium_angle=180,
        repulsion_energy=1, repulsion_distance=simulated_ftsz_diameter,
        patch_distance=patch_distance, patch_rigidity=5e4,
        patch_equilibrium_angle=angle, patch_angle_rigidity=patch1_angle_rigidity,
        patch_equilibrium_dihedral = 0, patch_dihedral_rigidity=0.,
        #interpatch_bond_distance=distance_dict[4], interpatch_bond_rigidity=0.,
        interpatch_bond_distance=0, interpatch_bond_rigidity=0.,
        patch2_distance=new_patch_distance, patch2_rigidity=5e4,
        patch2_equilibrium_angle=angle, patch2_angle_rigidity=patch2_angle_rigidity,
        #interpatch2_bond_distance=distance_dict[6], interpatch2_bond_rigidity=0.
        interpatch2_bond_distance=0, interpatch2_bond_rigidity=0.
    )
    
    
    # print to input file
    if not constriction:
        save_path = dirpath.joinpath('initial_state.lmpdat')
        structure_initialiser.write_to_file(
            filename=save_path,
            write_bonds=True, write_angles=True, write_dihedrals=False,
            n_atom_types=13, n_bond_types=6, n_angle_types=4, n_dihedral_types=None
        ) #I'm setting n_atom_types, n_bond_types and n_angle_types since the structure initaliser only has the seed, if there is a filament there (for testing purposes) they can be set to None
    
    
        parameter_file = dirpath.joinpath('parameters.json')
        with open(parameter_file, 'w') as f:
            json.dump(
                {
                    'simulation_parameters': simulation_parameters.__dict__,
                    'filament_parameters': filament_parameters.__dict__,
                    'treadmilling_parameters': treadmilling_parameters.__dict__,
                }, 
                f, indent=4
            )

        # write chemistry
        chemistry_dir = Path(treadmilling_parameters.template_path)
        chemistry_dir.mkdir(exist_ok=True)
        
    chemistry_writer = ChemistryWriter(
        equilibrium_config=equilibrium_helix,
        peratom_masses=False,
        mode='DoubleFilamentsTwoPatches',
        seed_mode='dimer',
        seed_type=12,
        patch1_distance=patch_distance,
        patch2_distance=new_patch_distance,
        step_helix=1
    )
    
    if not constriction and not restart:
        if simulation_parameters.hydrolysis_mode == 'explicit':
            explicit_hydrolysis = True
        else:
            explicit_hydrolysis = False

        #if parallel_filament:
        chemistry_writer.write_oligomer_treadmilling_two_templates(target_dir=chemistry_dir, add_pol_modulation=add_pol_modulation, add_depol_modulation=add_depol_modulation, write_two_oligomer_files=True, parallel_filament=parallel_filament, parallel_and_antiparallel=False, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches,thyd_premod=tau_hyd*tau_per_sec, thyd_postmod=tau_hyd_modulation*tau_per_sec, kon_premod=r_on/tau_per_sec, kon_postmod=r_on_modulation/tau_per_sec, reaction_time=simulation_parameters.reaction_time, add_volume_exclusion=add_volume_exclusion, r_off = r_off/tau_per_sec,r_off_modulation=r_off_modulation/tau_per_sec, explicit_hydrolysis=explicit_hydrolysis)
        chemistry_writer.write_nmer_chemistry(chemistry_dir, add_pol_modulation=add_pol_modulation, add_depol_modulation=add_depol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=False, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, thyd_premod=tau_hyd*tau_per_sec, thyd_postmod=tau_hyd_modulation*tau_per_sec, kon_premod=r_on/tau_per_sec, kon_postmod=r_on_modulation/tau_per_sec, reaction_time=simulation_parameters.reaction_time, add_volume_exclusion=add_volume_exclusion, r_off = r_off/tau_per_sec,r_off_modulation=r_off_modulation/tau_per_sec, explicit_hydrolysis=explicit_hydrolysis)
        if secnuc:
            chemistry_writer.write_secnuc_templates(chemistry_dir)
        
        # else:
        #     chemistry_writer.write_oligomer_treadmilling_two_templates(target_dir=chemistry_dir, add_pol_modulation=add_pol_modulation, add_depol_modulation=add_depol_modulation,add_volume_exclusion=add_volume_exclusion)
        #     chemistry_writer.write_nmer_chemistry(chemistry_dir, add_pol_modulation=add_pol_modulation, add_depol_modulation=add_depol_modulation,add_volume_exclusion=add_volume_exclusion)
        chemistry_writer.write_nucleation_chemistry(chemistry_dir)
        #adapted to double filaments in the function
    
    """
    if add_volume_exclusion:
        nooverlap_atoms = [[6, 7, 8, 9, 10, 11, 12, 13],[6, 7, 8, 12, 13]]
    else:
        nooverlap_atoms = [[6, 7, 8, 9, 10, 11, 12, 13],[6, 7, 8, 9, 10, 11, 12, 13]] #only 9,10,11 and 1,2,3 are included
    """
    
    if parallel_filament or repulsive_interaction_antiparallel_filaments: #these have 9-9 attraction so no volume exclusion, but they do have 9-11 volume exclusion
        overlap_cutoffs = [3, 1, 0.9, 3, 2, 0.9, 3, 3, 0.9, 11, 9, 0.1, 11, 10, 0.1, 11, 11, 0.1] 
    elif parallel_and_antiparallel or antiparallel_filament or antiparallel_filaments_modulate_both_patches: #no 9*11-9*11 volume exclusion
        overlap_cutoffs = [3, 1, 0.9, 3, 2, 0.9, 3, 3, 0.9]
    elif not are_patches_interacting:
        overlap_cutoffs = [3, 1, 0.9, 3, 2, 0.9, 3, 3, 0.9, 11, 9, 0.1, 11, 10, 0.1, 11, 11, 0.1] 
    
    nooverlap_atoms_secnuc = [6,6,6,7,6,8,7,6,7,7,7,8,8,6,8,7,8,8] #this one without WCA cosine squared 0.1 0.4 - antiparallel filaments - if there is attraction between 9-9 pearticles use this has WCA between 1*3 and 9*11
    
    if repulsive_interaction_antiparallel_filaments:
        nooverlap_atoms = [3,6,3,7,3,8,3,12,3,13,11,6,11,7,11,8,11,12,11,13,8,1,8,2,8,3,8,6,8,7,8,8,8,9,8,10,8,11,8,12,8,13]
    elif antiparallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches:# these have 9-9 sttraction so no volume exclusion
        nooverlap_atoms = [3,6,3,7,3,8,3,12,3,13,11,6,11,7,11,8,11,9,11,10,11,11,11,12,11,13,8,1,8,2,8,3,8,6,8,7,8,8,8,9,8,10,8,11,8,12,8,13] #this one without WCA cosine squared 0.1 0.4 - antiparallel filaments - if there is attraction between 9-9 pearticles use this has WCA between 1*3 and 9*11
    else:
        nooverlap_atoms = [3,6,3,7,3,8,3,12,3,13,11,6,11,7,11,8,11,12,11,13,8,1,8,2,8,3,8,6,8,7,8,8,8,9,8,10,8,11,8,12,8,13] #this one without WCA cosine squared 0.1 0.4 - parallel and antiparallel filaments - if there is attraction between 9-9 pearticles use this has WCA between 1*3 and 9*11
        
    
    filament_block = DoubleBADDoubleFilamentTwoPatchesBlock(filament_parameters=filament_parameters)
    if constriction:
        bad_filament_dump_block = BADFilamentDumpBlockConstriction(simulation_parameters=simulation_parameters, compress=True, write_bonds=False, write_angles=False, write_dihedrals=False, filenameextra=str(runtime_mins)+'mins' )
    else:
        bad_filament_dump_block = BADFilamentDumpBlock(simulation_parameters=simulation_parameters, compress=True, write_bonds=False, write_angles=False, write_dihedrals=False )
    treadmilling_block = TreadmillingBlockVolumeExclusion(
        simulation_parameters=simulation_parameters,
        treadmilling_parameters=treadmilling_parameters,
        nuctype='plane',
        max_nmer=chemistry_writer.max_nmer,
        stabilize_steps=0,
        nooverlap=nooverlap_atoms,
        nooverlap_atoms_secnuc=nooverlap_atoms_secnuc,
        overlap_cutoffs=overlap_cutoffs
    )
    
    if constriction:
        if vary_nucleation:
            filename = 'trajectory_constriction_' + str(runtime_mins) + 'mins_vary_nucleation'
        else:
            filename = 'trajectory_constriction_' + str(runtime_mins) + 'mins'
    elif restart:
        filename = 'trajectory_restart_' + str(max_restart_number)
    else:
        filename = 'trajectory'
    system_dump_block = SystemDumpBlock(
        simulation_parameters=simulation_parameters,
        compress=True, extra_dump_list = ['c_cPatchNeighbors', 'c_cPatchNeighbors2'], filename=filename
    )

    # load the input template and fill it using jinja
    if constriction:
        if vary_nucleation:
            template_path = Path(__file__).resolve().parent.joinpath('template_constriction.lmp')
            output_path = dirpath.joinpath('input_constriction_vary_nucleation.lmp')
        else:
            template_path = Path(__file__).resolve().parent.joinpath('template_constriction.lmp')
            output_path = dirpath.joinpath('input_constriction.lmp')
    elif restart:
        template_path = Path(__file__).resolve().parent.joinpath('template_restart.lmp')
        output_path = dirpath.joinpath('input_restart.lmp')
    else:
        template_path = Path(__file__).resolve().parent.joinpath('template.lmp')
        output_path = dirpath.joinpath('input.lmp')
    with open(template_path, 'r') as f:
        template = jinja2.Template(f.read())
    
    #ADD RECALCULATING OF Kon,Koff BASED ON MODULATION IN THE INPUT TEMPLATE
    template_output = template.render(
        simulation_parameters=simulation_parameters,
        filament_parameters=filament_parameters,
        treadmilling_parameters=treadmilling_parameters,
        filament_block=filament_block,
        bad_filament_dump_block=bad_filament_dump_block,
        treadmilling_block=treadmilling_block,
        system_dump_block=system_dump_block,
    )

    with open(output_path, 'w') as f:
        f.write(template_output)
    

if __name__ == '__main__':
    #main()
    ###### CHANGED MODULATION TO ACCOMODATE PARALLEL FILAMENTS!####
            
    real_system_size = 3000 # nm, was 1000 for treadmilling, 3000 corresponds to box size = 600, apparently Chris' setup, 2000 for square
    r_on = 20 #mean field rate was 9, explicit is 8
    tau_hyd = 0.5  #mean field rate was 15, explicit is 1
    r_off = 10 #mean field rate was 10, explicit is 2, new rates roff is 4, short fils roff is 6
    Nseeds = 15
    np.random.seed(0)
    seed = np.random.randint(1, 999999, Nseeds)
    #seed = all_seed[5:Nseeds+5]
    patch_distance = 0.5
    new_patch_distance = 0.9
    patch1_angle_rigidity = 5e4
    patch2_angle_rigidity = 5e4
    filament_stiffness = [1000]
    restart = False
    reservoir = False
    #secondary nucleation in plane
    secnuc = False
    prob_secnuc = 1 #probability of secondary nucleation, 0.1 is a good value
    prob_changetype = 0.01 #probability of changing type of filament, 0.1 is a good value
    r_secnuc = 1
    #nucleation on top of filaments - stacking 
    stacking = False
    prob_stacking = 0.1 #probability of stacking, 0.1 is a good value
    prob_changetype_stacking = 0.01 #probability of changing type of filament, 0.1 is a good value
    r_stacking = 1
    #nucleation rate depending on monomer number
    vary_nucleation = True
    max_atoms = 10000
    
    constriction = False
    if constriction:
        runtime_mins = 10 #longer runtime for constriction since it is slower
    else:
        runtime_mins = 30
    hydrolysis_mode = 'explicit' # 'explicit' or 'mean-field'
    
    #volume exclusion to fix error 
    add_volume_exclusion = True
    #if add_volume_exclusion:
    #    real_system_size *= 2.1 #make volume *2.1? Csuse the excluded volume width of filament goes from 1->2.1

    # write templates for modulated polymerization/depolymerization
    are_patches_interacting = True
    add_pol_modulation = True
    add_depol_modulation = True
    r_on_modulation = 20 # 11/20 for treadmilling, 9.5/10 for no treadmilling
    tau_hyd_modulation = 0.5
    r_off_modulation = 8 #
    
    add_min_treadmilling_profile = True
    add_alignment_force = False 
    
    #MAKEE FILAMENTS PARALLEL
    antiparallel_filament = True 
    repulsive_interaction_antiparallel_filaments = False
    antiparallel_filaments_modulate_both_patches= False 
    parallel_filament = False
    parallel_and_antiparallel = False 
    
    infinite_growth = False
    # if add_pol_modulation:
    #     if r_on_modulation > r_off:
    #         infinite_growth = True
            
    for stiffness in filament_stiffness:
        for s in seed:
            dirpath = Path(__file__).parent.resolve() / 'ring_D_1_dist0.9_explicithydrolysis_ron20_roff10_tau0.5_nuccoupledtoreservoir'/ 'antiparallel_filaments'/ 'modulation_justdepolymerisation_rnuc100_reservoir10000' / f'stiffness_{stiffness}_seed_{s}'
            dirpath.mkdir(parents=True, exist_ok=True)
            max_restart_number = 0
            if restart:
                max_restart_number = get_max_restart_number(dirpath)
                shutil.copy(Path(__file__).parent.resolve() / 'job_restart.sh', dirpath / 'job_restart.sh')
            if constriction:
                max_restart_number = get_max_restart_number(dirpath)
                if vary_nucleation:
                    shutil.copy(Path(__file__).parent.resolve() / 'job_constriction_varynuc.sh', dirpath / 'job_constriction.sh')
                else:
                    shutil.copy(Path(__file__).parent.resolve() / 'job_constriction.sh', dirpath / 'job_constriction.sh')
            elif secnuc:
                shutil.copy(Path(__file__).parent.resolve() / 'job_secnuc.sh', dirpath / 'job_secnuc.sh')
            else:
                shutil.copy(Path(__file__).parent.resolve() / 'job.sh', dirpath / 'job.sh')
            main(dirpath, real_system_size, r_on, tau_hyd, r_off, r_off_modulation, s, patch_distance,new_patch_distance, patch1_angle_rigidity, patch2_angle_rigidity, stiffness, are_patches_interacting, add_pol_modulation, add_depol_modulation, r_on_modulation, tau_hyd_modulation, infinite_growth, add_min_treadmilling_profile, add_volume_exclusion,add_alignment_force,parallel_filament, parallel_and_antiparallel,antiparallel_filament,antiparallel_filaments_modulate_both_patches,repulsive_interaction_antiparallel_filaments, runtime_mins, constriction, hydrolysis_mode, reservoir, secnuc, prob_secnuc, prob_changetype, r_secnuc,max_restart_number, restart, stacking, prob_stacking, prob_changetype_stacking, r_stacking, vary_nucleation, max_atoms)
            if constriction:
                subprocess.run(["sbatch", "job_constriction.sh"], cwd=dirpath)
            elif restart:
                subprocess.run(["sbatch", "job_restart.sh"], cwd=dirpath)
            elif secnuc:
                subprocess.run(["sbatch", "job_secnuc.sh"], cwd=dirpath)
            else:
                subprocess.run(["sbatch", "job.sh"], cwd=dirpath)