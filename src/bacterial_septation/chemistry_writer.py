from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

from bacterial_septation.structure_init import StructureInitialiser, BADUtils, TripleBeadPatchUtils, TripleBeadPatchDiffuseUtils
from bacterial_septation import escrt_utils

CHEMISTRY_MODES = ('BAD', 'ESCRT', 'ESCRTDoublePatch','DoubleBAD','DoubleFilaments','DoubleFilamentsTwoPatches', 'TripleBAD', 'TripleBeadPatchFilaments', 'TripleBeadPatchDiffuseFilaments')
SEED_MODES = ('dimer', 'trimer')

class BaseTopologyFactory():
    def __init__(self, equilibrium_config):
        self.equilibrium_helix = equilibrium_config
        self.monomer_size: int = 1

    @property
    def max_nmer(self):
        return self.equilibrium_helix.shape[0]//self.monomer_size

    def oligomer_head(self, start_index: int=0, structure_initialiser: StructureInitialiser | None = None):
        raise NotImplementedError

    def oligomer_tail(self, structure_initialiser: StructureInitialiser | None = None):
        raise NotImplementedError
    
    def fit_string(self, n_beads, n_fit, structure_initialiser: StructureInitialiser | None = None):
        raise NotImplementedError

    def straight_dimer(self, structure_initialiser: StructureInitialiser | None = None):
        raise NotImplementedError

    # optional overrides of the reaction initiator IDs (pre-template atom IDs) used by ChemistryWriter;
    # returning None keeps ChemistryWriter's default initiators
    def on_initiators(self, n_pre_beads: int) -> tuple[int, int] | None:
        return None

    def nmer_on_initiators(self, n_mer: int) -> tuple[int, int] | None:
        return None

    def off_initiators(self) -> tuple[int, int] | None:
        return None

    def denucleation_initiators(self) -> tuple[int, int] | None:
        return None

    # optional overrides of the edge atom IDs of the oligomer (OligomerOn / OligomerOff) maps
    def on_edges(self) -> tuple[int, ...] | None:
        return None

    def depol_fragment_atom(self) -> int | None:
        """Atom of the off (pre) templates used as fragment 2 for depol modulation, if not covered by the
        monomer_size 2/3/4 branches in ChemistryWriter. None = use those branches."""
        return None

    def off_edges(self, n_pre_beads: int) -> tuple[int, ...] | None:
        return None

    def shrink_tail(self, structure_initialiser: StructureInitialiser):
        # remove bonds angles and dihedrals from deleted bead
        monomer_size = self.monomer_size
        tail_indices = np.arange(1, monomer_size + 1)
        tail_types = structure_initialiser.atom_types_array[:monomer_size]

        ## make last monomer a tail type
        structure_initialiser.atom_types_array[monomer_size:2*monomer_size] = tail_types

        ## delete bonds, angles and dihedrals involving deleted bead
        ### mask of all bonds, angles and dihedrals involving first monomer
        bond_to_delete_mask = np.zeros(len(structure_initialiser.bond_array), dtype=bool)
        angle_to_delete_mask = np.zeros(len(structure_initialiser.angle_array), dtype=bool)
        dihedral_to_delete_mask = np.zeros(len(structure_initialiser.dihedral_array), dtype=bool)

        for tail_index in tail_indices:
            bond_to_delete_mask = np.logical_or(bond_to_delete_mask, np.any(structure_initialiser.bond_array[:, 2:] == tail_index, axis=1))

            if structure_initialiser.n_angles:
                angle_to_delete_mask = np.logical_or(angle_to_delete_mask, np.any(structure_initialiser.angle_array[:, 2:] == tail_index, axis=1))
            if structure_initialiser.n_dihedrals:
                dihedral_to_delete_mask = np.logical_or(dihedral_to_delete_mask, np.any(structure_initialiser.dihedral_array[:, 2:] == tail_index, axis=1))

        ## convert masks to indices to whole rows can be deleted
        bond_to_delete_indices = np.where(bond_to_delete_mask)[0]
        angle_to_delete_indices = np.where(angle_to_delete_mask)[0]
        dihedral_to_delete_indices = np.where(dihedral_to_delete_mask)[0]

        structure_initialiser.bond_array = np.delete(structure_initialiser.bond_array, bond_to_delete_indices, axis=0)
        structure_initialiser.angle_array = np.delete(structure_initialiser.angle_array, angle_to_delete_indices, axis=0)
        structure_initialiser.dihedral_array = np.delete(structure_initialiser.dihedral_array, dihedral_to_delete_indices, axis=0)

        return structure_initialiser

class BADTopologyFactory(BaseTopologyFactory):
    def __init__(self, equilibrium_config, step_helix: int, add_nn_angles: bool=True, add_nth_neigbour_bonds: bool =False):
        if len(equilibrium_config) < (4*step_helix):
            raise ValueError('Please supply a long enough equilibrium configuration.')

        self.monomer_size: int = 1

        self.equilibrium_helix = equilibrium_config[:max(5, 4*step_helix)]
        self.step_helix = step_helix
        self.add_nn_angles = add_nn_angles
        self.add_nth_neigbour_bonds = add_nth_neigbour_bonds

    def oligomer_head(self, start_index: int=0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser.add_as_bad_chain(
            self.equilibrium_helix[start_index:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )
        structure_initialiser.atom_types_array[-1] = 3

        return structure_initialiser
    
    def oligomer_tail(self, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser.add_as_bad_chain(
            self.equilibrium_helix[:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )
        structure_initialiser.atom_types_array[-len(self.equilibrium_helix[:])] = 2

        return structure_initialiser

    def nmer(self, n_mer, structure_initialiser: StructureInitialiser | None = None):
        if n_mer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the BAD topology as it can be expressed as an oligomer.')

        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser.add_as_bad_chain(
            self.equilibrium_helix[:n_mer], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )
        structure_initialiser.atom_types_array[-n_mer] = 2
        structure_initialiser.atom_types_array[-1] = 3

        return structure_initialiser

    def straight_dimer(self, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        parallel_line = np.zeros((2, 3), dtype=float)
        parallel_line[:, 0] = [2, 3]

        structure_initialiser.add_as_bad_chain(
            parallel_line, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )
        structure_initialiser.atom_types_array[-len(parallel_line)] = 2
        structure_initialiser.atom_types_array[-1] = 3

        return structure_initialiser

    def fit_string(self, n_beads, n_fit):
        return f"2 {' '.join([str(ii) for ii in range(1, n_beads+1)][-n_fit:])}"

class ESCRTTopologyFactory(BaseTopologyFactory):
    def __init__(self, equilibrium_config, use_rigid: bool=False):
        self.monomer_size: int = 3

        n_trimers_min = 4

        if len(equilibrium_config) < (n_trimers_min * self.monomer_size):
            raise ValueError(f'Please supply a long enough equilibrium configuration, at least {n_trimers_min} trimers.')

        # needs only 4 trimers for minimum structure
        self.equilibrium_helix = equilibrium_config[:n_trimers_min*self.monomer_size]
        self.use_rigid = use_rigid

    def oligomer_head(self, start_index = 0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser._add_as_escrt_chain(
            self.equilibrium_helix[start_index*self.monomer_size:],
            use_rigid=self.use_rigid
        )

        structure_initialiser.atom_types_array[:self.monomer_size] = escrt_utils.BULK_TYPES

        return structure_initialiser
    
    def oligomer_tail(self, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser._add_as_escrt_chain(
            self.equilibrium_helix[:],
            use_rigid=self.use_rigid
        )

        structure_initialiser.atom_types_array[-self.monomer_size:] = escrt_utils.BULK_TYPES

        return structure_initialiser
    
    def nmer(self, n_mer, structure_initialiser: StructureInitialiser | None = None):
        if n_mer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the ESCRT topology as it can be expressed as an oligomer.')
        
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser._add_as_escrt_chain(
            self.equilibrium_helix[:n_mer*self.monomer_size],
            use_rigid=self.use_rigid
        )

        return structure_initialiser
    
    def straight_dimer(self, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        if False:
            # now construct a triangle conformation from it
            # equally spaced in x, according to eq. config
            # equilateral triangle around 0, 0 in YZ-plane
            triangle_yz = np.zeros((2*self.monomer_size, 3), dtype=float)

            # first move x-coordinates forward
            x_step = np.linalg.norm(self.equilibrium_helix[0] - self.equilibrium_helix[self.monomer_size])
            triangle_yz[:self.monomer_size, 0] = 2
            triangle_yz[self.monomer_size:, 0] = 1 + 2*x_step

            # now the individual triplets have to be shifted to form a triangle
            equilibrium_distance = np.linalg.norm(self.equilibrium_helix[0] - self.equilibrium_helix[1])
            triangle_height = np.sqrt(3) * equilibrium_distance / 2
            triangle_yz[0::3, 1] = triangle_height/2
            triangle_yz[1::3, 1] = -triangle_height/2
            triangle_yz[2::3, 1] = -triangle_height/2

            triangle_yz[1::3, 2] = +equilibrium_distance/2
            triangle_yz[2::3, 2] = -equilibrium_distance/2
        else:
            main_beads = BADUtils.get_centered_line(2, r_zero=1)
            main_beads[:, 0] = [2, 3]
            membrane_patches = main_beads.copy()
            membrane_patches[:, 2] -= 0.5
            connecting_patches = main_beads.copy()
            connecting_patches[:, 1] += 0.5

            triangle_yz = np.zeros((len(main_beads)*3, 3))
            triangle_yz[0::3, :] = main_beads
            triangle_yz[1::3, :] = membrane_patches
            triangle_yz[2::3, :] = connecting_patches

        # equilibrium helix should have right structure, therefore get triangle side length from first two beads
        equilibrium_distance = np.linalg.norm(self.equilibrium_helix[0] - self.equilibrium_helix[1])

        structure_initialiser._add_as_escrt_chain(
            triangle_yz,
            use_rigid=self.use_rigid
        )

        return structure_initialiser

    def fit_string(self, n_beads, n_fit) -> str:
        return f"2 {' '.join([str(ii) for ii in range(1, n_beads+1)][-n_fit*self.monomer_size:])}"
    
class ESCRTDoublePatchTopologyFactory(BaseTopologyFactory):
    def __init__(self, equilibrium_config, use_rigid: bool=False):
        self.monomer_size: int = 3

        n_trimers_min = 4

        if len(equilibrium_config) < (n_trimers_min * self.monomer_size):
            raise ValueError(f'Please supply a long enough equilibrium configuration, at least {n_trimers_min} trimers.')

        # needs only 4 trimers for minimum structure
        self.equilibrium_helix = equilibrium_config[:n_trimers_min*self.monomer_size]
        self.use_rigid = use_rigid

    def oligomer_head(self, start_index = 0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser._add_as_escrt_chain(
            self.equilibrium_helix[start_index*self.monomer_size:],
            use_rigid=self.use_rigid
        )

        structure_initialiser.atom_types_array[:self.monomer_size] = escrt_utils.BULK_TYPES

        return structure_initialiser
    
    def oligomer_tail(self, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser._add_as_escrt_chain(
            self.equilibrium_helix[:],
            use_rigid=self.use_rigid
        )

        structure_initialiser.atom_types_array[-self.monomer_size:] = escrt_utils.BULK_TYPES

        return structure_initialiser
    
    def nmer(self, n_mer, structure_initialiser: StructureInitialiser | None = None):
        if n_mer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the ESCRT topology as it can be expressed as an oligomer.')
        
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        structure_initialiser._add_as_escrt_chain(
            self.equilibrium_helix[:n_mer*self.monomer_size],
            use_rigid=self.use_rigid
        )

        return structure_initialiser
    
    def straight_dimer(self, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        if False:
            # now construct a triangle conformation from it
            # equally spaced in x, according to eq. config
            # equilateral triangle around 0, 0 in YZ-plane
            triangle_yz = np.zeros((2*self.monomer_size, 3), dtype=float)

            # first move x-coordinates forward
            x_step = np.linalg.norm(self.equilibrium_helix[0] - self.equilibrium_helix[self.monomer_size])
            triangle_yz[:self.monomer_size, 0] = 2
            triangle_yz[self.monomer_size:, 0] = 1 + 2*x_step

            # now the individual triplets have to be shifted to form a triangle
            equilibrium_distance = np.linalg.norm(self.equilibrium_helix[0] - self.equilibrium_helix[1])
            triangle_height = np.sqrt(3) * equilibrium_distance / 2
            triangle_yz[0::3, 1] = triangle_height/2
            triangle_yz[1::3, 1] = -triangle_height/2
            triangle_yz[2::3, 1] = -triangle_height/2

            triangle_yz[1::3, 2] = +equilibrium_distance/2
            triangle_yz[2::3, 2] = -equilibrium_distance/2
        else:
            beta = math.radians(30)
            patch_distance = 0.5
            main_beads = BADUtils.get_centered_line(2, r_zero=1)
            main_beads[:, 0] = [2, 3]
            membrane_patches = main_beads.copy()
            connecting_patches = main_beads.copy()
            main_beads[:, 2] += patch_distance
            
            membrane_patches[:,2]+= patch_distance*(1-math.cos(beta))
            membrane_patches[:,1]-= patch_distance*math.sin(beta)
            #main_beads[:, 2] += patch_distance
            #connecting_patches = main_beads.copy()
            connecting_patches[:, 2]+= patch_distance*(1-math.cos(beta))
            connecting_patches[:,1]+= patch_distance*math.sin(beta)

            triangle_yz = np.zeros((len(main_beads)*3, 3))
            triangle_yz[0::3, :] = main_beads
            triangle_yz[1::3, :] = membrane_patches
            triangle_yz[2::3, :] = connecting_patches

        # equilibrium helix should have right structure, therefore get triangle side length from first two beads
        equilibrium_distance = np.linalg.norm(self.equilibrium_helix[0] - self.equilibrium_helix[1])

        structure_initialiser._add_as_escrt_chain(
            triangle_yz,
            use_rigid=self.use_rigid
        )

        return structure_initialiser

    def fit_string(self, n_beads, n_fit) -> str:
        return f"2 {' '.join([str(ii) for ii in range(1, n_beads+1)][-n_fit*self.monomer_size:])}"

class DoubleBADTopologyFactory(BaseTopologyFactory):
    def __init__(self, equilibrium_config, step_helix: int, add_nn_angles: bool=True, add_nth_neigbour_bonds: bool = False, r_zero: float=1.0, patch_distance: float|None=None):
        if len(equilibrium_config) < (8*step_helix):
            raise ValueError('Please supply a long enough equilibrium configuration.')

        self.monomer_size: int = 2

        self.equilibrium_helix = equilibrium_config[:max(5, 4*step_helix)*2]
        self.step_helix = step_helix
        self.add_nn_angles = add_nn_angles
        self.add_nth_neigbour_bonds = add_nth_neigbour_bonds
        self.r_zero = r_zero
        if patch_distance is None:
           patch_distance = r_zero / 2.
        self.patch_distance = patch_distance

    def oligomer_head(self, start_index = 0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        insert_helix = self.equilibrium_helix[start_index*2:, :].copy()

        structure_initialiser.add_as_double_bad_chain(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )

        # make last monomer a head type
        structure_initialiser.atom_types_array[-2] = 3
        structure_initialiser.atom_types_array[-1] = 8

        return structure_initialiser

    def oligomer_tail(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        structure_initialiser.add_as_double_bad_chain(
            self.equilibrium_helix[:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )

        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)] = 2
        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)+1] = 7
        
        return structure_initialiser

    def nmer(self, nmer, structure_initialiser: StructureInitialiser|None = None):
        if nmer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the double BAD topology as it can be expressed as an oligomer.')
        
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        insert_helix = self.equilibrium_helix[:nmer*2, :].copy()

        structure_initialiser.add_as_double_bad_chain(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )
        structure_initialiser.atom_types_array[-len(insert_helix)] = 2
        structure_initialiser.atom_types_array[-len(insert_helix)+1] = 7
        structure_initialiser.atom_types_array[-2] = 3
        structure_initialiser.atom_types_array[-1] = 8

        return structure_initialiser
    
    def straight_dimer(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        parallel_line = np.zeros((4, 3), dtype=float)
        # straight line along x
        parallel_line[:, 0] = [2, 2, 2+self.r_zero, 2+self.r_zero]
        # shift between beads in y is 0.5 r_zero
        parallel_line[1::2, 1] = [self.patch_distance, self.patch_distance] # distance of patches from center of helix beads

        structure_initialiser.add_as_double_bad_chain(
            parallel_line, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )

        structure_initialiser.atom_types_array[-4] = 2
        structure_initialiser.atom_types_array[-3] = 7
        structure_initialiser.atom_types_array[-2] = 3
        structure_initialiser.atom_types_array[-1] = 8
        return structure_initialiser
    
    def fit_string(self, n_beads, n_fit) -> str:
        all_indices = np.arange(1, n_beads+1)
        fit_indices = all_indices[-n_fit*2:]
        return f"2 {' '.join([str(index) for index in fit_indices])}"
    
class DoubleBADDoubleFilamentsTopologyFactory(BaseTopologyFactory):
    def __init__(self, equilibrium_config, step_helix: int, add_nn_angles: bool=False, add_nth_neigbour_bonds: bool = False, r_zero: float=1.0, patch_distance: float|None=None):
        if len(equilibrium_config) < (8*step_helix):
            raise ValueError('Please supply a long enough equilibrium configuration.')

        self.monomer_size: int = 2

        self.equilibrium_helix = equilibrium_config[:max(5, 4*step_helix)*2]
        self.step_helix = step_helix
        self.add_nn_angles = add_nn_angles
        self.add_nth_neigbour_bonds = add_nth_neigbour_bonds
        self.r_zero = r_zero
        if patch_distance is None:
           patch_distance = r_zero / 2.
        self.patch_distance = patch_distance
        
    def secnuc(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        structure_initialiser.add_as_double_bad_chain_double_filaments(
            self.equilibrium_helix[:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )
        
        structure_initialiser.angle_array = structure_initialiser.angle_array[:-1]
        
        return structure_initialiser
    
    def secnuc_with_nucleated_trimer(self, structure_initialiser: StructureInitialiser|None = None, equilibirum_distance: float = 0.1):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        structure_initialiser.add_as_double_bad_chain_double_filaments(
            self.equilibrium_helix[:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )
        
        structure_initialiser.angle_array = structure_initialiser.angle_array[:-1]
        
        self.nmer(3, structure_initialiser=structure_initialiser) #adding the trimer bonds, angles and types
        
        structure_initialiser.positions[-6:-4, 0] = structure_initialiser.positions[self.monomer_size*4, 0] # move trimer to correspond to secnuc
        structure_initialiser.positions[-4:-2, 0] = structure_initialiser.positions[self.monomer_size*3, 0]
        structure_initialiser.positions[-2:, 0] = structure_initialiser.positions[self.monomer_size*2, 0]
        
        structure_initialiser.positions[-5::2, 1] = structure_initialiser.positions[1, 1] + equilibirum_distance # move trimer patches to correspond to secnuc - moving by the 
        structure_initialiser.positions[-6::2, 1] = structure_initialiser.positions[1, 1] + equilibirum_distance + self.patch_distance # move trimer patches to correspond to secnuc - moving by the patch distance from the center of the helix beads
        
        return structure_initialiser

    def oligomer_head(self, start_index = 0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        insert_helix = self.equilibrium_helix[start_index*2:, :].copy()

        structure_initialiser.add_as_double_bad_chain_double_filaments(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )

        # make last monomer a head type
        structure_initialiser.atom_types_array[-2] = 3
        structure_initialiser.atom_types_array[-1] = 8

        return structure_initialiser

    def oligomer_tail(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        structure_initialiser.add_as_double_bad_chain_double_filaments(
            self.equilibrium_helix[:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )

        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)] = 2
        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)+1] = 7
        
        return structure_initialiser

    def nmer(self, nmer, structure_initialiser: StructureInitialiser|None = None):
        if nmer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the double BAD topology as it can be expressed as an oligomer.')
        
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        insert_helix = self.equilibrium_helix[:nmer*2, :].copy()

        structure_initialiser.add_as_double_bad_chain_double_filaments(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )
        structure_initialiser.atom_types_array[-len(insert_helix)] = 2
        structure_initialiser.atom_types_array[-len(insert_helix)+1] = 7
        structure_initialiser.atom_types_array[-2] = 3
        structure_initialiser.atom_types_array[-1] = 8

        return structure_initialiser
    
    def straight_dimer(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        parallel_line = np.zeros((4, 3), dtype=float)
        # straight line along x
        parallel_line[:, 0] = [2, 2, 2+self.r_zero, 2+self.r_zero]
        # shift between beads in y is 0.5 r_zero
        parallel_line[1::2, 1] = [self.patch_distance, self.patch_distance] # distance of patches from center of helix beads

        structure_initialiser.add_as_double_bad_chain_double_filaments(
            parallel_line, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )

        structure_initialiser.atom_types_array[-4] = 2
        structure_initialiser.atom_types_array[-3] = 7
        structure_initialiser.atom_types_array[-2] = 3
        structure_initialiser.atom_types_array[-1] = 8
        return structure_initialiser
    
    def fit_string(self, n_beads, n_fit) -> str:
        all_indices = np.arange(1, n_beads+1)
        fit_indices = all_indices[-n_fit*2:]
        return f"2 {' '.join([str(index) for index in fit_indices])} \n"
    
    def fit_string_secnuc(self, n_beads, n_fit) -> str:
        all_indices = np.arange(1, n_beads+1)
        fit_indices = all_indices[:-n_fit*self.monomer_size] # only fit the filament beads, not the patches
        return f"2 {' '.join([str(index) for index in fit_indices])} \n"
    
class DoubleBADDoubleFilamentsTwoPatchesTopologyFactory(BaseTopologyFactory):
    def __init__(self, equilibrium_config, step_helix: int, add_nn_angles: bool=False, add_nth_neigbour_bonds: bool = False, r_zero: float=1.0, patch1_distance: float|None=None, patch2_distance: float|None=None):
        if len(equilibrium_config) < (8*step_helix):
            raise ValueError('Please supply a long enough equilibrium configuration.')

        self.monomer_size: int = 3

        self.equilibrium_helix = equilibrium_config[:max(5, 4*step_helix)*self.monomer_size]
        self.step_helix = step_helix
        self.add_nn_angles = add_nn_angles
        self.add_nth_neigbour_bonds = add_nth_neigbour_bonds
        self.r_zero = r_zero
        if patch1_distance is None:
           patch1_distance = r_zero / 2.
        self.patch1_distance = patch1_distance
        self.patch2_distance = patch2_distance
        
    def secnuc(self,  start_index = 0,structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
            
        #INSERYING A TRIMER ANDMANUALLY ADDING ONE MAIN FILAMENT BEAD AT THE TIP AND BACK
        
        structure_initialiser.atom_types_array = np.append(structure_initialiser.atom_types_array, 1)
        structure_initialiser.molecule_ids = np.append(structure_initialiser.molecule_ids,structure_initialiser.n_molecules)
        
        new_bond_array = np.zeros((1, 4), dtype=int) #add one bond between the filament atoms
        new_bond_array[0,:] = 1,1,1,2 #type 1 bond in filament, between atoms 1 and 2 - filament main beads
        structure_initialiser.bond_array = np.append(structure_initialiser.bond_array, new_bond_array, axis=0)
        
        new_angle_array = np.zeros((1, 5), dtype=int)
        new_angle_array[0,:] = 1,2,1,2,2+self.monomer_size
        structure_initialiser.angle_array = np.append(structure_initialiser.angle_array, new_angle_array, axis=0)
        
    
        insert_helix = self.equilibrium_helix[start_index*self.monomer_size:, :].copy()
        initial_bead_filament = self.equilibrium_helix[start_index*self.monomer_size, :].copy()
        initial_bead_filament[0]-= 1
        #print(structure_initialiser.positions)
        #print(initial_bead_filament)
        #print(insert_helix)
        structure_initialiser.positions = np.append(structure_initialiser.positions, [initial_bead_filament], axis=0)

        self.nmer(3, change_types=False, structure_initialiser=structure_initialiser) #adding the trimer bonds, angles and types, as the full topology
        
        #DELETE TWO ANGLES THAT ARE WRONG
        a1, b1, c1 = 9, 8, 5
        a2, b2, c2 = 10, 8, 5
        mask = ~np.all(structure_initialiser.angle_array[:, 2:] == np.array([a1, b1, c1]), axis=1)
        structure_initialiser.angle_array = structure_initialiser.angle_array[mask]
        mask = ~np.all(structure_initialiser.angle_array[:, 2:] == np.array([a2, b2, c2]), axis=1)  
        structure_initialiser.angle_array = structure_initialiser.angle_array[mask]
        
        #add another main filament bead in the end as edge atom
        structure_initialiser.atom_types_array = np.append(structure_initialiser.atom_types_array, 1)
        structure_initialiser.molecule_ids = np.append(structure_initialiser.molecule_ids,structure_initialiser.n_molecules)
        
        new_bond_array = np.zeros((1, 4), dtype=int) #add one bond between the filament atoms
        new_bond_array[0,:] = 1,1,structure_initialiser.n_beads-2,structure_initialiser.n_beads + 1 #type 1 bond in filament, between atoms 1 and 2 - filament main beads
        structure_initialiser.bond_array = np.append(structure_initialiser.bond_array, new_bond_array, axis=0)
        
        new_angle_array = np.zeros((3, 5), dtype=int)
        new_angle_array[0,:] = 1,2,structure_initialiser.n_beads-5,structure_initialiser.n_beads-2,structure_initialiser.n_beads + 1
        new_angle_array[1,:] = 2,3, structure_initialiser.n_beads-1 ,structure_initialiser.n_beads-2,structure_initialiser.n_beads + 1
        new_angle_array[2,:] = 3,4, structure_initialiser.n_beads ,structure_initialiser.n_beads-2,structure_initialiser.n_beads + 1
        structure_initialiser.angle_array = np.append(structure_initialiser.angle_array, new_angle_array, axis=0)
        
        
        #STANDARD LOGIC 
        last_bead_filament = structure_initialiser.positions[-3].copy()
        #print(last_bead_filament)
        last_bead_filament[0] +=1 
        print(last_bead_filament)
        #print(structure_initialiser.positions)
        #print(initial_bead_filament)
        #print(insert_helix)
        structure_initialiser.positions = np.append(structure_initialiser.positions, [last_bead_filament], axis=0)
        
        ####CHANGE MIDDDLE MAIN ATOM TO TYPE 4 TO AVOID SECNUC EVERYWHERE
        structure_initialiser.atom_types_array[4]=4
        
        return structure_initialiser
    
    def secnuc_with_nucleated_dimer_3D(self, structure_initialiser: StructureInitialiser|None = None, equilibirum_distance: float = 0.1):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
            
        #INSERYING A TRIMER ANDMANUALLY ADDING ONE MAIN FILAMENT BEAD AT THE TIP AND BACK
        
        structure_initialiser = self.secnuc(structure_initialiser=structure_initialiser) # add the template initial filament
        
        self.nmer(2, structure_initialiser=structure_initialiser) #adding the trimer bonds, angles and types, as the full topology
        
        structure_initialiser.positions[-9:-6, 0] = structure_initialiser.positions[self.monomer_size*3, 0] # move trimer to correspond to secnuc
        structure_initialiser.positions[-6:-3, 0] = structure_initialiser.positions[self.monomer_size*2, 0]
        structure_initialiser.positions[-3:, 0] = structure_initialiser.positions[self.monomer_size, 0]
        
        structure_initialiser.positions[-7::3, 1] = structure_initialiser.positions[2, 1] + equilibirum_distance + self.patch1_distance + self.patch2_distance # move trimer patches to correspond to secnuc - moving by the 
        structure_initialiser.positions[-8::3, 1] = structure_initialiser.positions[2, 1] + equilibirum_distance # move trimer patches to correspond to secnuc - moving by the patch distance from the center of the helix beads
        structure_initialiser.positions[-9::3, 1] = structure_initialiser.positions[2, 1] + equilibirum_distance + self.patch1_distance
        
        
        return structure_initialiser
    
    def secnuc_with_nucleated_trimer(self, structure_initialiser: StructureInitialiser|None = None, equilibirum_distance: float = 0.1):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
            
        #INSERYING A TRIMER ANDMANUALLY ADDING ONE MAIN FILAMENT BEAD AT THE TIP AND BACK
        
        structure_initialiser = self.secnuc(structure_initialiser=structure_initialiser) # add the template initial filament
        
        self.nmer(3, structure_initialiser=structure_initialiser) #adding the trimer bonds, angles and types, as the full topology
        
        structure_initialiser.positions[-9:-6, 0] = structure_initialiser.positions[self.monomer_size*3, 0] # move trimer to correspond to secnuc
        structure_initialiser.positions[-6:-3, 0] = structure_initialiser.positions[self.monomer_size*2, 0]
        structure_initialiser.positions[-3:, 0] = structure_initialiser.positions[self.monomer_size, 0]
        
        structure_initialiser.positions[-7::3, 1] = structure_initialiser.positions[2, 1] + equilibirum_distance + self.patch1_distance + self.patch2_distance # move trimer patches to correspond to secnuc - moving by the 
        structure_initialiser.positions[-8::3, 1] = structure_initialiser.positions[2, 1] + equilibirum_distance # move trimer patches to correspond to secnuc - moving by the patch distance from the center of the helix beads
        structure_initialiser.positions[-9::3, 1] = structure_initialiser.positions[2, 1] + equilibirum_distance + self.patch1_distance
        
        
        return structure_initialiser

    def oligomer_head(self, type_to_add = 1, start_index = 0,structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
            
        #CHANGING OLIGOMER LOGIC TO ADD A FILAMENT BEAD AT THE START OF THE TEMPLATE SO I HAVE 1 EDGEID
        
        #2 templates needed: 
        structure_initialiser.atom_types_array = np.append(structure_initialiser.atom_types_array, type_to_add)
        structure_initialiser.molecule_ids = np.append(structure_initialiser.molecule_ids,structure_initialiser.n_molecules)
        
        new_bond_array = np.zeros((1, 4), dtype=int) #add one bond between the filament atoms
        new_bond_array[0,:] = 1,1,1,2 #type 1 bond in filament, between atoms 1 and 2 - filament main beads
        structure_initialiser.bond_array = np.append(structure_initialiser.bond_array, new_bond_array, axis=0)
        
        new_angle_array = np.zeros((1, 5), dtype=int)
        new_angle_array[0,:] = 1,2,1,2,2+self.monomer_size
        structure_initialiser.angle_array = np.append(structure_initialiser.angle_array, new_angle_array, axis=0)
        
        
        #STANDARD LOGIC
        insert_helix = self.equilibrium_helix[start_index*self.monomer_size:, :].copy()
        initial_bead_filament = self.equilibrium_helix[start_index*self.monomer_size, :].copy()
        initial_bead_filament[0]-= 1
        #print(structure_initialiser.positions)
        #print(initial_bead_filament)
        #print(insert_helix)
        structure_initialiser.positions = np.append(structure_initialiser.positions, [initial_bead_filament], axis=0)

        structure_initialiser.add_as_double_filament_with_two_patches(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )
        
        # make last monomer a head type
        structure_initialiser.atom_types_array[-3] = 3
        structure_initialiser.atom_types_array[-2] = 8
        structure_initialiser.atom_types_array[-1] = 11
        

        return structure_initialiser

    def oligomer_tail(self, type_to_add =1, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        #STANDARD TEMPLATE
        
        structure_initialiser.add_as_double_filament_with_two_patches(
            self.equilibrium_helix[:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )

        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)] = 2
        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)+1] = 7
        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)+2] = 10
        
        #2 templates needed: 
        structure_initialiser.atom_types_array = np.append(structure_initialiser.atom_types_array, type_to_add)
        structure_initialiser.molecule_ids = np.append(structure_initialiser.molecule_ids,structure_initialiser.n_molecules)
        
        new_bond_array = np.zeros((1, 4), dtype=int) #add one bond between the filament atoms
        new_bond_array[0,:] = len(structure_initialiser.bond_array)+1,1,len(structure_initialiser.atom_types_array) - self.monomer_size ,len(structure_initialiser.atom_types_array ) #type 1 bond in filament, between atoms 1 and 2 - filament main beads
        structure_initialiser.bond_array = np.append(structure_initialiser.bond_array, new_bond_array, axis=0)
        
        new_angle_array = np.zeros((1, 5), dtype=int)
        new_angle_array[0,:] = len(structure_initialiser.angle_array)+1,2,len(structure_initialiser.atom_types_array) -2*self.monomer_size ,len(structure_initialiser.atom_types_array) -self.monomer_size,len(structure_initialiser.atom_types_array)
        structure_initialiser.angle_array = np.append(structure_initialiser.angle_array, new_angle_array, axis=0)
        
        initial_bead_filament = self.equilibrium_helix[-self.monomer_size, :].copy()
        initial_bead_filament[0]+= 1
        structure_initialiser.positions = np.append(structure_initialiser.positions, [initial_bead_filament], axis=0)
        
        return structure_initialiser

    def nmer(self, nmer, change_types: bool = True, structure_initialiser: StructureInitialiser|None = None):
        if nmer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the double BAD topology as it can be expressed as an oligomer.')
        
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        insert_helix = self.equilibrium_helix[:nmer*3, :].copy()

        structure_initialiser.add_as_double_filament_with_two_patches(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )
        if change_types:
            structure_initialiser.atom_types_array[-len(insert_helix)] = 2
            structure_initialiser.atom_types_array[-len(insert_helix)+1] =7
            structure_initialiser.atom_types_array[-len(insert_helix)+2] = 10
            structure_initialiser.atom_types_array[-3] = 3
            structure_initialiser.atom_types_array[-2] = 8
            structure_initialiser.atom_types_array[-1] = 11

        return structure_initialiser
    
    def straight_dimer(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        parallel_line = np.zeros((6, 3), dtype=float)
        # straight line along x
        parallel_line[:, 0] = [2, 2, 2, 2+self.r_zero, 2+self.r_zero, 2+self.r_zero]
        # shift between beads in y is 0.5 r_zero
        parallel_line[1::3, 1] = [self.patch1_distance, self.patch1_distance] # distance of patches from center of helix beads
        parallel_line[2::3, 1] = [-self.patch2_distance, -self.patch2_distance] # distance of patches from center of helix beads

        structure_initialiser.add_as_double_filament_with_two_patches(
            parallel_line, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )

        structure_initialiser.atom_types_array[-6] = 2
        structure_initialiser.atom_types_array[-5] = 7
        structure_initialiser.atom_types_array[-4] = 10
        structure_initialiser.atom_types_array[-3] = 3
        structure_initialiser.atom_types_array[-2] = 8
        structure_initialiser.atom_types_array[-1] = 11
        
        return structure_initialiser
    
    def fit_string(self, n_beads, n_fit) -> str:
        all_indices = np.arange(1, n_beads+1)
        fit_indices = all_indices[-n_fit*self.monomer_size:]
        return f"2 {' '.join([str(index) for index in fit_indices])} \n"
    
    def fit_string_secnuc(self, n_beads, n_fit) -> str:
        all_indices = np.arange(1, n_beads+1)
        fit_indices = all_indices[:-n_fit*self.monomer_size] # only fit the filament beads, not the patches
        return f"2 {' '.join([str(index) for index in fit_indices])} \n"
    
    def get_all_combinations_of_type_9_and_14(self, structure_initialiser: StructureInitialiser | None = None):
        atom_types = structure_initialiser.atom_types_array
        indices_of_9 = np.where(atom_types == 9)[0]
        number_of_type_9 = len(indices_of_9)
        number_of_combinations = 2 ** number_of_type_9
        atom_type_list = []

        for i in range(number_of_combinations):
            atom_types_copy = atom_types.copy()
            for bit, idx in enumerate(indices_of_9):
                if (i >> bit) & 1:
                    atom_types_copy[idx] = 14
            atom_type_list.append(atom_types_copy)

        return atom_type_list

class TripleBADTopologyFactory(BaseTopologyFactory):
    def __init__(self, equilibrium_config, step_helix: int, add_nn_angles: bool=True, add_nth_neigbour_bonds: bool = False, r_zero: float=1.0, patch_distance: float|None=None):
        if len(equilibrium_config) < (8*step_helix):
            raise ValueError('Please supply a long enough equilibrium configuration.')

        self.monomer_size: int = 3

        self.equilibrium_helix = equilibrium_config[:max(5, 4*step_helix)*self.monomer_size]
        self.step_helix = step_helix
        self.add_nn_angles = add_nn_angles
        self.add_nth_neigbour_bonds = add_nth_neigbour_bonds
        self.r_zero = r_zero
        if patch_distance is None:
           patch_distance = r_zero / 2.
        self.patch_distance = patch_distance

    def oligomer_head(self, start_index = 0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        insert_helix = self.equilibrium_helix[start_index*self.monomer_size:, :].copy()

        structure_initialiser.add_as_triple_bad_chain(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )

        # make last monomer a head type
        structure_initialiser.atom_types_array[-3] = 11
        structure_initialiser.atom_types_array[-2] = 8
        structure_initialiser.atom_types_array[-1] = 3

        return structure_initialiser

    def oligomer_tail(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        structure_initialiser.add_as_triple_bad_chain(
            self.equilibrium_helix[:], step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds
        )

        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)] = 2
        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)+1] = 7
        structure_initialiser.atom_types_array[-len(self.equilibrium_helix)+2] = 10
        
        return structure_initialiser

    def nmer(self, nmer, structure_initialiser: StructureInitialiser|None = None):
        if nmer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the double BAD topology as it can be expressed as an oligomer.')
        
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        insert_helix = self.equilibrium_helix[:nmer*self.monomer_size, :].copy()
        
        structure_initialiser.add_as_triple_bad_chain(
            insert_helix, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )
        structure_initialiser.atom_types_array[-len(insert_helix)] = 2
        structure_initialiser.atom_types_array[-len(insert_helix)+1] = 7
        structure_initialiser.atom_types_array[-len(insert_helix)+2] = 10
        structure_initialiser.atom_types_array[-3] = 3
        structure_initialiser.atom_types_array[-2] = 8
        structure_initialiser.atom_types_array[-1] = 11

        return structure_initialiser
    
    def straight_dimer(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        
        parallel_line = np.zeros((2*self.monomer_size, 3), dtype=float)
        # straight line along x
        parallel_line[:, 0] = [2, 2, 2, 2+self.r_zero, 2+self.r_zero, 2+self.r_zero]
        # shift between beads in y is 0.5 r_zero
        parallel_line[1::3, 1] = [self.patch_distance, self.patch_distance] # distance of patches from center of helix beads
        parallel_line[2::3, 2] = [self.patch_distance, self.patch_distance] # distance other patches from center of helix beads in other axi

        structure_initialiser.add_as_triple_bad_chain(
            parallel_line, step_ab=self.step_helix, step_bc=self.step_helix,
            step_cd=self.step_helix, add_nn_angles=self.add_nn_angles, add_nth_neigbour_bonds=self.add_nth_neigbour_bonds,
            flexible_create=True
        )

        structure_initialiser.atom_types_array[-6] = 2
        structure_initialiser.atom_types_array[-5] = 7
        structure_initialiser.atom_types_array[-4] = 10
        structure_initialiser.atom_types_array[-3] = 3
        structure_initialiser.atom_types_array[-2] = 8
        structure_initialiser.atom_types_array[-1] = 11

        return structure_initialiser
    
    def fit_string(self, n_beads, n_fit) -> str:
        all_indices = np.arange(1, n_beads+1)
        fit_indices = all_indices[-n_fit*self.monomer_size:]
        return f"2 {' '.join([str(index) for index in fit_indices])} \n"

class TripleBeadPatchFilamentsTopologyFactory(BaseTopologyFactory):
    """Filament whose monomer is 3 collinear backbone beads + 1 explicit patch on the middle bead.

    Particle order per monomer: [b0, b1, b2, patch]  (monomer_size = 4)
    The whole monomer is added / removed in one treadmilling step.

    Types per monomer (all three backbone beads carry the state, the patch follows it):
        bulk monomer: [1, 1, 1, 6]
        tail monomer: [2, 2, 2, 7]
        head monomer: [3, 3, 3, 8]

    Geometry: consecutive backbone beads are r_zero/3 apart, so one monomer spans r_zero.
    Bonds/angles reuse the types of double_filament2d_bad_filament_block.lmp:
        bond 1 backbone, bond 3 bead-patch, bond 4 patch-patch, angle 2 backbone, angle 3 patch.

    Reaction initiators are overridden (on/off/nmer/denucleation) to an end bead and its 1-4 backbone
    neighbour, see on_initiators for why.
    Bonds: backbone (1) and bead-patch (3) only, no patch-patch bonds; one patch angle (patch-b1-b2).
    add_depol_modulation is supported: fragment 2 of the off templates is atom 8, the patch of the second
    monomer from the shrinking end (use with lammps_blocks.TripleBeadTreadmillingBlock for r_off modulation).
    Pol modulation fragments are only implemented for monomer_size 2/3, so keep add_pol_modulation False.
    """
    BULK_TYPES = (1, 1, 1, 6)
    TAIL_TYPES = (2, 2, 2, 7)
    HEAD_TYPES = (3, 3, 3, 8)

    def __init__(self, equilibrium_config, r_zero: float=1.0, patch_distance: float|None=None, n_template_monomers: int=5):
        self.monomer_size: int = 4

        if n_template_monomers < 3:
            raise ValueError('n_template_monomers must be at least 3.')
        if len(equilibrium_config) < n_template_monomers*self.monomer_size:
            raise ValueError(
                f'Please supply a long enough equilibrium configuration: need {n_template_monomers*self.monomer_size} '
                f'particles ({n_template_monomers} monomers of [b0, b1, b2, patch]), got {len(equilibrium_config)}.'
            )

        self.equilibrium_helix = np.asarray(equilibrium_config, dtype=float)[:n_template_monomers*self.monomer_size]
        self.r_zero = r_zero
        self.bead_distance = r_zero / 3.
        if patch_distance is None:
            patch_distance = r_zero / 2.
        self.patch_distance = patch_distance

    def _set_end_types(self, structure_initialiser: StructureInitialiser, n_added: int, tail: bool, head: bool):
        """Set tail types on the first and head types on the last of the n_added most recently added particles."""
        ms = self.monomer_size
        first = len(structure_initialiser.atom_types_array) - n_added
        if tail:
            structure_initialiser.atom_types_array[first:first+ms] = self.TAIL_TYPES
        if head:
            structure_initialiser.atom_types_array[-ms:] = self.HEAD_TYPES

    def oligomer_head(self, start_index = 0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        insert_helix = self.equilibrium_helix[start_index*self.monomer_size:, :].copy()
        structure_initialiser.add_as_triple_bead_patch_chain(insert_helix)

        # make last monomer a head type
        self._set_end_types(structure_initialiser, len(insert_helix), tail=False, head=True)

        return structure_initialiser

    def oligomer_tail(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        insert_helix = self.equilibrium_helix[:, :].copy()
        structure_initialiser.add_as_triple_bead_patch_chain(insert_helix)

        # make first monomer a tail type
        self._set_end_types(structure_initialiser, len(insert_helix), tail=True, head=False)

        return structure_initialiser

    def nmer(self, nmer, structure_initialiser: StructureInitialiser|None = None):
        if nmer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the triple bead patch topology as it can be expressed as an oligomer.')
        if nmer < 2:
            raise ValueError('An nmer needs at least 2 monomers to have a distinct head and tail.')

        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        insert_helix = self.equilibrium_helix[:nmer*self.monomer_size, :].copy()
        structure_initialiser.add_as_triple_bead_patch_chain(insert_helix)

        self._set_end_types(structure_initialiser, len(insert_helix), tail=True, head=True)

        return structure_initialiser

    def straight_dimer(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()

        # straight dimer along x starting at x=2 (seed dimer sits at x=0 and x=1)
        parallel_line = TripleBeadPatchUtils.get_straight_filament(
            n_monomers=2, bead_distance=self.bead_distance,
            patch_distance=self.patch_distance, start=(2., 0., 0.)
        )
        structure_initialiser.add_as_triple_bead_patch_chain(parallel_line)

        self._set_end_types(structure_initialiser, len(parallel_line), tail=True, head=True)

        return structure_initialiser

    def fit_string(self, n_beads, n_fit) -> str:
        all_indices = np.arange(1, n_beads+1)
        fit_indices = all_indices[-n_fit*self.monomer_size:]
        return f"2 {' '.join([str(index) for index in fit_indices])} \n"

    # Initiator choice matters twice in fix bond/react:
    #  1. both initiators must pick each other as their single partner -> each must have exactly one
    #     candidate of the other type at that bond distance;
    #  2. the superimpose algorithm starts from the initiators and has to guess whenever an atom has two
    #     unassigned neighbours of the same type (b1 has b0 and b2). Its backtracking is unreliable and fails
    #     for longer templates (e.g. 4mer -> 5mer growth never happened when starting at the head b1).
    # Starting at an END bead avoids all guesses: walking along the chain, every atom then has at most one
    # unassigned neighbour of each type. So initiators are end beads paired with their 1-4 neighbour
    # (3 bonds along the backbone), which is unique on both sides.
    def on_initiators(self, n_pre_beads: int) -> tuple[int, int]:
        # b2 (last bead) of the head monomer and b2 of the monomer before it (1-4 pair)
        return (n_pre_beads - 1, n_pre_beads - 1 - self.monomer_size)

    def nmer_on_initiators(self, n_mer: int) -> tuple[int, int]:
        return self.on_initiators(n_mer*self.monomer_size)

    def off_initiators(self) -> tuple[int, int]:
        return (1, 1 + self.monomer_size) # b0 (first bead) of the tail monomer and b0 of the next monomer (1-4 pair)

    def denucleation_initiators(self) -> tuple[int, int]:
        return (1, 1 + self.monomer_size) # b0 of the tail and b0 of the head monomer of the dimer (1-4 pair)

    # Edge atoms are the atoms bonded to something outside the oligomer template. bond/react never walks
    # through an edge atom, so marking the whole end monomer as edges (ChemistryWriter default) would make
    # b1, b2 and the patch of that monomer unreachable. Only the single bead bonded outside is an edge.
    def on_edges(self) -> tuple[int, ...]:
        return (1,) # b0 of the first monomer in OligomerOn_pre, bonded to the previous monomer's b2

    def off_edges(self, n_pre_beads: int) -> tuple[int, ...]:
        return (n_pre_beads - 1,) # b2 of the last monomer in OligomerOff_pre, bonded to the next monomer's b0

class TripleBeadPatchDiffuseFilamentsTopologyFactory(TripleBeadPatchFilamentsTopologyFactory):
    """Like TripleBeadPatchFilaments (3 backbone beads + short-range patch on the middle bead), plus a line of
    3 'diffuse' beads, one opposite every backbone bead on the other side than the patch. The backbone beads are
    meant to be purely steric, the diffuse beads carry the long-range diffuse interaction between filaments.

    Particle order per monomer: [b0, b1, b2, patch, d0, d1, d2]  (monomer_size = 7)
    Types per monomer:
        bulk monomer: [1, 1, 1, 6,  9,  9,  9]
        tail monomer: [2, 2, 2, 7, 10, 10, 10]
        head monomer: [3, 3, 3, 8, 11, 11, 11]
    The nucleation seed must therefore use types 12, 13 (seed_type=12).

    Geometry: backbone beads r_zero/3 apart, patch at +patch_distance (y) from b1, d_i at -diffuse_distance (y)
    from b_i. Bonds 1 (backbone), 3 (b1-patch), 5 (b_i-d_i); angles 2 (backbone), 3 (patch-b1-b2),
    4 (d0-b0-b1, d1-b1-b2, d2-b2-b1), i.e. double_filament2d_2patches_bad_filament_block.lmp.

    Initiators/edges follow TripleBeadPatchFilaments (end bead + its 1-4 backbone neighbour, which stays the
    only type-1 1-4 neighbour because diffuse beads are types 9-11 and not bonded to each other).
    add_depol_modulation: fragment 2 of the off templates is the type-6 patch of the second monomer (atom 11).

    Edge trimming: in the oligomer growth/shrink templates (OligomerOn/OligomerOff) the edge atom is a backbone
    end bead (b0 of the first / b2 of the last template monomer). Its diffuse bead is bonded only to that edge
    atom, and bond/react never walks through edge atoms, so it could never be matched ('Excessive iteration of
    superimpose algorithm'). That diffuse bead is therefore left out of those templates (oligomer_head drops d0
    of the first monomer, oligomer_tail drops d2 of the last). It stays in the simulation untouched, it is
    simply not part of the reaction. nmer/nucleation templates are whole filaments and keep all atoms.
    """
    BULK_TYPES = (1, 1, 1, 6, 9, 9, 9)
    TAIL_TYPES = (2, 2, 2, 7, 10, 10, 10)
    HEAD_TYPES = (3, 3, 3, 8, 11, 11, 11)

    def __init__(self, equilibrium_config, r_zero: float=1.0, patch_distance: float|None=None, diffuse_distance: float|None=None, n_template_monomers: int=5):
        self.monomer_size: int = 7

        if n_template_monomers < 3:
            raise ValueError('n_template_monomers must be at least 3.')
        if len(equilibrium_config) < n_template_monomers*self.monomer_size:
            raise ValueError(
                f'Please supply a long enough equilibrium configuration: need {n_template_monomers*self.monomer_size} '
                f'particles ({n_template_monomers} monomers of [b0, b1, b2, patch, d0, d1, d2]), got {len(equilibrium_config)}.'
            )

        self.equilibrium_helix = np.asarray(equilibrium_config, dtype=float)[:n_template_monomers*self.monomer_size]
        self.r_zero = r_zero
        self.bead_distance = r_zero / 3.
        self.patch_distance = r_zero / 2. if patch_distance is None else patch_distance
        self.diffuse_distance = r_zero / 2. if diffuse_distance is None else diffuse_distance

    @staticmethod
    def _add_chain(structure_initialiser: StructureInitialiser, positions):
        structure_initialiser.add_as_triple_bead_patch_diffuse_chain(positions)

    @staticmethod
    def _remove_atom(structure_initialiser: StructureInitialiser, index: int):
        """Remove atom index (1-based) with all its bonds/angles/dihedrals and renumber the rest."""
        keep = np.arange(structure_initialiser.n_beads) != index - 1
        structure_initialiser.positions = structure_initialiser.positions[keep]
        structure_initialiser.atom_types_array = structure_initialiser.atom_types_array[keep]
        structure_initialiser.molecule_ids = structure_initialiser.molecule_ids[keep]
        for name in ('bond_array', 'angle_array', 'dihedral_array'):
            topology = getattr(structure_initialiser, name)
            if not len(topology):
                continue
            topology = topology[~np.any(topology[:, 2:] == index, axis=1)].copy()
            atoms = topology[:, 2:]
            atoms[atoms > index] -= 1
            topology[:, 2:] = atoms
            topology[:, 0] = np.arange(1, len(topology) + 1)
            setattr(structure_initialiser, name, topology)

    def oligomer_head(self, start_index = 0, structure_initialiser: StructureInitialiser | None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        insert_helix = self.equilibrium_helix[start_index*self.monomer_size:, :].copy()
        first = structure_initialiser.n_beads + 1
        self._add_chain(structure_initialiser, insert_helix)
        self._set_end_types(structure_initialiser, len(insert_helix), tail=False, head=True)
        self._remove_atom(structure_initialiser, first + 4) # d0 of the first monomer, hangs on the edge atom b0
        return structure_initialiser

    def oligomer_tail(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        insert_helix = self.equilibrium_helix[:, :].copy()
        self._add_chain(structure_initialiser, insert_helix)
        self._set_end_types(structure_initialiser, len(insert_helix), tail=True, head=False)
        self._remove_atom(structure_initialiser, structure_initialiser.n_beads) # d2 of the last monomer, hangs on the edge atom b2
        return structure_initialiser

    def nmer(self, nmer, structure_initialiser: StructureInitialiser|None = None):
        if nmer > self.equilibrium_helix.shape[0]//self.monomer_size:
            raise ValueError('The nmer you are trying to create is not available with the triple bead patch diffuse topology as it can be expressed as an oligomer.')
        if nmer < 2:
            raise ValueError('An nmer needs at least 2 monomers to have a distinct head and tail.')
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        insert_helix = self.equilibrium_helix[:nmer*self.monomer_size, :].copy()
        self._add_chain(structure_initialiser, insert_helix)
        self._set_end_types(structure_initialiser, len(insert_helix), tail=True, head=True)
        return structure_initialiser

    def straight_dimer(self, structure_initialiser: StructureInitialiser|None = None):
        if structure_initialiser is None:
            structure_initialiser = StructureInitialiser()
        # straight dimer along x starting at x=2 (seed dimer sits at x=0 and x=1)
        parallel_line = TripleBeadPatchDiffuseUtils.get_straight_filament(
            n_monomers=2, bead_distance=self.bead_distance,
            patch_distance=self.patch_distance, diffuse_distance=self.diffuse_distance, start=(2., 0., 0.)
        )
        self._add_chain(structure_initialiser, parallel_line)
        self._set_end_types(structure_initialiser, len(parallel_line), tail=True, head=True)
        return structure_initialiser

    def _b2_of_last_monomer(self, n_beads: int) -> int:
        return n_beads - self.monomer_size + 3 # 1-based index of b2 in [b0, b1, b2, patch, d0, d1, d2]

    def on_initiators(self, n_pre_beads: int) -> tuple[int, int]:
        # b2 of the head monomer and b2 of the monomer before it (1-4 pair along the backbone)
        head_b2 = self._b2_of_last_monomer(n_pre_beads)
        return (head_b2, head_b2 - self.monomer_size)

    def off_edges(self, n_pre_beads: int) -> tuple[int, ...]:
        # b2 of the last monomer (bonded to the next monomer's b0); its d2 was removed by oligomer_tail
        return (self._b2_of_last_monomer(n_pre_beads + 1),)

    def depol_fragment_atom(self) -> int:
        return self.monomer_size + 4 # type-6 patch of the second monomer from the shrinking end

class ChemistryWriter():
    def __init__(self, 
        equilibrium_config: np.ndarray,
        mode='BAD',
        peratom_masses: bool = False,
        seed_type: int = 4,
        seed_mode: str = 'dimer',
        **kwargs,
    ):
        if mode not in CHEMISTRY_MODES:
            raise ValueError(f'Chemistry mode must be one of {CHEMISTRY_MODES}, got {mode}.')

        if mode == 'BAD':
            self.topology_factory = BADTopologyFactory(
                equilibrium_config,
                **kwargs,
            )
        
        elif mode == 'ESCRT':
            self.topology_factory = ESCRTTopologyFactory(
                equilibrium_config,
                **kwargs,
            )
            
        elif mode == 'ESCRTDoublePatch':
            self.topology_factory = ESCRTDoublePatchTopologyFactory(
                equilibrium_config,
                **kwargs,
            )

        elif mode == 'DoubleBAD':
            self.topology_factory = DoubleBADTopologyFactory(
                equilibrium_config,
                **kwargs,
            )
            
        elif mode == 'DoubleFilaments':
            self.topology_factory = DoubleBADDoubleFilamentsTopologyFactory(
                equilibrium_config,
                **kwargs,
            )
            
        elif mode == 'DoubleFilamentsTwoPatches':
            self.topology_factory = DoubleBADDoubleFilamentsTwoPatchesTopologyFactory(
                equilibrium_config,
                **kwargs,
            )

        elif mode == 'TripleBAD':
            self.topology_factory = TripleBADTopologyFactory(
                equilibrium_config,
                **kwargs,
            )

        elif mode == 'TripleBeadPatchFilaments':
            self.topology_factory = TripleBeadPatchFilamentsTopologyFactory(
                equilibrium_config,
                **kwargs,
            )

        elif mode == 'TripleBeadPatchDiffuseFilaments':
            self.topology_factory = TripleBeadPatchDiffuseFilamentsTopologyFactory(
                equilibrium_config,
                **kwargs,
            )

        self.peratom_masses = peratom_masses

        # determine seed properties for nucleation
        self.seed_type = seed_type
        if seed_mode not in SEED_MODES:
            raise ValueError(f'Seed mode must be one of {SEED_MODES}, got {seed_mode}.')
        self.seed_mode = seed_mode
        if seed_mode == 'dimer':
            self.seed_size = 2
        elif seed_mode == 'trimer':
            self.seed_size = 3

    @property
    def max_nmer(self):
        return self.topology_factory.max_nmer

    @staticmethod
    def _topology_sorter(new_index_array: np.ndarray, target_array: np.ndarray):
        has_new = np.array([np.isin(new_index_array, target_array[row, 2:]).any() for row in range(target_array.shape[0])], dtype=bool)
        return_array = np.vstack((target_array[~has_new, :], target_array[has_new, :]))
        return_array[:, 0] = np.arange(1, return_array.shape[0]+1)

        return return_array

    @staticmethod
    def new_topology_last(structure_initialiser: StructureInitialiser, n_start: int, monomer_size: int) -> StructureInitialiser:
        new_index_array = np.atleast_1d(np.arange(n_start + 1, n_start + monomer_size + 1))

        structure_initialiser.bond_array = ChemistryWriter._topology_sorter(new_index_array, structure_initialiser.bond_array)
        structure_initialiser.angle_array = ChemistryWriter._topology_sorter(new_index_array, structure_initialiser.angle_array)
        structure_initialiser.dihedral_array = ChemistryWriter._topology_sorter(new_index_array, structure_initialiser.dihedral_array)
        
        return structure_initialiser

    def write_on_pre(self, target_file_path: Path, type_to_add: int=1, add_pol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, write_two_oligomer_files: bool=False, n_fit: int|None=None,enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()

        if write_two_oligomer_files:
            structure_initialiser = self.topology_factory.oligomer_head(start_index=1,type_to_add = type_to_add)
        else:
            structure_initialiser = self.topology_factory.oligomer_head(start_index=1)
            
        if add_pol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches:
            n_fragments = 4
        elif add_pol_modulation and parallel_filament:
            n_fragments = 5
        elif add_pol_modulation and parallel_and_antiparallel:
            n_fragments = 5
        elif add_pol_modulation and antiparallel_filaments_modulate_both_patches:
            n_fragments = 5
        else:
            n_fragments = 3
            
        if n_fit is None:
            n_fit = structure_initialiser.n_beads
            
        if enable_templates_for9_and_14:
            atom_type_list = self.topology_factory.get_all_combinations_of_type_9_and_14(structure_initialiser=structure_initialiser)
            for idx, atom_types in enumerate(atom_type_list):
                structure_initialiser.atom_types_array = atom_types
                target_file_path_with_idx = target_file_path.with_name(f"{target_file_path.stem}_template_{idx+1}{target_file_path.suffix}").resolve()
                structure_initialiser.write_to_template_file(target_file_path_with_idx, n_fragments=n_fragments)
                
                with open(target_file_path_with_idx, 'a') as f:
                    f.write(f"\n\nFragments\n\n")
                    f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                    f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
                    f.write(f"3 {structure_initialiser.n_beads} \n")
                    if add_pol_modulation:
                        if self.topology_factory.monomer_size == 2:
                            f.write(f"4 {structure_initialiser.n_beads-2} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                        elif self.topology_factory.monomer_size == 3:
                            f.write(f"4 {structure_initialiser.n_beads-4} \n")
                    if add_pol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches):
                        if self.topology_factory.monomer_size == 2:
                            f.write(f"5 {structure_initialiser.n_beads-1} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                        elif self.topology_factory.monomer_size == 3:
                            f.write(f"5 {structure_initialiser.n_beads-3} \n")
                        
                    f.write(f"\n\nMolecules\n\n")
                    for ii in range(structure_initialiser.n_beads):
                        f.write(f"{ii+1} 1\n")

        else:

            structure_initialiser.write_to_template_file(target_file_path, n_fragments=n_fragments)


            with open(target_file_path, 'a') as f:
                f.write(f"\n\nFragments\n\n")
                f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
                f.write(f"3 {structure_initialiser.n_beads} \n")
                if add_pol_modulation:
                    if self.topology_factory.monomer_size == 2:
                        f.write(f"4 {structure_initialiser.n_beads-2} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                    elif self.topology_factory.monomer_size == 3:
                        f.write(f"4 {structure_initialiser.n_beads-4} \n")
                if add_pol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches):
                    if self.topology_factory.monomer_size == 2:
                        f.write(f"5 {structure_initialiser.n_beads-1} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                    elif self.topology_factory.monomer_size == 3:
                        f.write(f"5 {structure_initialiser.n_beads-3} \n")
                    
                f.write(f"\n\nMolecules\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1\n")

    def write_on_post(self, target_file_path: Path, type_to_add: int=1, write_two_oligomer_files: bool=False, n_fit: int|None=None, enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()
        
        if write_two_oligomer_files:
            structure_initialiser = self.topology_factory.oligomer_head(start_index=0,type_to_add = type_to_add)
        else:
            structure_initialiser = self.topology_factory.oligomer_head(start_index=0)

        #structure_initialiser = self.topology_factory.oligomer_head(start_index=0)
        ChemistryWriter.new_topology_last(
            structure_initialiser,
            n_start=structure_initialiser.n_beads - self.topology_factory.monomer_size,
            monomer_size=self.topology_factory.monomer_size
        )
        
        if n_fit is None:
            n_fit = structure_initialiser.n_beads//self.topology_factory.monomer_size
        if write_two_oligomer_files:
            n_fit+=1

        if enable_templates_for9_and_14:
            struc_intialiser_pre  = self.topology_factory.oligomer_head(start_index=1,type_to_add = type_to_add)
            atom_type_list = self.topology_factory.get_all_combinations_of_type_9_and_14(structure_initialiser=struc_intialiser_pre)
            for idx, atom_types in enumerate(atom_type_list):
                structure_initialiser.atom_types_array[:-self.topology_factory.monomer_size*2] = atom_types[:-self.topology_factory.monomer_size]
                target_file_path_with_idx = target_file_path.with_name(f"{target_file_path.stem}_template_{idx+1}{target_file_path.suffix}").resolve()
                structure_initialiser.write_to_template_file(target_file_path_with_idx, n_fragments=2)
                
                with open(target_file_path_with_idx, 'a') as f:
                    if self.peratom_masses:
                        f.write(f"\n\nMasses\n\n")
                        for ii in range(structure_initialiser.n_beads):
                            f.write(f"{ii+1} 1.0\n")

                    f.write("\n\nCoords\n\n")
                    for ii, coord in enumerate(structure_initialiser.positions):
                        f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

                    f.write(f"\n\nMolecules\n\n")
                    for ii in range(structure_initialiser.n_beads):
                        f.write(f"{ii+1} 1\n")

                    f.write(f"\n\nFragments\n\n")
                    f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                    f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))

        else:
            structure_initialiser.write_to_template_file(target_file_path, n_fragments=2)
        
            with open(target_file_path, 'a') as f:
                if self.peratom_masses:
                    f.write(f"\n\nMasses\n\n")
                    for ii in range(structure_initialiser.n_beads):
                        f.write(f"{ii+1} 1.0\n")

                f.write("\n\nCoords\n\n")
                for ii, coord in enumerate(structure_initialiser.positions):
                    f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

                f.write(f"\n\nMolecules\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1\n")

                f.write(f"\n\nFragments\n\n")
                f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))

    def write_surface_on_pre(self, target_file_path: Path, surface_type: int=5, n_fit: int|None=None):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = StructureInitialiser()
        
        # prepend a membrane particle
        structure_initialiser.positions = np.append([[0.0, 0.0, 0.0]], structure_initialiser.positions, axis=0)
        structure_initialiser.atom_types_array = np.append([surface_type], structure_initialiser.atom_types_array, axis=0)

        structure_initialiser = self.topology_factory.oligomer_head(start_index=1, structure_initialiser=structure_initialiser)

        structure_initialiser.write_to_template_file(target_file_path, n_fragments=2)

        if n_fit is None:
            n_fit = (structure_initialiser.n_beads - 1)//self.topology_factory.monomer_size

        with open(target_file_path, 'a') as f: 
            f.write(f"\n\nFragments\n\n")
            f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
            f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit)) # account for membrane particle here
            # f.write(f"2 {' '.join([str(ii) for ii in range(2, structure_initialiser.n_beads+1)][-n_fit:])}")

            f.write(f"\n\nMolecules\n\n")
            f.write("1 0\n") # TODO: add a molecule type for the surface particles
            for ii in range(1, structure_initialiser.n_beads):
                f.write(f"{ii+1} 1\n")

    def write_surface_on_post(self, target_file_path: Path, d_surface: float=1.0, surface_type: int=5, com: np.ndarray|None=None, n_fit: int|None=None):
        # the center of mass of the helix is assumed to be zero, from it the surface bead is generated.
        target_file_path = Path(target_file_path).resolve()

        if com is None:
            com = np.zeros(3, dtype=float)

        structure_initialiser = StructureInitialiser()
        equilibrium_helix = self.topology_factory.oligomer_head(start_index=0).positions

        # first add a single surface particle to the structure initialiser
        vec_last_bead = equilibrium_helix[-1, :] - com
        vec_last_bead[:] = vec_last_bead[:] / np.linalg.norm(vec_last_bead)
        surface_pos = equilibrium_helix[-1, :] + (vec_last_bead * d_surface)

        structure_initialiser.positions = np.append(structure_initialiser.positions, surface_pos[np.newaxis, :], axis=0)
        structure_initialiser.atom_types_array = np.append(structure_initialiser.atom_types_array, [surface_type], axis=0)

        structure_initialiser = self.topology_factory.oligomer_head(start_index=0, structure_initialiser=structure_initialiser)

        structure_initialiser.write_to_template_file(target_file_path, n_fragments=2)

        if n_fit is None:
            n_fit = (structure_initialiser.n_beads - 1)//self.topology_factory.monomer_size

        with open(target_file_path, 'a') as f:
            if self.peratom_masses:
                f.write(f"\n\nMasses\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1.0\n")

            f.write("\n\nCoords\n\n")
            for ii, coord in enumerate(structure_initialiser.positions):
                f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

            f.write(f"\n\nMolecules\n\n")
            f.write("1 0\n") # TODO: add a molecule type for the surface particles
            for ii in range(1, structure_initialiser.n_beads):
                f.write(f"{ii+1} 1\n")

            f.write(f"\n\nFragments\n\n")
            f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
            f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
            # f.write(f"2 {' '.join([str(ii) for ii in range(2, structure_initialiser.n_beads+1)][-(n_fit+1):])}")

    def write_surface_on_map(self, target_file_path: Path, surface_type: int=5):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = StructureInitialiser()

        # prepend a membrane particle
        structure_initialiser.positions = np.append([[0.0, 0.0, 0.0]], structure_initialiser.positions, axis=0)
        structure_initialiser.atom_types_array = np.append([surface_type], structure_initialiser.atom_types_array, axis=0)

        structure_initialiser = self.topology_factory.oligomer_head(start_index=0, structure_initialiser=structure_initialiser)

        map_file = MapFile(
            equivalences=tuple(zip(
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1),
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1),
            )),
            initiators=(1, structure_initialiser.n_beads-self.topology_factory.monomer_size), # membrane particle and last helix particle
            edges=tuple(range(2, 2+self.topology_factory.monomer_size)), # first particle is membrane, therefore start from 2
            creates=tuple(range(
                1+structure_initialiser.n_beads-self.topology_factory.monomer_size, 1+structure_initialiser.n_beads
            )),
            constraints=('custom "rxnsum(v_kon,1) > random(0,1,v_vSeed)"',)
        )

        map_file.write_to_file(target_file_path)

    def write_on_map(self, target_file_path: Path, type_to_add:int =1, write_two_oligomer_files: bool=False, add_pol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, kon_premod: float=0.0, kon_postmod: float=0.0, reaction_time: float=0.0, add_volume_exclusion: bool=False):
        target_file_path = Path(target_file_path).resolve()
        
        if write_two_oligomer_files:
            structure_initialiser = self.topology_factory.oligomer_head(start_index=0,type_to_add = type_to_add)
        else:
            structure_initialiser = self.topology_factory.oligomer_head(start_index=0)
        
        # initiators here are either last two beads or 2 beads from last monomer
        # this breaks if other indices are used, no idea why
        if self.topology_factory.monomer_size == 1:
            initiators = (structure_initialiser.n_beads-2, structure_initialiser.n_beads-1)
        else:
            initiators = (structure_initialiser.n_beads-2*self.topology_factory.monomer_size+1, structure_initialiser.n_beads-2*self.topology_factory.monomer_size+2) #second one was +2 not +3
        custom_initiators = self.topology_factory.on_initiators(structure_initialiser.n_beads - self.topology_factory.monomer_size)
        if custom_initiators is not None:
            initiators = custom_initiators

        if add_pol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches:
            constraints = 'custom "rxnsum(v_kon,4) > random(0,1,v_vSeed)"'
        elif add_pol_modulation and parallel_filament:
            constraints = f'custom "(({kon_postmod}-{kon_premod})*{reaction_time}*(rxnsum(v_neigh9,4)+rxnsum(v_neigh6,5)-rxnsum(v_neigh9,4)*rxnsum(v_neigh6,5))+rxnsum(v_konin,1))*rxnsum(v_expfactor,1) > random(0,1,v_vSeed)"'
        elif add_pol_modulation and antiparallel_filaments_modulate_both_patches:
            constraints = f'custom "(({kon_postmod}-{kon_premod})*{reaction_time}*(rxnsum(v_neigh9,5)+rxnsum(v_neigh6,4)-rxnsum(v_neigh9,5)*rxnsum(v_neigh6,4))+rxnsum(v_konin,1))*rxnsum(v_expfactor,1) > random(0,1,v_vSeed)"'
        elif add_pol_modulation and parallel_and_antiparallel:
            constraints = f'custom "(({kon_postmod}-{kon_premod})*{reaction_time}*(1 - (1-rxnsum(v_neigh9,4))*(1-rxnsum(v_neigh6,4))*(1-rxnsum(v_neigh6,5))*(1-rxnsum(v_neigh9,5)))+rxnsum(v_konin,1))*rxnsum(v_expfactor,1) > random(0,1,v_vSeed)"'
        else:
            constraints = 'custom "rxnsum(v_kon,1) > random(0,1,v_vSeed)"'
            
        
        final_constraints = (constraints,)
            
        if write_two_oligomer_files:
            edges = tuple([1])
        else:
            edges = tuple(range(1, 1+self.topology_factory.monomer_size))
        custom_edges = self.topology_factory.on_edges()
        if custom_edges is not None and not write_two_oligomer_files:
            edges = custom_edges

        map_file = MapFile(
            equivalences=tuple(zip(
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1),
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1),
            )),
            initiators=initiators,
            edges=edges,
            creates=tuple(range(
                1+structure_initialiser.n_beads-self.topology_factory.monomer_size, 1+structure_initialiser.n_beads
            )),
            constraints=final_constraints
        )

        map_file.write_to_file(target_file_path)

    def write_off_pre(self, target_file_path: Path, type_to_add:int=1, write_two_oligomer_files: bool=False, add_depol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()
        
        if add_depol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches:
            n_fragments = 2
        elif add_depol_modulation and parallel_filament:
            n_fragments = 3
        elif add_depol_modulation and parallel_and_antiparallel:
            n_fragments = 3
        elif add_depol_modulation and antiparallel_filaments_modulate_both_patches:
            n_fragments = 3
        else:
            n_fragments = 1

        if write_two_oligomer_files:
            structure_initialiser = self.topology_factory.oligomer_tail(type_to_add)
        else:
            structure_initialiser = self.topology_factory.oligomer_tail()
            
        if enable_templates_for9_and_14:
            atom_type_list = self.topology_factory.get_all_combinations_of_type_9_and_14(structure_initialiser=structure_initialiser)
            for idx, atom_types in enumerate(atom_type_list):
                structure_initialiser.atom_types_array = atom_types
                target_file_path_with_idx = target_file_path.with_name(f"{target_file_path.stem}_template_{idx+1}{target_file_path.suffix}").resolve()
                structure_initialiser.write_to_template_file(target_file_path_with_idx, n_fragments=n_fragments)

                with open(target_file_path_with_idx, 'a') as f:
                    f.write(f"\n\nFragments\n\n")
                    f.write(f"1 1\n")
                    if add_depol_modulation:
                        if self.topology_factory.monomer_size == 2:
                            f.write(f"2 4 \n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                        elif self.topology_factory.monomer_size == 3:
                            f.write(f"2 5\n")  # adding a fragment for polymer modulation - the patch on the shrinking end # adding a fragment for polymer modulation - the patch on the shrinking end
                        elif self.topology_factory.monomer_size == 4:
                            f.write(f"2 8\n")  # TripleBeadPatchFilaments: patch (atom 8) of the second monomer from the shrinking end
                        elif self.topology_factory.depol_fragment_atom() is not None:
                            f.write(f"2 {self.topology_factory.depol_fragment_atom()}\n")  # e.g. TripleBeadPatchDiffuseFilaments
                    if add_depol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches):
                        if self.topology_factory.monomer_size == 2:
                            f.write(f"3 5 \n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                        elif self.topology_factory.monomer_size == 3:
                            f.write(f"3 6\n")  # adding a fragment for polymer modulation - the patch on the shrinking end # adding a fragment for polymer modulation - the patch on the shrinking end
        
        else:

            structure_initialiser.write_to_template_file(target_file_path, n_fragments=n_fragments)

            with open(target_file_path, 'a') as f:
                f.write(f"\n\nFragments\n\n")
                f.write(f"1 1\n")
                if add_depol_modulation:
                    if self.topology_factory.monomer_size == 2:
                        f.write(f"2 4 \n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                    elif self.topology_factory.monomer_size == 3:
                        f.write(f"2 5\n")  # adding a fragment for polymer modulation - the patch on the shrinking end # adding a fragment for polymer modulation - the patch on the shrinking end
                    elif self.topology_factory.monomer_size == 4:
                        f.write(f"2 8\n")  # TripleBeadPatchFilaments: patch (atom 8) of the second monomer from the shrinking end
                    elif self.topology_factory.depol_fragment_atom() is not None:
                        f.write(f"2 {self.topology_factory.depol_fragment_atom()}\n")  # e.g. TripleBeadPatchDiffuseFilaments
                if add_depol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches):
                    if self.topology_factory.monomer_size == 2:
                        f.write(f"3 5 \n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                    elif self.topology_factory.monomer_size == 3:
                        f.write(f"3 6\n")  # adding a fragment for polymer modulation - the patch on the shrinking end # adding a fragment for polymer modulation - the patch on the shrinking end
            

    def write_off_post(self, target_file_path: Path, type_to_add: int=1, write_two_oligomer_files: bool=False, enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()

        if write_two_oligomer_files:
            structure_initialiser = self.topology_factory.oligomer_tail(type_to_add= type_to_add)
            structure_initialiser = self.topology_factory.shrink_tail(structure_initialiser)
        else:
            structure_initialiser = self.topology_factory.oligomer_tail()
            structure_initialiser = self.topology_factory.shrink_tail(structure_initialiser)

        if enable_templates_for9_and_14:
            structure_initialiser_pre = self.topology_factory.oligomer_tail(type_to_add)
            atom_type_list = self.topology_factory.get_all_combinations_of_type_9_and_14(structure_initialiser=structure_initialiser_pre)
            for idx, atom_types in enumerate(atom_type_list):
                structure_initialiser.atom_types_array[self.topology_factory.monomer_size*2+1:] = atom_types[self.topology_factory.monomer_size*2+1:]
                target_file_path_with_idx = target_file_path.with_name(f"{target_file_path.stem}_template_{idx+1}{target_file_path.suffix}").resolve()
                structure_initialiser.write_to_template_file(target_file_path_with_idx, n_fragments=0)
        else:
            structure_initialiser.write_to_template_file(target_file_path, n_fragments=0)

    def write_off_map(self, target_file_path: Path, type_to_add: int=1, add_depol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, thyd_premod: float=0.0, thyd_postmod: float=0.0, write_two_oligomer_files: bool=False, explicit_hydrolysis: bool=False, r_off: float=0.0, r_off_modulation: float=0.0,reaction_time: float=0.0):
        target_file_path = Path(target_file_path).resolve()

        if write_two_oligomer_files:
            structure_initialiser = self.topology_factory.oligomer_tail(type_to_add = type_to_add)
            edges = tuple([structure_initialiser.n_beads])
        else:            
            structure_initialiser = self.topology_factory.oligomer_tail()
            edges = tuple(range(1 + structure_initialiser.n_beads - self.topology_factory.monomer_size, 1 + structure_initialiser.n_beads))
            custom_edges = self.topology_factory.off_edges(structure_initialiser.n_beads)
            if custom_edges is not None:
                edges = custom_edges
        
        if explicit_hydrolysis:
            if add_depol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches:
                first_constraint = 'custom "rxnsum(v_koff, 2) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and parallel_filament:
                first_constraint = f'custom "(({r_off_modulation}-{r_off})*{reaction_time}*(rxnsum(v_neigh9,2)+rxnsum(v_neigh6,3)-rxnsum(v_neigh9,2)*rxnsum(v_neigh6,3))+{r_off}*{reaction_time}) > random(0,1,v_vSeed)"' 
            elif add_depol_modulation and antiparallel_filaments_modulate_both_patches:
                first_constraint = f'custom "(({r_off_modulation}-{r_off})*{reaction_time}*(rxnsum(v_neigh9,3)+rxnsum(v_neigh6,2)-rxnsum(v_neigh9,3)*rxnsum(v_neigh6,2))+{r_off}*{reaction_time}) > random(0,1,v_vSeed)"' 
            elif add_depol_modulation and parallel_and_antiparallel:
                first_constraint = f'custom "(({r_off_modulation}-{r_off})*{reaction_time}*(1- (1-rxnsum(v_neigh9,2))*(1-rxnsum(v_neigh6,3))*(1-rxnsum(v_neigh6,2))*(1-rxnsum(v_neigh9,3)))+{r_off}*{reaction_time}) > random(0,1,v_vSeed)"'
            else:
                first_constraint = 'custom "rxnsum(v_koff, 1) > random(0,1,v_vSeed)"'
        else:
            if add_depol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches:
                first_constraint = 'custom "rxnsum(v_koff, 2) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and parallel_filament:
                first_constraint = f'custom "1.0-exp(-rxnsum(v_vSA,2)/((({thyd_postmod}-{thyd_premod})*(rxnsum(v_neigh9,2)+rxnsum(v_neigh6,3)-rxnsum(v_neigh9,2)*rxnsum(v_neigh6,3))+{thyd_premod})/v_tstep)) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and antiparallel_filaments_modulate_both_patches:
                first_constraint = f'custom "1.0-exp(-rxnsum(v_vSA,2)/((({thyd_postmod}-{thyd_premod})*(rxnsum(v_neigh9,3)+rxnsum(v_neigh6,2)-rxnsum(v_neigh9,3)*rxnsum(v_neigh6,2))+{thyd_premod})/v_tstep)) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and parallel_and_antiparallel:
                first_constraint = f'custom "1.0-exp(-rxnsum(v_vSA,2)/((({thyd_postmod}-{thyd_premod})*(1- (1-rxnsum(v_neigh9,2))*(1-rxnsum(v_neigh6,3))*(1-rxnsum(v_neigh6,2))*(1-rxnsum(v_neigh9,3)))+{thyd_premod})/v_tstep)) > random(0,1,v_vSeed)"'
            else:
                first_constraint = 'custom "rxnsum(v_koff, 1) > random(0,1,v_vSeed)"'

        map_file = MapFile(
            equivalences=tuple(zip(
                range(1, structure_initialiser.n_beads+1),
                range(1, structure_initialiser.n_beads+1)
            )),
            initiators=self.topology_factory.off_initiators() or (1, 1 + self.topology_factory.monomer_size),
            edges=edges,
            deletes=tuple(range(1, 1 + self.topology_factory.monomer_size)),
            constraints=(
                first_constraint,
                'custom "rxnsum(v_vIsHydrolysed, 1)"',
            )
        )

        map_file.write_to_file(target_file_path)
        
    def write_secnuc_pre(self, target_file_path: Path):
        target_file_path = Path(target_file_path).resolve()
        
        n_fragments = 2
        structure_initialiser = self.topology_factory.secnuc()

        structure_initialiser.write_to_template_file(target_file_path, n_fragments=n_fragments)

        with open(target_file_path, 'a') as f:
            f.write(f"\n\nFragments\n\n")
            f.write(f"1 1\n")
            if self.topology_factory.monomer_size == 2:
                f.write(f"2 9\n")
            elif self.topology_factory.monomer_size == 3:
                f.write(f"2 11\n")
                
            f.write(f"\n\nMolecules\n\n")
            for ii in range(structure_initialiser.n_beads):
                f.write(f"{ii+1} 1\n")
            
    def write_secnuc_post(self, target_file_path: Path):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = self.topology_factory.secnuc_with_nucleated_trimer()
        
        structure_initialiser.write_to_template_file(target_file_path, n_fragments=2)
        
        with open(target_file_path, 'a') as f:
            if self.peratom_masses:
                f.write(f"\n\nMasses\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1.0\n")

            f.write("\n\nCoords\n\n")
            for ii, coord in enumerate(structure_initialiser.positions):
                f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

            f.write(f"\n\nMolecules\n\n")
            for ii in range(structure_initialiser.n_beads - self.topology_factory.monomer_size*3):
                f.write(f"{ii+1} 1\n")
            for jj in range(self.topology_factory.monomer_size*3):
                f.write(f"{ii+jj+2} 2\n")

            f.write(f"\n\nFragments\n\n")
            f.write(f"1 {structure_initialiser.n_beads-3*self.topology_factory.monomer_size-1}\n")
            f.write(self.topology_factory.fit_string_secnuc(structure_initialiser.n_beads, n_fit=3))
            
    def write_secnuc_map(self, target_file_path: Path ):
        target_file_path = Path(target_file_path).resolve()
        
        structure_initialiser = self.topology_factory.secnuc_with_nucleated_trimer()
        
        # initiators here are either last two beads or 2 beads from last monomer
        # this breaks if other indices are used, no idea why
        initiators = (5,6) 

        constraint1 = 'custom "rxnsum(v_ksecnuc,1) > random(0,1,v_vSeed)"'
        constraint2 = 'custom "rxnsum(v_vSA,1) > rxnsum(v_vSA,2)"' #ensures antiparallel arrangement between nucleated filament and existing filament
        
        final_constraints = (constraint1,constraint2)
            
        if self.topology_factory.monomer_size == 2:
            edges = tuple(np.append([1,2],[self.topology_factory.monomer_size*5-1, self.topology_factory.monomer_size*5])) # first two beads can bind to each other, then the next four are the ones that create the new trimer
            creates=tuple(range(1+self.topology_factory.monomer_size*5, 1+structure_initialiser.n_beads))
            equivalences=tuple(zip(range(1, self.topology_factory.monomer_size*5+1),range(1, self.topology_factory.monomer_size*5+1),))
        elif self.topology_factory.monomer_size == 3:
            edges = tuple(np.append([1],[self.topology_factory.monomer_size*3+2])) # first bead can bind to each other, then the next six are the ones that create the new trimer
            creates = tuple(range(self.topology_factory.monomer_size*3+3, 1+structure_initialiser.n_beads))
            equivalences=tuple(zip(range(1, self.topology_factory.monomer_size*3+3),range(1, self.topology_factory.monomer_size*3+3),))
        
        
        map_file = MapFile(
            equivalences=equivalences,
            initiators=initiators,
            edges=edges,
            creates=creates,
            constraints=final_constraints
        )

        map_file.write_to_file(target_file_path)


    def write_nuc_map(self, target_file_path: Path):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = StructureInitialiser()
        if self.seed_mode == 'dimer':
            structure_initialiser.add_seed_dimer(self.seed_type)
        elif self.seed_mode == 'trimer':
            structure_initialiser.add_seed_trimer(self.seed_type)

        structure_initialiser = self.topology_factory.straight_dimer(structure_initialiser=structure_initialiser)

        map_file = MapFile(
            equivalences=tuple(zip(range(1, 1 + self.seed_size), range(1, 1 + self.seed_size))),
            initiators=(1, 2), # two of the nucleator atoms
            edges=(),
            creates=tuple(range(
                1 + self.seed_size, 1 + self.seed_size + 2 * self.topology_factory.monomer_size # creates all atoms after seed atoms
            )),
            constraints=('custom "rxnsum(v_knuc,1) > random(0,1,v_vSeed)"',)
        )

        map_file.write_to_file(target_file_path)

    def write_nuc_pre(self, target_file_path: Path):
        structure_initialiser = StructureInitialiser()
        if self.seed_mode == 'dimer':
            structure_initialiser.add_seed_dimer(self.seed_type)
        elif self.seed_mode == 'trimer':
            structure_initialiser.add_seed_trimer(self.seed_type)

        structure_initialiser.write_to_template_file(target_file_path, n_fragments=1)

        with open(target_file_path, 'a') as f:
            f.write(f"\n\nFragments\n\n")
            f.write(f"1 2\n") # use one of the seed atoms to extract nucleation rate from?

            f.write(f"\n\nMolecules\n\n")
            for ii in range(0, self.seed_size):
                f.write(f"{ii+1} 1\n")

    def write_nuc_post(self, target_file_path: Path):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = StructureInitialiser()
        if self.seed_mode == 'dimer':
            structure_initialiser.add_seed_dimer(self.seed_type)
        elif self.seed_mode == 'trimer':
            structure_initialiser.add_seed_trimer(self.seed_type)

        structure_initialiser = self.topology_factory.straight_dimer(structure_initialiser=structure_initialiser)

        structure_initialiser.write_to_template_file(target_file_path, n_fragments=1)
        
        with open(target_file_path, 'a') as f:
            if self.peratom_masses:
                f.write(f"\n\nMasses\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1.0\n")
            
            f.write("\n\nCoords\n\n")
            for ii, coord in enumerate(structure_initialiser.positions):
                f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

            if True:
                f.write(f"\n\nMolecules\n\n")
                for ii in range(0, self.seed_size):
                    f.write(f"{ii+1} 1\n")

                for ii in range(self.seed_size, structure_initialiser.n_beads):
                    f.write(f"{ii+1} 2\n")

            f.write(f"\n\nFragments\n\n")
            f.write(f"1 1 2\n") # taken from Chris' scripts, not sure why this is done

    def write_denucleation(self, target_dir: Path):
        target_dir = Path(target_dir).resolve()

        structure_initialiser = self.topology_factory.straight_dimer()

        structure_initialiser.write_to_template_file(target_dir / "Denucleation_pre.txt", n_fragments=1)

        with open(target_dir / "Denucleation_pre.txt", 'a') as f:
            f.write(f"\n\nFragments\n\n")
            f.write(f"1 1\n")

        structure_initialiser.write_to_template_file(target_dir / "Denucleation_post.txt", n_fragments=0)

        map_file = MapFile(
            equivalences=tuple(zip(
                range(1, structure_initialiser.n_beads+1),
                range(1, structure_initialiser.n_beads+1)
            )),
            initiators=self.topology_factory.denucleation_initiators() or (1, 2), # nucleation dimer
            edges=(),
            deletes=tuple(range(1, structure_initialiser.n_beads+1)),
            constraints=(
                'custom "rxnsum(v_koff, 1) > random(0,1,v_vSeed)"',
                'custom "rxnsum(v_vIsHydrolysed, 1)"',
            )
        )

        map_file.write_to_file(target_dir / "Denucleation_map.txt")

    def write_nmer_on_pre(self, target_file_path: Path, n_mer: int, add_pol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, n_fit: int|None=None,enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = self.topology_factory.nmer(n_mer)
        
        if add_pol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches and n_mer > 2:
            n_fragments = 4
        elif add_pol_modulation and parallel_filament and n_mer > 2:
            n_fragments = 5
        elif add_pol_modulation and parallel_and_antiparallel and n_mer > 2:
            n_fragments = 5
        elif add_pol_modulation and antiparallel_filaments_modulate_both_patches and n_mer > 2:
            n_fragments = 5
        else:
            n_fragments = 3
            
        if n_fit is None:
            n_fit = n_mer

        
        if enable_templates_for9_and_14:
            atom_types_list_all = self.topology_factory.get_all_combinations_of_type_9_and_14(structure_initialiser)
            
            for i, atom_types_list in enumerate(atom_types_list_all):
                structure_initialiser.atom_types_array = atom_types_list
                target_file_path_addition = target_file_path.stem + f"_template_{i+1}.txt"
                target_file_i = Path(target_file_path.parent / target_file_path_addition).resolve()
                structure_initialiser.write_to_template_file(target_file_i, n_fragments=n_fragments)
                
                
                with open(target_file_i, 'a') as f:
                    f.write(f"\n\nFragments\n\n")
                    f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                    f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
                    f.write(f"3 {structure_initialiser.n_beads} \n ")
                    if add_pol_modulation and n_mer > 2:
                        if self.topology_factory.monomer_size==2:
                            f.write(f"4 {structure_initialiser.n_beads-2} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                        elif self.topology_factory.monomer_size==3:
                            f.write(f"4 {structure_initialiser.n_beads-4} \n") 
                    if add_pol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches) and n_mer > 2:
                        if self.topology_factory.monomer_size==2:
                            f.write(f"5 {structure_initialiser.n_beads-3} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                        elif self.topology_factory.monomer_size==3:
                            f.write(f"5 {structure_initialiser.n_beads-3} \n") 
                    # f.write(f"2 {' '.join([str(ii) for ii in range(1, structure_initialiser.n_beads+1)][-n_fit:])}")

                    f.write(f"\n\nMolecules\n\n")
                    for ii in range(structure_initialiser.n_beads):
                        f.write(f"{ii+1} 1\n")
        else:
        
            structure_initialiser.write_to_template_file(target_file_path, n_fragments=n_fragments)

            with open(target_file_path, 'a') as f:
                f.write(f"\n\nFragments\n\n")
                f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
                f.write(f"3 {structure_initialiser.n_beads} \n ")
                if add_pol_modulation and n_mer > 2:
                    if self.topology_factory.monomer_size==2:
                        f.write(f"4 {structure_initialiser.n_beads-2} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                    elif self.topology_factory.monomer_size==3:
                        f.write(f"4 {structure_initialiser.n_beads-4} \n") 
                if add_pol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches) and n_mer > 2:
                    if self.topology_factory.monomer_size==2:
                        f.write(f"5 {structure_initialiser.n_beads-3} \n")  # adding a fragment for polymer modulation - the patch on the growing end
                    elif self.topology_factory.monomer_size==3:
                        f.write(f"5 {structure_initialiser.n_beads-3} \n") 
                # f.write(f"2 {' '.join([str(ii) for ii in range(1, structure_initialiser.n_beads+1)][-n_fit:])}")

                f.write(f"\n\nMolecules\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1\n")

    def write_nmer_on_post(self, target_file_path: Path, n_mer: int, n_fit: int|None=None, enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = self.topology_factory.nmer(n_mer+1)
        ChemistryWriter.new_topology_last(
            structure_initialiser,
            n_start=structure_initialiser.n_beads - self.topology_factory.monomer_size,
            monomer_size=self.topology_factory.monomer_size
        )
        
        if n_fit is None:
            n_fit = n_mer + 1
        
        if enable_templates_for9_and_14:
            strc_initialiser_prestate = self.topology_factory.nmer(n_mer)
            atom_types_list_all = self.topology_factory.get_all_combinations_of_type_9_and_14(strc_initialiser_prestate)
            
            for i, atom_types_list in enumerate(atom_types_list_all):
                if len(strc_initialiser_prestate.atom_types_array) >6:
                    structure_initialiser.atom_types_array[:(n_mer-1)*self.topology_factory.monomer_size] = atom_types_list[:(n_mer-1)*self.topology_factory.monomer_size]
                target_file_path_addition = target_file_path.stem + f"_template_{i+1}.txt"
                target_file_i = Path(target_file_path.parent / target_file_path_addition).resolve()
                structure_initialiser.write_to_template_file(target_file_i, n_fragments=2)
                
                with open(target_file_i, 'a') as f:
                    if self.peratom_masses:
                        f.write(f"\n\nMasses\n\n")
                        for ii in range(structure_initialiser.n_beads):
                            f.write(f"{ii+1} 1.0\n")
                    
                    f.write("\n\nCoords\n\n")
                    for ii, coord in enumerate(structure_initialiser.positions):
                        f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

                    f.write(f"\n\nMolecules\n\n")
                    for ii in range(structure_initialiser.n_beads):
                        f.write(f"{ii+1} 1\n")

                    f.write(f"\n\nFragments\n\n")
                    f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                    f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
                    # f.write(f"2 {' '.join([str(ii) for ii in range(1, structure_initialiser.n_beads+1)][-(n_fit+1):])}")

        else:
        
            structure_initialiser.write_to_template_file(target_file_path, n_fragments=2)

            with open(target_file_path, 'a') as f:
                if self.peratom_masses:
                    f.write(f"\n\nMasses\n\n")
                    for ii in range(structure_initialiser.n_beads):
                        f.write(f"{ii+1} 1.0\n")
                
                f.write("\n\nCoords\n\n")
                for ii, coord in enumerate(structure_initialiser.positions):
                    f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

                f.write(f"\n\nMolecules\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1\n")

                f.write(f"\n\nFragments\n\n")
                f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
                f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
                # f.write(f"2 {' '.join([str(ii) for ii in range(1, structure_initialiser.n_beads+1)][-(n_fit+1):])}")

    def write_nmer_on_map(self, target_file_path: Path, n_mer: int, add_pol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, kon_premod: float=0.0, kon_postmod: float=0.0, reaction_time: float=0.0, add_volume_exclusion: bool=False):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = self.topology_factory.nmer(n_mer+1)
        
        if add_pol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches and n_mer > 2:
            constraints = 'custom "rxnsum(v_kon,4) > random(0,1,v_vSeed)"'
        elif add_pol_modulation and parallel_filament and n_mer > 2:
            constraints = f'custom "(({kon_postmod}-{kon_premod})*{reaction_time}*(rxnsum(v_neigh9,4)+rxnsum(v_neigh6,5)-rxnsum(v_neigh9,4)*rxnsum(v_neigh6,5))+rxnsum(v_konin,1))*rxnsum(v_expfactor,1) > random(0,1,v_vSeed)"'
        elif add_pol_modulation and antiparallel_filaments_modulate_both_patches and n_mer > 2:
            constraints = f'custom "(({kon_postmod}-{kon_premod})*{reaction_time}*(rxnsum(v_neigh9,5)+rxnsum(v_neigh6,4)-rxnsum(v_neigh9,5)*rxnsum(v_neigh6,4))+rxnsum(v_konin,1))*rxnsum(v_expfactor,1) > random(0,1,v_vSeed)"'
        elif add_pol_modulation and parallel_and_antiparallel and n_mer > 2:
            constraints = f'custom "(({kon_postmod}-{kon_premod})*{reaction_time}*(1-(1-rxnsum(v_neigh9,4))*(1-rxnsum(v_neigh6,4))*(1-rxnsum(v_neigh6,5))*(1-rxnsum(v_neigh9,5)))+rxnsum(v_konin,1))*rxnsum(v_expfactor,1) > random(0,1,v_vSeed)"'
        else:
            constraints = 'custom "rxnsum(v_kon,1) > random(0,1,v_vSeed)"'
            
        
        final_constraints = (constraints,)

        map_file = MapFile(
            equivalences=tuple(zip(
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1),
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1)
            )),
            initiators=self.topology_factory.nmer_on_initiators(n_mer) or (structure_initialiser.n_beads-(n_mer+1)*self.topology_factory.monomer_size+1, structure_initialiser.n_beads-n_mer*self.topology_factory.monomer_size+1), # last two monomers, only works this way for some reason
            #initiators = (n_mer*self.topology_factory.monomer_size - (self.topology_factory.monomer_size-1),n_mer*self.topology_factory.monomer_size),
            edges=(),
            creates=tuple(range(
                1+structure_initialiser.n_beads - self.topology_factory.monomer_size, 1+structure_initialiser.n_beads
            )),
            constraints=final_constraints
        )

        map_file.write_to_file(target_file_path)

    def write_nmer_surface_on_pre(self, target_file_path: Path, n_mer: int, n_fit: int|None=None, surface_type: int=5):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = StructureInitialiser()

        # first a membrane particle
        structure_initialiser.positions = np.append(structure_initialiser.positions, [[0.0, 0.0, 0.0]], axis=0)
        structure_initialiser.atom_types_array = np.append(structure_initialiser.atom_types_array, [surface_type], axis=0)

        structure_initialiser = self.topology_factory.nmer(n_mer, structure_initialiser=structure_initialiser)

        structure_initialiser.write_to_template_file(target_file_path, n_fragments=2)

        if n_fit is None:
            n_fit = (structure_initialiser.n_beads - 1)//self.topology_factory.monomer_size

        with open(target_file_path, 'a') as f:
            f.write(f"\n\nFragments\n\n")
            f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
            f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
            # f.write(f"2 {' '.join([str(ii) for ii in range(2, structure_initialiser.n_beads+1)][-n_fit:])}")

    def write_nmer_surface_on_post(self, target_file_path: Path, n_mer, d_surface: float=1.0, surface_type: int=5, com: np.ndarray|None=None, n_fit: int|None=None):
        if com is None:
            com = np.zeros(3, dtype=float)

        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = StructureInitialiser()
        equilibrium_helix = self.topology_factory.nmer(n_mer+1).positions

        # first add a single surface particle to the structure initialiser
        vec_last_bead = equilibrium_helix[-1, :] - com
        vec_last_bead[:] = vec_last_bead[:] / np.linalg.norm(vec_last_bead)
        surface_pos = equilibrium_helix[-1, :] + (vec_last_bead * d_surface)

        structure_initialiser.positions = np.append(structure_initialiser.positions, surface_pos[np.newaxis, :], axis=0)
        structure_initialiser.atom_types_array = np.append(structure_initialiser.atom_types_array, [surface_type], axis=0)

        structure_initialiser = self.topology_factory.nmer(n_mer+1, structure_initialiser=structure_initialiser)

        structure_initialiser.write_to_template_file(target_file_path, n_fragments=2)

        if n_fit is None:
            n_fit = (structure_initialiser.n_beads - 1)//self.topology_factory.monomer_size
        
        with open(target_file_path, 'a') as f:
            if self.peratom_masses:
                f.write(f"\n\nMasses\n\n")
                for ii in range(structure_initialiser.n_beads):
                    f.write(f"{ii+1} 1.0\n")

            f.write("\n\nCoords\n\n")
            for ii, coord in enumerate(structure_initialiser.positions):
                f.write(f"{ii+1} {coord[0]} {coord[1]} {coord[2]}\n")

            f.write(f"\n\nMolecules\n\n")
            f.write("1 0\n") # TODO: add a molecule type for the surface particles
            for ii in range(1, structure_initialiser.n_beads):
                f.write(f"{ii+1} 0\n")

            f.write(f"\n\nFragments\n\n")
            f.write(f"1 {structure_initialiser.n_beads-self.topology_factory.monomer_size+1}\n")
            f.write(self.topology_factory.fit_string(structure_initialiser.n_beads, n_fit))
            # f.write(f"2 {' '.join([str(ii) for ii in range(2, structure_initialiser.n_beads+1)][-(n_fit+1):])}")

    def write_nmer_surface_on_map(self, target_file_path: Path, n_mer: int, surface_type: int=5):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = self.topology_factory.nmer(n_mer+1)

        # first a membrane particle
        structure_initialiser.positions = np.append([[0.0, 0.0, 0.0]], structure_initialiser.positions, axis=0)
        structure_initialiser.atom_types_array = np.append([surface_type], structure_initialiser.atom_types_array, axis=0)

        map_file = MapFile(
            equivalences=tuple(zip(
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1),
                range(1, structure_initialiser.n_beads-self.topology_factory.monomer_size+1),
            )),
            initiators=(1, structure_initialiser.n_beads-self.topology_factory.monomer_size), # membrane particle and last helix particle that was there before
            edges=(),
            creates=tuple(range(
                1+structure_initialiser.n_beads-self.topology_factory.monomer_size, 1+structure_initialiser.n_beads
            )),
            constraints=('custom "rxnsum(v_kon,1) > random(0,1,v_vSeed)"',)
        )

        map_file.write_to_file(target_file_path)

    def write_nmer_off_pre(self, target_file_path: Path, n_mer: int, add_depol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()
        
        if add_depol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches and n_mer > 2:
            n_fragments = 2
        elif add_depol_modulation and parallel_filament and n_mer > 2:
            n_fragments = 3
        elif add_depol_modulation and parallel_and_antiparallel and n_mer > 2:
            n_fragments = 3
        elif add_depol_modulation and antiparallel_filaments_modulate_both_patches and n_mer > 2:
            n_fragments = 3
        else:
            n_fragments = 1

        structure_initialiser = self.topology_factory.nmer(n_mer)
        
        if enable_templates_for9_and_14:
            atom_types_list_all = self.topology_factory.get_all_combinations_of_type_9_and_14(structure_initialiser)
            
            for i, atom_types_list in enumerate(atom_types_list_all):
                structure_initialiser.atom_types_array = atom_types_list
                target_file_path_addition = target_file_path.stem + f"_template_{i+1}.txt"
                target_file_i = Path(target_file_path.parent / target_file_path_addition).resolve()
                structure_initialiser.write_to_template_file(target_file_i, n_fragments=n_fragments)
                
                with open(target_file_i, 'a') as f:
                    f.write(f"\n\nFragments\n\n")
                    f.write(f"1 1\n")
                    if add_depol_modulation and n_mer > 2:
                        if self.topology_factory.monomer_size == 2:
                            f.write(f"2 4 \n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                        elif self.topology_factory.monomer_size == 3:
                            f.write(f"2 5\n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                        elif self.topology_factory.monomer_size == 4:
                            f.write(f"2 8\n")  # TripleBeadPatchFilaments: patch (atom 8) of the second monomer from the shrinking end
                        elif self.topology_factory.depol_fragment_atom() is not None:
                            f.write(f"2 {self.topology_factory.depol_fragment_atom()}\n")  # e.g. TripleBeadPatchDiffuseFilaments
                    if add_depol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches) and n_mer > 2:
                        if self.topology_factory.monomer_size == 2:
                            f.write(f"3 5 \n")  # adding a fragment for parallel filament modulation - the patch on the shrinking end
                        elif self.topology_factory.monomer_size == 3:
                            f.write(f"3 6\n")  # adding a fragment for parallel filament modulation - the patch on the shrinking end

        else:
            structure_initialiser.write_to_template_file(target_file_path, n_fragments=n_fragments)

            with open(target_file_path, 'a') as f:
                f.write(f"\n\nFragments\n\n")
                f.write(f"1 1\n")
                if add_depol_modulation and n_mer > 2:
                    if self.topology_factory.monomer_size == 2:
                        f.write(f"2 4 \n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                    elif self.topology_factory.monomer_size == 3:
                        f.write(f"2 5\n")  # adding a fragment for polymer modulation - the patch on the shrinking end
                    elif self.topology_factory.monomer_size == 4:
                        f.write(f"2 8\n")  # TripleBeadPatchFilaments: patch (atom 8) of the second monomer from the shrinking end
                    elif self.topology_factory.depol_fragment_atom() is not None:
                        f.write(f"2 {self.topology_factory.depol_fragment_atom()}\n")  # e.g. TripleBeadPatchDiffuseFilaments
                if add_depol_modulation and (parallel_filament or parallel_and_antiparallel or antiparallel_filaments_modulate_both_patches) and n_mer > 2:
                    if self.topology_factory.monomer_size == 2:
                        f.write(f"3 5 \n")  # adding a fragment for parallel filament modulation - the patch on the shrinking end
                    elif self.topology_factory.monomer_size == 3:
                        f.write(f"3 6\n")  # adding a fragment for parallel filament modulation - the patch on the shrinking end

    def write_nmer_off_post(self, target_file_path: Path, n_mer: int, enable_templates_for9_and_14: bool=False):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = self.topology_factory.nmer(n_mer)
        structure_initialiser = self.topology_factory.shrink_tail(structure_initialiser)
        
        if enable_templates_for9_and_14:
            structure_initialiser_pre = self.topology_factory.nmer(n_mer)
            atom_types_list_all = self.topology_factory.get_all_combinations_of_type_9_and_14(structure_initialiser_pre)
            
            for i, atom_types_list in enumerate(atom_types_list_all):
                structure_initialiser.atom_types_array[self.topology_factory.monomer_size*2+1:] = atom_types_list[self.topology_factory.monomer_size*2+1:]
                target_file_path_addition = target_file_path.stem + f"_template_{i+1}.txt"
                target_file_i = Path(target_file_path.parent / target_file_path_addition).resolve()
                structure_initialiser.write_to_template_file(target_file_i, n_fragments=0)
        
        else:
            structure_initialiser.write_to_template_file(target_file_path, n_fragments=0)

    def write_nmer_off_map(self, target_file_path: Path, n_mer:int, add_depol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, thyd_premod: float = 0.0, thyd_postmod: float = 0.0, r_off: float = 0.0, r_off_modulation: float = 0.0, reaction_time: float = 0.0, explicit_hydrolysis: bool=False):
        target_file_path = Path(target_file_path).resolve()

        structure_initialiser = self.topology_factory.nmer(n_mer)
        
        if explicit_hydrolysis:
            if add_depol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches:
                first_constraint = 'custom "rxnsum(v_koff, 2) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and parallel_filament:
                first_constraint = f'custom "(({r_off_modulation}-{r_off})*{reaction_time}*(rxnsum(v_neigh9,2)+rxnsum(v_neigh6,3)-rxnsum(v_neigh9,2)*rxnsum(v_neigh6,3))+{r_off}*{reaction_time}) > random(0,1,v_vSeed)"' 
            elif add_depol_modulation and antiparallel_filaments_modulate_both_patches:
                first_constraint = f'custom "(({r_off_modulation}-{r_off})*{reaction_time}*(rxnsum(v_neigh9,3)+rxnsum(v_neigh6,2)-rxnsum(v_neigh9,3)*rxnsum(v_neigh6,2))+{r_off}*{reaction_time}) > random(0,1,v_vSeed)"' 
            elif add_depol_modulation and parallel_and_antiparallel:
                first_constraint = f'custom "(({r_off_modulation}-{r_off})*{reaction_time}*(1- (1-rxnsum(v_neigh9,2))*(1-rxnsum(v_neigh6,3))*(1-rxnsum(v_neigh6,2))*(1-rxnsum(v_neigh9,3)))+{r_off}*{reaction_time}) > random(0,1,v_vSeed)"'
            else:
                first_constraint = 'custom "rxnsum(v_koff, 1) > random(0,1,v_vSeed)"'
        else:
            if add_depol_modulation and not parallel_filament and not parallel_and_antiparallel and not antiparallel_filaments_modulate_both_patches and n_mer > 2:
                first_constraint = 'custom "rxnsum(v_koff, 2) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and parallel_filament and n_mer > 2:
                first_constraint = f'custom "1.0-exp(-rxnsum(v_vSA,2)/((({thyd_postmod}-{thyd_premod})*(rxnsum(v_neigh9,2)+rxnsum(v_neigh6,3)-rxnsum(v_neigh9,2)*rxnsum(v_neigh6,3))+{thyd_premod})/v_tstep)) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and antiparallel_filaments_modulate_both_patches and n_mer > 2:
                first_constraint = f'custom "1.0-exp(-rxnsum(v_vSA,2)/((({thyd_postmod}-{thyd_premod})*(rxnsum(v_neigh9,3)+rxnsum(v_neigh6,2)-rxnsum(v_neigh9,3)*rxnsum(v_neigh6,2))+{thyd_premod})/v_tstep)) > random(0,1,v_vSeed)"'
            elif add_depol_modulation and parallel_and_antiparallel and n_mer > 2:
                first_constraint = f'custom "1.0-exp(-rxnsum(v_vSA,2)/((({thyd_postmod}-{thyd_premod})*(1- (1-rxnsum(v_neigh9,2))*(1-rxnsum(v_neigh6,2))*(1-rxnsum(v_neigh6,3))*(1-rxnsum(v_neigh9,3)))+{thyd_premod})/v_tstep)) > random(0,1,v_vSeed)"'
            else:
                first_constraint = 'custom "rxnsum(v_koff, 1) > random(0,1,v_vSeed)"'

        map_file = MapFile(
            equivalences=tuple(zip(
                range(1, structure_initialiser.n_beads+1),
                range(1, structure_initialiser.n_beads+1)
            )),
            initiators=self.topology_factory.off_initiators() or (1, 1 + self.topology_factory.monomer_size),
            edges=(),
            deletes=tuple(range(1, 1 + self.topology_factory.monomer_size)),
            constraints=(
                first_constraint,
                'custom "rxnsum(v_vIsHydrolysed, 1)"',
            )
        )

        map_file.write_to_file(target_file_path)

    def write_oligomer_treadmilling(self, target_dir: Path, add_pol_modulation: bool=False, add_depol_modulation: bool=False, parallel_filament: bool=False, n_fit: int|None=None):       
        self.write_on_pre(target_dir / 'OligomerOn_pre.txt', add_pol_modulation=add_pol_modulation, parallel_filament=parallel_filament, n_fit=n_fit)
        self.write_on_post(target_dir / 'OligomerOn_post.txt', n_fit=n_fit)
        self.write_on_map(target_dir / 'OligomerOn_map.txt', add_pol_modulation=add_pol_modulation)
        self.write_off_pre(target_dir / 'OligomerOff_pre.txt', add_depol_modulation=add_depol_modulation, parallel_filament=parallel_filament)
        self.write_off_post(target_dir / 'OligomerOff_post.txt')
        self.write_off_map(target_dir / 'OligomerOff_map.txt', add_depol_modulation=add_depol_modulation)
        
    def write_secnuc_templates(self, target_dir: Path):       
        self.write_secnuc_pre(target_dir / 'SecondaryNuc_pre.txt')
        self.write_secnuc_post(target_dir / 'SecondaryNuc_post.txt')
        self.write_secnuc_map(target_dir / 'SecondaryNuc_map.txt')
    
    def write_oligomer_treadmilling_two_templates(self, target_dir: Path, add_pol_modulation: bool=False, add_depol_modulation: bool=False, write_two_oligomer_files: bool=True, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, thyd_premod: float=0.0, thyd_postmod: float=0.0, kon_premod: float=0.0, kon_postmod: float=0.0, reaction_time: float=0.0, r_off: float=0.0, r_off_modulation: float=0.0, explicit_hydrolysis: bool=False, n_fit: int|None=None, add_volume_exclusion: bool=False, enable_templates_for9_and_14: bool=False):       
        self.write_on_pre(target_dir / 'OligomerOn_pre.txt', type_to_add = 2, add_pol_modulation=add_pol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, n_fit=n_fit, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_on_post(target_dir / 'OligomerOn_post.txt', type_to_add = 2, n_fit=n_fit, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_on_map(target_dir / 'OligomerOn_map.txt', type_to_add = 2, add_pol_modulation=add_pol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, kon_premod=kon_premod, kon_postmod=kon_postmod, reaction_time=reaction_time, write_two_oligomer_files=write_two_oligomer_files, add_volume_exclusion=add_volume_exclusion )
        
        self.write_on_pre(target_dir / 'OligomerOn2_pre.txt', type_to_add = 1, add_pol_modulation=add_pol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, n_fit=n_fit, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_on_post(target_dir / 'OligomerOn2_post.txt', type_to_add = 1, n_fit=n_fit, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_on_map(target_dir / 'OligomerOn2_map.txt', type_to_add = 1, add_pol_modulation=add_pol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, kon_premod=kon_premod, kon_postmod=kon_postmod, reaction_time=reaction_time, write_two_oligomer_files=write_two_oligomer_files, add_volume_exclusion=add_volume_exclusion)
        
        self.write_off_pre(target_dir / 'OligomerOff_pre.txt', type_to_add = 3, add_depol_modulation=add_depol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_off_post(target_dir / 'OligomerOff_post.txt', type_to_add = 3, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_off_map(target_dir / 'OligomerOff_map.txt', type_to_add = 3, add_depol_modulation=add_depol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, thyd_premod=thyd_premod, thyd_postmod=thyd_postmod, r_off=r_off, r_off_modulation=r_off_modulation, explicit_hydrolysis=explicit_hydrolysis, reaction_time=reaction_time, write_two_oligomer_files=write_two_oligomer_files)
        
        self.write_off_pre(target_dir / 'OligomerOff2_pre.txt', type_to_add = 1, add_depol_modulation=add_depol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_off_post(target_dir / 'OligomerOff2_post.txt', type_to_add = 1, write_two_oligomer_files=write_two_oligomer_files, enable_templates_for9_and_14=enable_templates_for9_and_14)
        self.write_off_map(target_dir / 'OligomerOff2_map.txt', type_to_add = 1, add_depol_modulation=add_depol_modulation, parallel_filament=parallel_filament, parallel_and_antiparallel=parallel_and_antiparallel, antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches, thyd_premod=thyd_premod, thyd_postmod=thyd_postmod, r_off=r_off, r_off_modulation=r_off_modulation, explicit_hydrolysis=explicit_hydrolysis, reaction_time=reaction_time, write_two_oligomer_files=write_two_oligomer_files)
    
    def write_surface_oligomer_treadmilling(
            self, 
            target_dir: Path, 
            surface_type: int=5, 
            d_surface: float=1.0, 
            com: np.ndarray|None=None, 
            n_fit: int|None=None
        ):
        self.write_surface_on_pre(
            target_dir / 'OligomerOn_pre.txt',
            surface_type=surface_type, n_fit=n_fit
        )
        self.write_surface_on_post(
            target_dir / 'OligomerOn_post.txt',
            d_surface=d_surface, surface_type=surface_type, com=com, n_fit=n_fit
        )
        self.write_surface_on_map(
            target_dir / 'OligomerOn_map.txt',
            surface_type=surface_type
        )
        self.write_off_pre(target_dir / 'OligomerOff_pre.txt')
        self.write_off_post(target_dir / 'OligomerOff_post.txt')
        self.write_off_map(target_dir / 'OligomerOff_map.txt')

    def write_nmer_chemistry(self, target_dir: Path, add_pol_modulation: bool=False, add_depol_modulation: bool=False, parallel_filament: bool=False, parallel_and_antiparallel: bool=False, antiparallel_filaments_modulate_both_patches: bool=False, thyd_premod: float=0.0, thyd_postmod: float=0.0, kon_premod: float=0.0, kon_postmod: float=0.0, reaction_time: float=0.0, r_off: float=0.0, r_off_modulation: float=0.0, explicit_hydrolysis: bool=False, n_fit: int|None=None, add_volume_exclusion: bool=False, enable_templates_for9_and_14: bool=False):
        n_mer_max = self.topology_factory.max_nmer

        for n_mer in range(2, n_mer_max):
            self.write_nmer_off_pre(
                target_dir / f'{n_mer+1:d}mer_off_pre.txt',
                n_mer=n_mer+1,
                add_depol_modulation=add_depol_modulation,
                parallel_filament=parallel_filament,
                parallel_and_antiparallel=parallel_and_antiparallel,
                antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches,
                enable_templates_for9_and_14=enable_templates_for9_and_14
            )
            self.write_nmer_off_map(
                target_dir / f'{n_mer+1:d}mer_off_map.txt',
                n_mer=n_mer+1,
                add_depol_modulation=add_depol_modulation,
                parallel_filament=parallel_filament,
                parallel_and_antiparallel=parallel_and_antiparallel,
                antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches,
                thyd_premod=thyd_premod,
                thyd_postmod=thyd_postmod,
                r_off=r_off,
                r_off_modulation=r_off_modulation,
                explicit_hydrolysis=explicit_hydrolysis,
                reaction_time=reaction_time
            )
            self.write_nmer_off_post(
                target_dir / f'{n_mer+1:d}mer_off_post.txt',
                n_mer=n_mer+1,
                enable_templates_for9_and_14=enable_templates_for9_and_14
            )

            self.write_nmer_on_pre(
                target_dir / f'{n_mer:d}mer_on_pre.txt',
                n_mer, n_fit=n_fit,
                add_pol_modulation=add_pol_modulation,
                parallel_filament=parallel_filament,
                parallel_and_antiparallel=parallel_and_antiparallel,
                antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches,
                enable_templates_for9_and_14=enable_templates_for9_and_14
            )
            self.write_nmer_on_map(
                target_dir / f'{n_mer:d}mer_on_map.txt',
                n_mer, 
                add_pol_modulation=add_pol_modulation,
                parallel_filament=parallel_filament,
                parallel_and_antiparallel=parallel_and_antiparallel,
                antiparallel_filaments_modulate_both_patches=antiparallel_filaments_modulate_both_patches,
                kon_premod=kon_premod,
                kon_postmod=kon_postmod,
                reaction_time=reaction_time,
                add_volume_exclusion=add_volume_exclusion
            )
            self.write_nmer_on_post(
                target_dir / f'{n_mer:d}mer_on_post.txt',
                n_mer, n_fit=n_fit,
                enable_templates_for9_and_14=enable_templates_for9_and_14
            )

    def write_surface_nmer_chemistry(
        self,
        target_dir: Path,
        surface_type: int=5,
        d_surface: float=1.0,
        com: np.ndarray|None=None,
        n_fit: int|None=None
    ):
        n_mer_max = self.topology_factory.max_nmer

        for n_mer in range(2, n_mer_max):

            self.write_nmer_off_pre(
                target_dir / f'{n_mer+1:d}mer_off_pre.txt',
                n_mer=n_mer+1
            )
            self.write_nmer_off_map(
                target_dir / f'{n_mer+1:d}mer_off_map.txt',
                n_mer=n_mer+1
            )
            self.write_nmer_off_post(
                target_dir / f'{n_mer+1:d}mer_off_post.txt',
                n_mer=n_mer+1
            )
            self.write_nmer_surface_on_pre(
                target_dir / f'{n_mer:d}mer_on_pre.txt',
                n_mer=n_mer, surface_type=surface_type, n_fit=n_fit
            )
            self.write_nmer_surface_on_post(
                target_dir / f'{n_mer:d}mer_on_post.txt',
                n_mer=n_mer, d_surface=d_surface, surface_type=surface_type, com=com, n_fit=n_fit
            )
            self.write_nmer_surface_on_map(
                target_dir / f'{n_mer:d}mer_on_map.txt',
                n_mer=n_mer, surface_type=surface_type
            )

    def write_nucleation_chemistry(self, target_dir: Path):
        self.write_nuc_pre(target_dir / 'Nucleation_pre.txt')
        self.write_nuc_map(target_dir / 'Nucleation_map.txt')
        self.write_nuc_post(target_dir / 'Nucleation_post.txt')
        self.write_denucleation(target_dir)

@dataclass
class MapFile:
    equivalences: tuple[tuple[int, int], ...]
    initiators: tuple[int, int]
    edges: tuple[int, ...]
    constraints: tuple[str, ...] = field(default_factory = lambda: ())
    creates: tuple[int, ...] = field(default_factory = lambda: ())
    deletes: tuple[int, ...] = field(default_factory = lambda: ())
    title: str = field(default_factory = lambda: 'Automatically generated map file')

    def write_to_file(self, target_file_path: Path):
        with open(target_file_path, 'w') as f:
            f.write(f'# {self.title} \n\n')
            f.write(f"{len(self.equivalences)} equivalences\n")
            f.write(f"{len(self.edges)} edgeIDs\n")
            f.write(f"{len(self.creates)} createIDs\n")
            f.write(f"{len(self.deletes)} deleteIDs\n")
            f.write(f"{len(self.constraints)} constraints\n")

            if len(self.initiators) != 2:
                raise ValueError('Initiators must be a tuple of length 2')

            f.write(f"\n\nInitiatorIDs\n\n")
            for initiator in self.initiators:
                f.write(f"{initiator}\n")

            if len(self.edges):
                f.write(f"\n\nEdgeIDs\n\n")
                for edge in self.edges:
                    f.write(f"{edge}\n")

            f.write(f"\n\nEquivalences\n\n")
            for ii in range(len(self.equivalences)):
                f.write(f"{self.equivalences[ii][0]} {self.equivalences[ii][1]}\n")

            if len(self.creates):
                f.write(f"\n\nCreateIDs\n\n")
                for create in self.creates:
                    f.write(f"{create}\n")
            
            if len(self.deletes):
                f.write(f"\n\nDeleteIDs\n\n")
                for delete in self.deletes:
                    f.write(f"{delete}\n")

            f.write(f"\n\nConstraints\n\n")
            for constraint in self.constraints:
                f.write(f"{constraint}\n")