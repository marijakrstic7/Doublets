from dataclasses import dataclass
from pathlib import Path
import re
from io import StringIO
import warnings
from math import sqrt

from ase.geometry import get_angles, get_dihedrals
import numpy as np
import jinja2

from .ylz_utils import _YLZUtils, ALMOST_SPHERICAL_ELLIPSOID_DIMENSIONS
from .escrt_utils import _ESCRTUtils

class StructureInitialiser:
    def __init__(self):
        self.positions: np.ndarray[float] = np.empty((0, 3), dtype=float)
        self.atom_types_array: np.ndarray[int] = np.empty((0, ), dtype=int)
        self.molecule_ids: np.ndarray[int] = np.empty((0, ), dtype=int)
        self.bond_array: np.ndarray[int] = np.empty((0, 4), dtype=int)
        self.angle_array: np.ndarray[int] = np.empty((0, 5), dtype=int)
        self.dihedral_array: np.ndarray[int] = np.empty((0, 6), dtype=int)
        self.quaternions: np.ndarray[float] = np.empty((0, 4), dtype=float)
        self.ellipsoid_indices: tuple = ()

        self.box: np.ndarray = np.array([
            [-100, 100],
            [-100, 100],
            [-100, 100]
        ], dtype=float)

    @property
    def n_beads(self):
        return len(self.positions)
    
    @property
    def n_bonds(self):
        return len(self.bond_array)

    @property
    def n_angles(self):
        return len(self.angle_array)
    
    @property
    def n_dihedrals(self):
        return len(self.dihedral_array)
    
    @property
    def n_atom_types(self):
        return np.max(self.atom_types_array)
    
    @property
    def n_bond_types(self):
        if not len(self.bond_array):
            return 0

        return np.max(self.bond_array[:, 1])
    
    @property
    def n_angle_types(self):
        if not len(self.angle_array):
            return 0
        
        return np.max(self.angle_array[:, 1])
    
    @property
    def n_dihedral_types(self):
        if not len(self.dihedral_array):
            return 0
        
        return np.max(self.dihedral_array[:, 1])
    
    @property
    def n_molecules(self):
        # no particles in system
        if not len(self.molecule_ids):
            return 0

        # no molecules except type 0
        if not len(self.molecule_ids[self.molecule_ids > 0]):
            return 0

        # count molecules
        return len(np.unique(self.molecule_ids[self.molecule_ids > 0]))

    def add_seed_dimer(self, seed_type: int = 4, seed_molecule: int = 0):
        # add seed atoms
        self.positions = np.append(self.positions, np.array([[0, 0, 0], [1, 0, 0]]), axis=0)
        self.atom_types_array = np.append(self.atom_types_array, [seed_type, seed_type+1], axis=0)
        self.bond_array = np.append(
            self.bond_array,
            [[self.n_bonds+1, 1, self.n_beads-1, self.n_beads]],
            axis=0
        )
        self.molecule_ids = np.append(self.molecule_ids, [seed_molecule, seed_molecule], axis=0)

    def add_seed_trimer(self, seed_type: int = 4, seed_molecule: int = 0, atom_diameter: float = 1.0):
        # add seed atoms as equilateral triangle in x and z
        self.positions = np.append(self.positions, np.array([[0, 0, 0], [atom_diameter,0, 0], [atom_diameter/2, 0, 2.0]]), axis=0)
        self.atom_types_array = np.append(self.atom_types_array, [seed_type, seed_type+1, seed_type+2], axis=0)
        # connect seed atoms with bonds
        self.bond_array = np.append(
            self.bond_array,
            [
                [self.n_bonds+1, 1, self.n_beads-2, self.n_beads-1],
                [self.n_bonds+2, 1, self.n_beads-1, self.n_beads],
                [self.n_bonds+3, 1, self.n_beads-2, self.n_beads],

            ],
            axis=0
        )
        self.molecule_ids = np.append(self.molecule_ids, [seed_molecule, seed_molecule, seed_molecule], axis=0)

    def _add_bad_topology(self,
            starting_index: int, n_monomers: int,
            add_nn_angles=False, add_nth_neigbour_bonds=False,
            step_ab:int=1, step_bc:int=1, step_cd:int=1,
            angle_type:int=1, dihedral_type:int=1, nn_angle_type:int=2,
            monomer_size: int=1,
            flexible_create=False,
            add_dihedrals: bool=True
        ):

        # add bonds, angles and dihedrals
        self.bond_array = np.append(self.bond_array, BADUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=starting_index,
            starting_bond=self.n_bonds+1, monomer_size=monomer_size
        ), axis=0)

        if BADUtils._nangles_from_steps(n_monomers, step_ab, step_bc) > 0:
            self.angle_array = np.append(self.angle_array, BADUtils._get_angle_array(
                n_monomers=n_monomers, step_ab=step_ab,
                step_bc=step_bc, angle_type=angle_type,
                starting_index=starting_index, starting_angle=self.n_angles+1,
                monomer_size=monomer_size
            ), axis=0)
        elif not flexible_create:
            raise Warning(f"Did not create angles for {n_monomers} beads and steps {step_ab} and {step_bc}")

        if add_dihedrals:
            if BADUtils._ndihedrals_from_steps(n_monomers, step_ab, step_bc, step_cd) > 0:
                self.dihedral_array = np.append(self.dihedral_array, BADUtils._get_dihedral_array(
                    n_monomers=n_monomers, step_ab=step_ab,
                    step_bc=step_bc, step_cd=step_cd, dihedral_type=dihedral_type,
                    starting_index=starting_index, starting_dihedral=self.n_dihedrals+1,
                    monomer_size=monomer_size
                ), axis=0)
            elif not flexible_create:
                raise Warning(f"Did not create dihedrals for {n_monomers} beads and steps {step_ab}, {step_bc} and {step_cd}")

        if add_nn_angles and (BADUtils._nangles_from_steps(n_monomers, 1, 1)):
            angle_array_nn = BADUtils._get_angle_array(
                n_monomers=n_monomers, step_ab=1,
                step_bc=1, angle_type=nn_angle_type,
                starting_index=starting_index, starting_angle=self.n_angles+1,
                monomer_size=monomer_size
            )
            self.angle_array = np.vstack((self.angle_array, angle_array_nn))
        elif add_nn_angles and not flexible_create:
            raise Warning(f"Did not create nn angles for {n_monomers} beads and steps {1} and {1}")
        
        if add_nth_neigbour_bonds:
            bond_array_nn = BADUtils._get_bond_array(
                n_monomers=n_monomers, starting_index=starting_index, starting_bond=self.n_bonds+1,
                bond_type=2, step_ab=step_ab,
                monomer_size=monomer_size
            )
            self.bond_array = np.vstack((self.bond_array, bond_array_nn))

    def add_as_bad_chain(self, 
        positions, atom_type:int=1, add_nn_angles=False, add_nth_neigbour_bonds=False,
        step_ab:int=1, step_bc:int=1, step_cd:int=1,
        angle_type:int=1, dihedral_type:int=1, nn_angle_type:int=2,
        molecule_id: None | int = None,
        flexible_create=False
    ):
        # needed to properly add bonds and dihedrals
        cur_nbeads = self.n_beads
        n_added_beads = len(positions)

        # add new positions and types
        self.positions = np.append(self.positions, positions, axis=0)
        self.atom_types_array = np.append(self.atom_types_array, np.array([atom_type]*n_added_beads, dtype=int))

        if molecule_id is None:
            molecule_id = self.n_molecules + 1
        self.molecule_ids = np.append(self.molecule_ids, np.array([molecule_id]*n_added_beads, dtype=int))

        self._add_bad_topology(
            n_monomers=n_added_beads, starting_index=cur_nbeads+1,
            add_nn_angles=add_nn_angles, add_nth_neigbour_bonds=add_nth_neigbour_bonds,
            step_ab=step_ab, step_bc=step_bc, step_cd=step_cd,
            angle_type=angle_type, dihedral_type=dihedral_type, nn_angle_type=nn_angle_type,
            flexible_create=flexible_create,
            monomer_size=1
        )

    def add_as_double_bad_chain(self,
        positions, atom_type:int=1, subatom_type: int=6,
        add_nn_angles=False, add_nth_neigbour_bonds=False,
        step_ab:int=1, step_bc:int=1, step_cd:int=1,
        angle_type:int=1, dihedral_type:int=1, nn_angle_type:int=2,
        molecule_id: None | int = None,
        flexible_create=False, monomer_size: int=2
    ):
        cur_nbeads = self.n_beads
        n_monomers = len(positions) // 2
        
        self.positions = np.append(self.positions, positions, axis=0)

        new_atom_types = np.ones(len(positions), dtype=int)
        new_atom_types[::monomer_size] = atom_type
        new_atom_types[1::monomer_size] = subatom_type

        self.atom_types_array = np.append(self.atom_types_array, new_atom_types)
        if molecule_id is None:
            # already added one molecule in add_as_bad_chain, this should be same molecule
            molecule_id = self.n_molecules
        self.molecule_ids = np.append(self.molecule_ids, np.array([molecule_id]*len(positions), dtype=int))

        self._add_bad_topology(
            n_monomers=n_monomers, starting_index=cur_nbeads+1,
            add_nn_angles=add_nn_angles, add_nth_neigbour_bonds=add_nth_neigbour_bonds,
            step_ab=step_ab, step_bc=step_bc, step_cd=step_cd,
            angle_type=angle_type, dihedral_type=dihedral_type, nn_angle_type=nn_angle_type,
            flexible_create=flexible_create,
            monomer_size=monomer_size
        )

        subbond_array = DoubleBADUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_bond=self.n_bonds+1,
            bad_bond_type=1, dipole_bond_type=3, chain_bond_type=4, step_ab=step_ab, add_bad_bonds=False,
            monomer_size=monomer_size
        )

        self.bond_array = np.append(self.bond_array, subbond_array, axis=0)

        subangle_array = DoubleBADUtils._subatom_angles(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_angle=self.n_angles+1,
            monomer_size=monomer_size
        )
        self.angle_array = np.append(self.angle_array, subangle_array, axis=0)

        # add dihedrals (polar interaction) between monomers
        subdihedral_array = DoubleBADUtils._subatom_dihedrals(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_dihedral=self.n_dihedrals+1,
            monomer_size=monomer_size
        )

        self.dihedral_array = np.append(self.dihedral_array, subdihedral_array, axis=0)
        
    def add_as_double_bad_chain_double_filaments(self,
        positions, atom_type:int=1, subatom_type: int=6,
        add_nn_angles=True, add_nth_neigbour_bonds=False,
        step_ab:int=1, step_bc:int=1, step_cd:int=1,
        angle_type:int=2, dihedral_type:int=1, nn_angle_type:int=2,
        molecule_id: None | int = None,
        flexible_create=False, monomer_size: int=2
    ):
        cur_nbeads = self.n_beads
        n_monomers = len(positions) // 2
        
        self.positions = np.append(self.positions, positions, axis=0)

        new_atom_types = np.ones(len(positions), dtype=int)
        new_atom_types[::monomer_size] = atom_type
        new_atom_types[1::monomer_size] = subatom_type

        self.atom_types_array = np.append(self.atom_types_array, new_atom_types)
        if molecule_id is None:
            # already added one molecule in add_as_bad_chain, this should be same molecule
            molecule_id = self.n_molecules
        self.molecule_ids = np.append(self.molecule_ids, np.array([molecule_id]*len(positions), dtype=int))

        self._add_bad_topology(
            n_monomers=n_monomers, starting_index=cur_nbeads+1,
            add_nn_angles=add_nn_angles, add_nth_neigbour_bonds=add_nth_neigbour_bonds,
            step_ab=step_ab, step_bc=step_bc, step_cd=step_cd,
            angle_type=angle_type, dihedral_type=dihedral_type, nn_angle_type=nn_angle_type,
            flexible_create=flexible_create,
            monomer_size=monomer_size,
            add_dihedrals=False
        )

        subbond_array = DoubleBADUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_bond=self.n_bonds+1,
            bad_bond_type=1, dipole_bond_type=3, chain_bond_type=4, step_ab=step_ab, add_bad_bonds=False,
            monomer_size=monomer_size, add_chain_bonds=True
        )

        self.bond_array = np.append(self.bond_array, subbond_array, axis=0)

        subangle_array = DoubleBADUtils._subatom_angles_double_filaments(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_angle=self.n_angles+1,
            monomer_size=monomer_size
        )
        self.angle_array = np.append(self.angle_array, subangle_array, axis=0)

    def add_as_triple_bead_patch_chain(self,
        positions, bead_type: int=1, patch_type: int=6,
        backbone_bond_type: int=1, patch_bond_type: int=3, interpatch_bond_type: int=4,
        backbone_angle_type: int=2, patch_angle_type: int=3,
        add_interpatch_bonds: bool=False,
        two_sided_patch_angle: bool=False,
        molecule_id: None | int = None,
        monomer_size: int=4
    ):
        """Add a filament whose monomers are 3 collinear backbone beads plus one patch on the middle bead.

        Particle order per monomer is [b0, b1, b2, patch], see TripleBeadPatchUtils.get_straight_filament.

        Topology (types chosen to reuse double_filament2d_bad_filament_block.lmp):
            bonds:  b-b along the whole backbone (incl. across monomers) -> backbone_bond_type (1)
                    b1-patch                                             -> patch_bond_type (3)
                    patch-patch of consecutive monomers (optional, off)  -> interpatch_bond_type (4)
            angles: every consecutive backbone triplet                   -> backbone_angle_type (2)
                    patch-b1-b2 (+ patch-b1-b0 if two_sided_patch_angle) -> patch_angle_type (3)
            no dihedrals
        """
        if monomer_size != 4:
            raise ValueError('add_as_triple_bead_patch_chain only supports monomer_size=4 (3 beads + 1 patch).')

        positions = np.asarray(positions, dtype=float)
        if len(positions) % monomer_size:
            raise ValueError(f'Number of positions ({len(positions)}) must be a multiple of the monomer size ({monomer_size}).')

        cur_nbeads = self.n_beads
        n_monomers = len(positions) // monomer_size

        self.positions = np.append(self.positions, positions, axis=0)

        new_atom_types = np.full(len(positions), bead_type, dtype=int)
        new_atom_types[3::monomer_size] = patch_type
        self.atom_types_array = np.append(self.atom_types_array, new_atom_types)

        if molecule_id is None:
            molecule_id = self.n_molecules + 1
        self.molecule_ids = np.append(self.molecule_ids, np.array([molecule_id]*len(positions), dtype=int))

        # 1-based atom indices of the four particles of every monomer
        starts = cur_nbeads + 1 + monomer_size*np.arange(n_monomers)
        b0, b1, b2, patch = starts, starts + 1, starts + 2, starts + 3
        backbone = np.column_stack((b0, b1, b2)).ravel() # backbone beads in chain order

        # bonds: [bond_id, type, atom1, atom2]
        bond_blocks = [
            (backbone_bond_type, backbone[:-1], backbone[1:]),
            (patch_bond_type, b1, patch),
        ]
        if add_interpatch_bonds:
            bond_blocks.append((interpatch_bond_type, patch[:-1], patch[1:]))

        for bond_type, atom1, atom2 in bond_blocks:
            if not len(atom1):
                continue
            new_bonds = np.zeros((len(atom1), 4), dtype=int)
            new_bonds[:, 0] = np.arange(self.n_bonds + 1, self.n_bonds + 1 + len(atom1))
            new_bonds[:, 1] = bond_type
            new_bonds[:, 2] = atom1
            new_bonds[:, 3] = atom2
            self.bond_array = np.append(self.bond_array, new_bonds, axis=0)

        # angles: [angle_id, type, atom1, atom2 (centre), atom3]
        angle_blocks = [
            (backbone_angle_type, backbone[:-2], backbone[1:-1], backbone[2:]),
            (patch_angle_type, patch, b1, b2),
        ]
        if two_sided_patch_angle:
            angle_blocks.append((patch_angle_type, patch, b1, b0))
        for angle_type, atom1, atom2, atom3 in angle_blocks:
            if not len(atom1):
                continue
            new_angles = np.zeros((len(atom1), 5), dtype=int)
            new_angles[:, 0] = np.arange(self.n_angles + 1, self.n_angles + 1 + len(atom1))
            new_angles[:, 1] = angle_type
            new_angles[:, 2] = atom1
            new_angles[:, 3] = atom2
            new_angles[:, 4] = atom3
            self.angle_array = np.append(self.angle_array, new_angles, axis=0)

    def add_as_triple_bead_patch_diffuse_chain(self,
        positions, bead_type: int=1, patch_type: int=6, diffuse_type: int=9,
        backbone_bond_type: int=1, patch_bond_type: int=3, diffuse_bond_type: int=5,
        backbone_angle_type: int=2, patch_angle_type: int=3, diffuse_angle_type: int=4,
        molecule_id: None | int = None,
        monomer_size: int=7
    ):
        """Add a filament whose monomers are 3 collinear backbone beads, one patch on the middle bead and one
        diffuse bead opposite each backbone bead (on the other side than the patch).

        Particle order per monomer is [b0, b1, b2, patch, d0, d1, d2], see TripleBeadPatchDiffuseUtils.

        Topology (types match double_filament2d_2patches_bad_filament_block.lmp):
            bonds:  b-b along the whole backbone (incl. across monomers) -> backbone_bond_type (1)
                    b1-patch                                             -> patch_bond_type (3)
                    b_i-d_i                                              -> diffuse_bond_type (5)
            angles: every consecutive backbone triplet                   -> backbone_angle_type (2)
                    patch-b1-b2                                          -> patch_angle_type (3)
                    d0-b0-b1, d1-b1-b2, d2-b2-b1                         -> diffuse_angle_type (4)
            no dihedrals; no bonds between diffuse beads (they would make the bond/react superimpose ambiguous)
        """
        if monomer_size != 7:
            raise ValueError('add_as_triple_bead_patch_diffuse_chain only supports monomer_size=7.')

        positions = np.asarray(positions, dtype=float)
        if len(positions) % monomer_size:
            raise ValueError(f'Number of positions ({len(positions)}) must be a multiple of the monomer size ({monomer_size}).')

        cur_nbeads = self.n_beads
        n_monomers = len(positions) // monomer_size

        self.positions = np.append(self.positions, positions, axis=0)

        new_atom_types = np.full(len(positions), bead_type, dtype=int)
        new_atom_types[3::monomer_size] = patch_type
        for kk in (4, 5, 6):
            new_atom_types[kk::monomer_size] = diffuse_type
        self.atom_types_array = np.append(self.atom_types_array, new_atom_types)

        if molecule_id is None:
            molecule_id = self.n_molecules + 1
        self.molecule_ids = np.append(self.molecule_ids, np.array([molecule_id]*len(positions), dtype=int))

        # 1-based atom indices of the particles of every monomer
        starts = cur_nbeads + 1 + monomer_size*np.arange(n_monomers)
        b0, b1, b2, patch = starts, starts + 1, starts + 2, starts + 3
        d0, d1, d2 = starts + 4, starts + 5, starts + 6
        backbone = np.column_stack((b0, b1, b2)).ravel() # backbone beads in chain order

        # bonds: [bond_id, type, atom1, atom2]
        bond_blocks = [
            (backbone_bond_type, backbone[:-1], backbone[1:]),
            (patch_bond_type, b1, patch),
            (diffuse_bond_type, b0, d0),
            (diffuse_bond_type, b1, d1),
            (diffuse_bond_type, b2, d2),
        ]
        for bond_type, atom1, atom2 in bond_blocks:
            if not len(atom1):
                continue
            new_bonds = np.zeros((len(atom1), 4), dtype=int)
            new_bonds[:, 0] = np.arange(self.n_bonds + 1, self.n_bonds + 1 + len(atom1))
            new_bonds[:, 1] = bond_type
            new_bonds[:, 2] = atom1
            new_bonds[:, 3] = atom2
            self.bond_array = np.append(self.bond_array, new_bonds, axis=0)

        # angles: [angle_id, type, atom1, atom2 (centre), atom3]
        angle_blocks = [
            (backbone_angle_type, backbone[:-2], backbone[1:-1], backbone[2:]),
            (patch_angle_type, patch, b1, b2),
            (diffuse_angle_type, d0, b0, b1),
            (diffuse_angle_type, d1, b1, b2),
            (diffuse_angle_type, d2, b2, b1),
        ]
        for angle_type, atom1, atom2, atom3 in angle_blocks:
            if not len(atom1):
                continue
            new_angles = np.zeros((len(atom1), 5), dtype=int)
            new_angles[:, 0] = np.arange(self.n_angles + 1, self.n_angles + 1 + len(atom1))
            new_angles[:, 1] = angle_type
            new_angles[:, 2] = atom1
            new_angles[:, 3] = atom2
            new_angles[:, 4] = atom3
            self.angle_array = np.append(self.angle_array, new_angles, axis=0)

    def add_as_double_filament_with_two_patches(self,
        positions, atom_type:int=1, patch1_type: int=6, patch2_type: int=9,
        add_nn_angles=True, add_nth_neigbour_bonds=False,
        step_ab:int=1, step_bc:int=1, step_cd:int=1,
        angle_type:int=2, dihedral_type:int=1, nn_angle_type:int=2,
        molecule_id: None | int = None,
        flexible_create=False, monomer_size: int=3
    ):
        cur_nbeads = self.n_beads
        n_monomers = len(positions) // monomer_size

        self.positions = np.append(self.positions, positions, axis=0)

        new_atom_types = np.ones(len(positions), dtype=int)
        new_atom_types[::monomer_size] = atom_type
        new_atom_types[1::monomer_size] = patch1_type
        new_atom_types[2::monomer_size] = patch2_type

        self.atom_types_array = np.append(self.atom_types_array, new_atom_types)
        if molecule_id is None:
            # already added one molecule in add_as_bad_chain, this should be same molecule
            molecule_id = self.n_molecules
        self.molecule_ids = np.append(self.molecule_ids, np.array([molecule_id]*len(positions), dtype=int))

        self._add_bad_topology(
            n_monomers=n_monomers, starting_index=cur_nbeads+1,
            add_nn_angles=add_nn_angles, add_nth_neigbour_bonds=add_nth_neigbour_bonds,
            step_ab=step_ab, step_bc=step_bc, step_cd=step_cd,
            angle_type=angle_type, dihedral_type=dihedral_type, nn_angle_type=nn_angle_type,
            flexible_create=flexible_create,
            monomer_size=monomer_size,
            add_dihedrals=False
        ) #creates bonds, angles for main filament

        subbond_array = DoubleBADTwoPatchesUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_bond=self.n_bonds+1,
            bad_bond_type=1, dipole_bond_type=3, chain_bond_type=4,
            dipole_second_bond_type=5, second_chain_bond_type=6, step_ab=step_ab, add_bad_bonds=False,
            monomer_size=monomer_size, add_chain_bonds=False
        )

        self.bond_array = np.append(self.bond_array, subbond_array, axis=0)

        subangle_array = DoubleBADTwoPatchesUtils._subatom_angles(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_angle=self.n_angles+1,
            monomer_size=monomer_size, angle_type_patch1=3, angle_type_patch2=4
        )
        self.angle_array = np.append(self.angle_array, subangle_array, axis=0)


    def add_as_triple_bad_chain(self,
        positions, atom_type:int=1, subatom_type: int=6,
        add_nn_angles=False, add_nth_neigbour_bonds=False,
        step_ab:int=1, step_bc:int=1, step_cd:int=1,
        angle_type:int=1, dihedral_type:int=1, nn_angle_type:int=2,
        molecule_id: None | int = None,
        flexible_create=False, monomer_size: int=3
    ):  
        cur_nbeads = self.n_beads

        n_monomers = len(positions) // 3

        self.positions = np.append(self.positions, positions, axis=0)

        new_atom_types = np.ones(len(positions), dtype=int)
        new_atom_types[::monomer_size] = atom_type
        new_atom_types[1::monomer_size] = subatom_type
        new_atom_types[2::monomer_size] = subatom_type + monomer_size

        self.atom_types_array = np.append(self.atom_types_array, new_atom_types)
        if molecule_id is None:
            # already added one molecule in add_as_bad_chain, this should be same molecule
            molecule_id = self.n_molecules
        self.molecule_ids = np.append(self.molecule_ids, np.array([molecule_id]*len(positions), dtype=int))

        self._add_bad_topology(
            n_monomers=n_monomers, starting_index=cur_nbeads+1,
            add_nn_angles=add_nn_angles, add_nth_neigbour_bonds=add_nth_neigbour_bonds,
            step_ab=step_ab, step_bc=step_bc, step_cd=step_cd,
            angle_type=angle_type, dihedral_type=dihedral_type, nn_angle_type=nn_angle_type,
            flexible_create=flexible_create,
            monomer_size=monomer_size
        )

        # self.bond_array = np.zeros((0, 4), dtype=int)
        self.angle_array = np.zeros((0, 5), dtype=int)
        self.dihedral_array = np.zeros((0, 6), dtype=int)

        subbond_array = DoubleBADUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_bond=self.n_bonds+1,
            bad_bond_type=1, dipole_bond_type=3, chain_bond_type=4, step_ab=step_ab, add_bad_bonds=False,
            monomer_size=monomer_size, add_scaffolding=True
        )

        self.bond_array = np.append(self.bond_array, subbond_array, axis=0)

        subangle_array = DoubleBADUtils._subatom_angles(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_angle=self.n_angles+1,
            monomer_size=monomer_size
        )
        # self.angle_array = np.append(self.angle_array, subangle_array, axis=0)

        # add dihedrals (polar interaction) between monomers
        subdihedral_array = DoubleBADUtils._subatom_dihedrals(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_dihedral=self.n_dihedrals+1,
            monomer_size=monomer_size
        )

        # self.dihedral_array = np.append(self.dihedral_array, subdihedral_array, axis=0)

        attractor_bond_array = TripleBADUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=cur_nbeads+1, starting_bond=self.n_bonds+1, triangle_scaffold=True
        )

        self.bond_array = np.append(self.bond_array, attractor_bond_array, axis=0)

        subangles = TripleBADUtils._triplet_angles(
            n_monomers=n_monomers,
            starting_index=cur_nbeads+1,
            starting_angle=self.n_angles+1,
            monomer_size=monomer_size
        )

        # self.angle_array = np.append(self.angle_array, subangles, axis=0)

        subdihedrals = TripleBADUtils._subatom_dihedrals(
            n_monomers=n_monomers,
            starting_index=cur_nbeads+1,
            starting_dihedral=self.n_dihedrals+1,
            monomer_size=monomer_size
        )

        # self.dihedral_array = np.append(self.dihedral_array, subdihedrals, axis=0)

    def _add_as_escrt_chain(self,
        positions: np.ndarray,
        use_rigid: bool=False
    ):
        if len(positions) % 3 != 0:
            raise ValueError(f"Positions must consist of triplets, therefore a multiple of 3, got {len(positions)} positions.")
        
        n_triplets = len(positions) // 3
        mol_ids, atom_types = _ESCRTUtils.mols_and_types(n_triplets)

        # bonds come without bond_id in front so add it here
        escrt_bonds = np.array(_ESCRTUtils.get_bonds(
            N=n_triplets, use_rigid=use_rigid
        ), dtype=int)
        bond_array = np.zeros((len(escrt_bonds), 4), dtype=int)
        bond_array[:, 0] = np.arange(self.n_bonds+1, self.n_bonds+1+len(escrt_bonds))
        bond_array[:, 1:] = escrt_bonds
        bond_array[:, -2:] += self.n_beads  # add current number of beads to atom indices

        bond_lengths = np.array(_ESCRTUtils.get_bond_lengths(
            coordinates=positions, use_rigid=use_rigid
        ))

        self.positions = np.append(self.positions, positions, axis=0)
        self.atom_types_array = np.append(self.atom_types_array, atom_types)
        self.molecule_ids = np.append(self.molecule_ids, np.array(mol_ids, dtype=int))
        self.bond_array = np.append(self.bond_array, bond_array, axis=0)

        return bond_lengths
    
    def _add_as_escrt_double_patch_chain(self,
        positions: np.ndarray,
        use_rigid: bool=False
    ):
        if len(positions) % 3 != 0:
            raise ValueError(f"Positions must consist of triplets, therefore a multiple of 3, got {len(positions)} positions.")
        
        n_triplets = len(positions) // 3
        mol_ids, atom_types = _ESCRTUtils.mols_and_types(n_triplets) #can stay, just assigngs mols and types the same way as it would for an escrt chain

        # bonds come without bond_id in front so add it here
        escrt_bonds = np.array(_ESCRTUtils.get_bonds(
            N=n_triplets, use_rigid=use_rigid
        ), dtype=int)
        bond_array = np.zeros((len(escrt_bonds), 4), dtype=int)
        bond_array[:, 0] = np.arange(self.n_bonds+1, self.n_bonds+1+len(escrt_bonds))
        bond_array[:, 1:] = escrt_bonds
        bond_array[:, -2:] += self.n_beads  # add current number of beads to atom indices

        bond_lengths = np.array(_ESCRTUtils.get_bond_lengths(
            coordinates=positions, use_rigid=use_rigid
        ))

        self.positions = np.append(self.positions, positions, axis=0)
        self.atom_types_array = np.append(self.atom_types_array, atom_types)
        self.molecule_ids = np.append(self.molecule_ids, np.array(mol_ids, dtype=int))
        self.bond_array = np.append(self.bond_array, bond_array, axis=0)

        return bond_lengths

    def add_escrt_chain(self,
        n_triplets: int, n_circumference: int,
        use_rigid: bool=False,
        radius_multiplier: float=1,
        position_offset: np.ndarray[float] | None = None
    ):
        if position_offset is None:
            position_offset = np.array([0, 0, 0], dtype=float)
        position_offset = np.array(position_offset, dtype=float)

        if position_offset.shape != (3,):
            raise ValueError(f"Position offset must be a 3D vector, got shape {position_offset.shape}.")
    
        positions = _ESCRTUtils.get_coordinates(
            N=n_triplets, Nc=n_circumference,
            radius_multiplier=radius_multiplier,
            use_rigid=use_rigid,
            position_offset=position_offset
        )
        
        return self._add_as_escrt_chain(
            positions=positions, use_rigid=use_rigid
        )

    @staticmethod
    def get_centered_cylinder(radius, length, bead_radius, verbose=False):
        n_beads = int(2*np.pi*radius/(2*bead_radius))
        n_rings = int(length/(2*bead_radius))
        
        coordinates = np.zeros((n_beads*n_rings, 3), dtype=np.float32)
        y_coordinates, y_step = np.linspace(-length/2, length/2, n_rings, endpoint=False, retstep=True)
        coordinates[:, 1] = np.repeat(y_coordinates, n_beads)
        angles = np.linspace(0, 2*np.pi, n_beads, endpoint=False)
        coordinates[:, 0] = np.tile(radius * np.cos(angles), n_rings).flatten()
        coordinates[:, 2] = np.tile(radius * np.sin(angles), n_rings).flatten()

        true_rdist = np.mean(np.linalg.norm(
            coordinates[:n_beads, [0, 2]] - np.roll(coordinates[:n_beads, [0, 2]], 1, axis=0), 
            axis=-1
        ))

        if verbose:
            print(f'True in ring distance: {true_rdist}')
            print(f'True ring to ring distance: {y_step}')

        return coordinates
        
    def add_centered_ylz_cylinder(
        self, 
        n_total_approx: int, n_circumference: int,
        particle_diameter: float = 1.0,
        particle_type: int = 5,
        molecule_id: int = 0,
        overwrite_box: bool = True
    ):
        coordinates, orientations, quaternions, box_dimensions = _YLZUtils.get_ylz_centered_cylinder(
            n_total_approx=n_total_approx,
            n_circumference=n_circumference,
            particle_diameter=particle_diameter,
        )

        self.ellipsoid_indices += tuple(range(self.n_beads, self.n_beads + len(coordinates)))

        self.positions = np.append(self.positions, coordinates, axis=0)
        self.atom_types_array = np.append(self.atom_types_array, np.ones(len(coordinates), dtype=int) * particle_type, axis=0)
        self.molecule_ids = np.append(self.molecule_ids, np.ones(len(coordinates), dtype=int) * molecule_id, axis=0)
        self.quaternions = np.append(self.quaternions, quaternions, axis=0)

        if overwrite_box:
            self.box = np.array([
                [-box_dimensions[0]/2., box_dimensions[0]/2.],
                [-box_dimensions[1]/2., box_dimensions[1]/2.],
                [-box_dimensions[2]/2., box_dimensions[2]/2.]
            ], dtype=float)

    def load_ylz_trajectory(self, initialisation_trajectory: str | Path, overwrite_box: bool = True, ylz_type: int = 5):
        initialised_structure_path = Path(initialisation_trajectory).resolve()
        if not initialised_structure_path.exists():
            raise FileNotFoundError(f"Could not YLZ initialisation file at '{initialised_structure_path}'.")

        # load initialised structure
        BOX_RE = re.compile(r'ITEM:\sBOX.*([\s\n0-9\.e\-\+]*)')
        TRAJ_RE = re.compile(r'ITEM:\sATOMS.*([\s\n0-9\.e\-\+]*)')
        
        file_content = open(initialised_structure_path, 'r').read()
        match_iterator = re.finditer(TRAJ_RE, file_content)
        box_iterator = re.finditer(BOX_RE, file_content)

        # move through trajectory and get last frame
        indices, positions, quaternions, boxes = [], [], [], []
        for data_block, box_block in zip(match_iterator, box_iterator):
            data_sio = StringIO(data_block.group(1))
            indices.append(np.loadtxt(data_sio, usecols=0, dtype=int))
            data_sio = StringIO(data_block.group(1))
            positions.append(np.loadtxt(data_sio, usecols=[3, 4, 5], dtype=float))
            data_sio = StringIO(data_block.group(1))
            quaternions.append(np.loadtxt(data_sio, usecols=[9, 10, 11, 12]))
            boxes.append(np.loadtxt(StringIO(box_block.group(1)), dtype=float))

        indices, positions, quaternions, box = indices[-1], positions[-1], quaternions[-1], boxes[-1]
        if overwrite_box:
            self.box = box

        self.ellipsoid_indices += tuple(range(self.n_beads, self.n_beads + len(positions)))
        self.positions = np.append(self.positions, positions, axis=0)
        self.atom_types_array = np.append(self.atom_types_array, np.ones(len(positions), dtype=int) * ylz_type, axis=0)
        self.molecule_ids = np.append(self.molecule_ids, np.zeros(len(positions), dtype=int), axis=0)
        self.quaternions = np.append(self.quaternions, quaternions, axis=0)

        return len(positions)

    @classmethod
    def write_connected_chain(cls,
        positions: np.ndarray, filename: Path,
        step_ab:int=1, step_bc:int=1, step_cd:int=1, add_nn_angles=False,
        angle_type:int=1, dihedral_type:int=1,
        write_angles:bool=True, write_dihedrals:bool=True,
    ):
        '''Historical function for creating and writing a connected chain to file.'''
        structure_initialiser = cls()
        structure_initialiser.add_as_bad_chain(
            positions=positions, add_nn_angles=add_nn_angles,
            step_ab=step_ab, step_bc=step_bc, step_cd=step_cd,
            angle_type=angle_type, dihedral_type=dihedral_type
        )

        structure_initialiser.write_to_file(filename, write_angles=write_angles, write_dihedrals=write_dihedrals)

    def _write_atoms(self, f, atom_style='molecular'):
        f.write(f'Atoms # style: {atom_style}\n\n')

        if atom_style == 'atomic':
            for line in range(self.n_beads):
                # id type x y z
                f.write(f"\t{line+1} {self.atom_types_array[line]} {self.positions[line, 0]} {self.positions[line, 1]} {self.positions[line, 2]}\n")
        elif atom_style == 'molecular': # TODO: this writes every atom as molecule 1 once its initiated, is this a problem
            for line in range(self.n_beads):
                # id mol_id type x y z
                f.write(f"\t{line+1} {self.molecule_ids[line]} {self.atom_types_array[line]} {self.positions[line, 0]} {self.positions[line, 1]} {self.positions[line, 2]} \n")
        elif atom_style == 'ellipsoid':
            # id type ellipsoid_flag density x y z
            for line in range(self.n_beads):
                f.write(f"\t{line+1} {self.atom_types_array[line]} {1 if line in self.ellipsoid_indices else 0} 1 {self.positions[line, 0]} {self.positions[line, 1]} {self.positions[line, 2]}\n")
        elif atom_style == 'hybrid_ellipsoid_molecular':
            # id type x y z ellipsoid_flag density mol_id
            for line in range(self.n_beads):
                f.write(f"\t{line+1} {self.atom_types_array[line]} {self.positions[line, 0]} {self.positions[line, 1]} {self.positions[line, 2]} {1 if line in self.ellipsoid_indices else 0} 1 {self.molecule_ids[line]}\n")
        else:
            raise ValueError(f"Unknown or not implemented atom style '{atom_style}' in StructureInitialiser._write_atoms.")
    
    def write_to_file(
            self,
            filename: Path, 
            atom_style:str='molecular',
            write_bonds:bool=True, write_angles:bool=True, write_dihedrals:bool=True,
            write_ellipsoids:bool=False,
            n_atom_types:int=None, n_bond_types:int=None, 
            n_angle_types:int=None, n_dihedral_types:int=None
        ):
        filename = Path(filename).resolve()

        n_beads = self.n_beads
        n_bonds = self.n_bonds
        n_angles = self.n_angles
        n_dihedrals = self.n_dihedrals

        if n_atom_types is None:
            n_atom_types = self.n_atom_types
        if n_bond_types is None:
            n_bond_types = self.n_bond_types
        if n_angle_types is None:
            n_angle_types = self.n_angle_types
        if n_dihedral_types is None:
            n_dihedral_types = self.n_dihedral_types

        with open(filename, 'w') as f:
            f.write('Automatically generated data file\n\n')
            f.write(f"{n_beads} atoms\n")
            if write_bonds:
                f.write(f"{n_bonds} bonds\n")
            if write_angles:
                f.write(f"{n_angles} angles\n")
            if write_dihedrals:
                f.write(f"{n_dihedrals} dihedrals\n")
            if write_ellipsoids:
                f.write(f"{len(self.ellipsoid_indices)} ellipsoids\n")
            f.write('\n')
            f.write(f"{n_atom_types} atom types\n")
            f.write(f"{n_bond_types} bond types\n")
            f.write(f"{n_angle_types} angle types\n")
            if write_dihedrals:
                f.write(f"{n_dihedral_types} dihedral types\n")
            f.write('\n')

            # write box bounds
            for ii, coord in enumerate(['x', 'y', 'z']):
                f.write(f"{self.box[ii, 0]} {self.box[ii, 1]} {coord}lo {coord}hi\n")
            f.write('\n')

            # write atoms using writer to handle different atom styles
            self._write_atoms(f, atom_style=atom_style)

            if write_bonds:
                f.write('\nBonds\n\n')
                f.write('\t')
                np.savetxt(f, self.bond_array, fmt='%u', newline='\n\t', delimiter=' ')

            if write_angles and n_angles>0:
                f.write('\nAngles\n\n')
                f.write('\t')
                np.savetxt(f, self.angle_array, fmt='%u', newline='\n\t', delimiter=' ')

            if write_dihedrals:
                f.write('\nDihedrals\n\n')
                f.write('\t')
                np.savetxt(f, self.dihedral_array, fmt='%u', newline='\n\t', delimiter=' ')

            if write_ellipsoids:
                f.write('\nEllipsoids\n\n')
                f.write('\t')
                ellipsoid_array = np.hstack((
                    np.tile(ALMOST_SPHERICAL_ELLIPSOID_DIMENSIONS, (self.quaternions.shape[0], 1)),
                    self.quaternions
                ))
                ellipsoid_array = np.append(np.arange(1, len(self.ellipsoid_indices)+1)[:, np.newaxis], ellipsoid_array, axis=1)
                np.savetxt(
                    f, ellipsoid_array, 
                    fmt='%u '+r' '.join(
                        [r'%.5e']*(ellipsoid_array.shape[1]-1)
                    ),
                    newline='\n\t', delimiter=' ')

    def write_to_template_file(
            self,
            filename: Path,
            n_fragments: int=1
        ):
        filename = Path(filename).resolve()

        with open(filename, 'w') as f:
            f.write('# Automatically generated molecule file \n\n')
            f.write(f"{self.n_beads} atoms\n")
            f.write(f"{self.n_bonds} bonds\n")
            f.write(f"{self.n_angles} angles\n")
            f.write(f"{self.n_dihedrals} dihedrals\n")
            f.write(f"{n_fragments} fragments\n")

            f.write(f"\n\nTypes\n\n")
            for ii, type in enumerate(self.atom_types_array):
                f.write(f"{ii+1} {type}\n")

            f.write(f"\n\nBonds\n\n")
            np.savetxt(f, self.bond_array, fmt='%u', newline='\n', delimiter=' ')

            if self.n_angles > 0:
                f.write(f"\n\nAngles\n\n")
                np.savetxt(f, self.angle_array, fmt='%u', newline='\n', delimiter=' ')

            if self.n_dihedrals > 0:
                f.write(f"\n\nDihedrals\n\n")
                np.savetxt(f, self.dihedral_array, fmt='%u', newline='\n', delimiter=' ')

    @staticmethod
    def write_oligomer_post_file(coordinates, output_path: Path, template_path=None):
        if template_path is None:
            template_path = Path(__file__).resolve().parent.parent.parent / "lammps_chemistry_templates" / "nn_treadmilling_templates" / "OligomerOn_post_template.txt"

        coords = [JinjaCoord(index=ii+1, x=coord[0], y=coord[1], z=coord[2]) for ii, coord in enumerate(coordinates)]

        with open(template_path, 'r') as f:
            template = jinja2.Template(f.read())

        template_output = template.render(coords=coords)

        with open(output_path, 'w') as f:
            f.write(template_output)

class BADUtils:
    '''Utility class for writing BAD (bonds, angles dihedrals) files for the structure initialiser.'''
    @staticmethod
    def get_centered_line(length, r_zero):
        x_coords= np.arange(-length/2., length/2., r_zero)
        n_beads = len(x_coords)
        positions = np.zeros((n_beads, 3), dtype=float)
        positions[:, 0] = x_coords

        return positions
    
    @staticmethod
    def get_centered_helix(r_zero, length, radius, rev_dist):
        helix_a = radius
        helix_b = rev_dist/(2*np.pi) # dx = 2*pi*b for one revolution

        t_max = length/(2*helix_b) # x_max = length/2 = helix_b * t_min
        arclength = 2*t_max * np.sqrt(helix_a**2 + helix_b**2) # arclength = T * sqrt(a^2 + b^2)

        s_vals = np.arange(0, arclength, r_zero) # dirty trick to get number of beads
        t_vals = s_vals/(np.sqrt(helix_a**2 + helix_b**2)) - t_max
        n_beads = len(t_vals)

        positions = np.zeros((n_beads, 3), dtype=float)
        positions[:, 0] = helix_b * t_vals
        positions[:, 1] = helix_a * np.cos(t_vals)
        positions[:, 2] = helix_a * np.sin(t_vals)

        return positions
    
    @staticmethod
    def _get_bond_array(n_monomers: int, starting_index:int=1, starting_bond:int=1, bond_type:int=1, step_ab:int=1, monomer_size:int=1):
        # array of format [bond_id, type, atom1, atom2], 1 based indexing
        n_bonds = n_monomers - step_ab

        bond_array = np.zeros((n_bonds, 4), dtype=int)
        bond_array[:, 0] = np.arange(starting_bond, n_bonds+starting_bond)
        bond_array[:, 1] = bond_type
        bond_array[:, 2] = np.arange(
            starting_index, 
            starting_index+(n_monomers - step_ab)*monomer_size, 
            monomer_size
        ) # atom1
        bond_array[:, 3] = bond_array[:, 2] + (step_ab * monomer_size) # atom2

        return bond_array
    
    @staticmethod
    def _nangles_from_steps(n_monomers: int, step_ab: int, step_bc: int):
        return n_monomers - step_ab - step_bc

    @staticmethod
    def _get_angle_array(
        n_monomers:int,
        step_ab:int=1, step_bc:int=1,
        angle_type:int=1,
        starting_index:int=1, starting_angle:int=1,
        monomer_size:int=1
    ):
        """Get array of indices defining the angles between three beads.

        Args:
            n_beads (int): _description_
            step_ab (int, optional): _description_. Defaults to 1.
            step_bc (int, optional): _description_. Defaults to 1.
            angle_type (int, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """
        n_angles = BADUtils._nangles_from_steps(n_monomers, step_ab, step_bc)

        if n_angles < 1:
            raise ValueError(f'Cannot define any angles for {n_monomers} with steps {step_ab} and {step_bc}')

        angle_array = np.zeros((n_angles, 5), dtype=int)
        angle_array[:, 0] = np.arange(starting_angle, n_angles+starting_angle)
        angle_array[:, 1] = angle_type
        
        angle_array[:, 2] = np.arange(
            starting_index, 
            starting_index+(n_monomers - step_ab - step_bc)*monomer_size, 
            monomer_size
        ) # atom1
        angle_array[:, 3] = angle_array[:, 2] + (step_ab * monomer_size) # atom2
        angle_array[:, 4] = angle_array[:, 3] + (step_bc * monomer_size) # atom3

        return angle_array
    
    @staticmethod
    def _ndihedrals_from_steps(n_monomers: int, step_ab: int, step_bc: int, step_cd):
        return n_monomers - step_ab - step_bc - step_cd
    
    @staticmethod
    def _get_dihedral_array(
        n_monomers:int, 
        step_ab:int=1, step_bc:int=1, step_cd:int=1,
        dihedral_type:int=1, 
        starting_index:int=1, starting_dihedral:int=1,
        monomer_size: int=1
    ):
        """_summary_

        Args:
            n_beads (int): _description_
            step_ab (int, optional): _description_. Defaults to 1.
            step_bc (int, optional): _description_. Defaults to 1.
            step_cd (int, optional): _description_. Defaults to 1.
            dihedral_type (int, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """
        n_dihedrals = BADUtils._ndihedrals_from_steps(n_monomers, step_ab, step_bc, step_cd)

        if n_dihedrals < 1:
            raise ValueError(f'Cannot define any dihedrals for {n_monomers} with steps {step_ab}, {step_bc} and {step_cd}')

        dihedral_array = np.zeros((n_dihedrals, 6), dtype=int)
        dihedral_array[:, 0] = np.arange(starting_dihedral, n_dihedrals+starting_dihedral)
        dihedral_array[:, 1] = dihedral_type
        
        dihedral_array[:, 2] = np.arange(
            starting_index, 
            starting_index+(n_monomers - step_ab - step_bc - step_cd)*monomer_size, 
            monomer_size
        ) # atom1
        dihedral_array[:, 3] = dihedral_array[:, 2] + (step_ab * monomer_size) # atom2
        dihedral_array[:, 4] = dihedral_array[:, 3] + (step_bc * monomer_size) # atom3
        dihedral_array[:, 5] = dihedral_array[:, 4] + (step_cd * monomer_size) # atom4

        return dihedral_array

    @staticmethod
    def get_helix_properties(helix, step_ab:int=1, step_bc:int=1, step_cd:int=1, verbose:bool=False):
        # get equilibrium properties of this equilibrium_helix
        angle_indices = BADUtils._get_angle_array(len(helix), step_ab, step_bc)
        angles = get_angles(
            helix[angle_indices[:, 2]-1, :] - helix[angle_indices[:, 3]-1, :], # BA
            helix[angle_indices[:, 4]-1, :] - helix[angle_indices[:, 3]-1, :] # BC
        )

        dihedral_indices = BADUtils._get_dihedral_array(len(helix), step_ab, step_bc, step_cd)
        dihedrals = get_dihedrals(
            helix[dihedral_indices[:, 3]-1, :] - helix[dihedral_indices[:, 2]-1, :], # AB
            helix[dihedral_indices[:, 4]-1, :] - helix[dihedral_indices[:, 3]-1, :], # BC
            helix[dihedral_indices[:, 5]-1, :] - helix[dihedral_indices[:, 4]-1, :] # CD
        )
        
        if verbose:
            print(f"Found {angle_indices.shape[0]} angles")
            print(f"Found {dihedral_indices.shape[0]} dihedrals")
            print(f'Average bond angle: {np.mean(angles)}, min: {np.min(angles)}, max: {np.max(angles)}')
            print(f'Average dihedral angle: {np.mean(dihedrals)}, min: {np.min(dihedrals)}, max: {np.max(dihedrals)}')

        return np.mean(angles), np.mean(dihedrals)

class DoubleBADUtils:
    @staticmethod
    def get_centered_helix(r_zero, length, radius, rev_dist, patch_distance: float|None=None):
        """Get coordinates of a centered helix with 2 beads per monomer to simulate the inner structure of monomers.

        Args:
            r_zero (_type_): _description_
            length (_type_): _description_
            radius (_type_): _description_
            rev_dist (_type_): _description_
            patch_distance (float, optional): Distance that patches should be away from center of helix beads. Defaults to None and is then chosen to be r_zero/2.

        Returns:
            _type_: _description_
        """
        helix_a = radius
        helix_b = rev_dist/(2*np.pi) # dx = 2*pi*b for one revolution

        if patch_distance is None:
            patch_distance = r_zero / 2.

        t_max = length/(2*helix_b) # x_max = length/2 = helix_b * t_min
        arclength = 2*t_max * np.sqrt(helix_a**2 + helix_b**2) # arclength = T * sqrt(a^2 + b^2)

        s_vals = np.arange(0, arclength, r_zero) # dirty trick to get number of beads
        t_vals = s_vals/(np.sqrt(helix_a**2 + helix_b**2)) - t_max
        n_beads = len(t_vals)

        positions = np.zeros((n_beads*2, 3), dtype=float)
        positions[::2, 0] = helix_b * t_vals
        positions[::2, 1] = helix_a * np.cos(t_vals)
        positions[::2, 2] = helix_a * np.sin(t_vals)

        positions[1::2, 0] = helix_b * t_vals
        positions[1::2, 1] = (helix_a + patch_distance) * np.cos(t_vals) # second bead offset by one radius
        positions[1::2, 2] = (helix_a + patch_distance) * np.sin(t_vals)

        return positions
    
    @staticmethod
    def _get_bond_array(
            n_monomers: int,
            starting_index:int=1,
            starting_bond:int=1, 
            bad_bond_type:int=1,
            dipole_bond_type: int = 3, chain_bond_type: int = 4, 
            step_ab:int=1,
            add_bad_bonds: bool=True,
            add_scaffolding: bool=False,
            monomer_size: int=2,
            add_chain_bonds: bool=True
        ):
        # array of format [bond_id, type, atom1, atom2], 1 based indexing

        if add_bad_bonds:
            bad_bonds = BADUtils._get_bond_array(
                n_monomers=n_monomers,
                starting_index=starting_index,
                starting_bond=starting_bond,
                bond_type=bad_bond_type,
                step_ab=step_ab,
                monomer_size=monomer_size
            )
            n_bad_bonds = bad_bonds.shape[0]
        else:
            bad_bonds = np.empty((0, 4), dtype=int)
            n_bad_bonds = 0
        
        # now add bonds between the chain of added minibeads
        # mostly useful for adding all subbeads
        if add_chain_bonds:
            chain_bonds = BADUtils._get_bond_array(
                n_monomers=n_monomers,
                starting_index=starting_index+1,
                starting_bond=starting_bond+n_bad_bonds,
                bond_type=chain_bond_type,
                step_ab=step_ab,
                monomer_size=monomer_size
            )
            n_chain_bonds = chain_bonds.shape[0]
        else:
            n_chain_bonds = 0

        # finally add the bond that makes this structure dipolar
        # i.e. bond between minibead and bead
        n_dipole_bonds = n_monomers
        dipole_start = starting_bond + n_bad_bonds + n_chain_bonds
        dipole_bond_array = np.zeros((n_dipole_bonds, 4), dtype=int)
        dipole_bond_array[:, 0] = np.arange(dipole_start, dipole_start+n_dipole_bonds)
        dipole_bond_array[:n_dipole_bonds, 1] = dipole_bond_type
        dipole_bond_array[:n_dipole_bonds, 2] = np.arange(starting_index, starting_index+monomer_size*n_monomers, monomer_size)
        dipole_bond_array[:n_dipole_bonds, 3] = np.arange(starting_index+1, starting_index+1+monomer_size*n_monomers, monomer_size)

        if add_scaffolding:
            warnings.warn("Adding scaffolding bonds breaks treadmilling functionality.", UserWarning)
            # add scaffolding bonds to keep the structure straight
            # i.e. cross bonds between minibeads and beads of neighbouring monomers
            # unfortunately this breaks treadmilling

            n_dipole_bonds = (n_monomers-1)*2
            dipole_start = starting_bond + n_bad_bonds + n_chain_bonds + len(dipole_bond_array)
            scaffold_array = np.zeros((n_dipole_bonds, 4), dtype=int)
            scaffold_array[:, 0] = np.arange(dipole_start, dipole_start+n_dipole_bonds)
            scaffold_array[:n_dipole_bonds, 1] = 5
            scaffold_array[:n_dipole_bonds//2, 2] = np.arange(starting_index, starting_index+monomer_size*(n_monomers-1), monomer_size) # main atoms
            scaffold_array[:n_dipole_bonds//2, 3] = np.arange(starting_index+monomer_size+1, starting_index+monomer_size*n_monomers, monomer_size) # subatoms of next monomer
            scaffold_array[n_dipole_bonds//2:, 2] = np.arange(starting_index+monomer_size, starting_index+monomer_size*n_monomers, monomer_size) # next main atom
            scaffold_array[n_dipole_bonds//2:, 3] = np.arange(starting_index+1, starting_index-monomer_size+monomer_size*n_monomers, monomer_size) # subatoms

            dipole_bond_array = np.vstack((dipole_bond_array, scaffold_array))
            
        if add_chain_bonds:
            res = np.vstack((bad_bonds, chain_bonds, dipole_bond_array)) 
        else:
            res = np.vstack((bad_bonds, dipole_bond_array))

        return res

    @staticmethod
    def _subatom_angles(n_monomers:int, starting_index:int, starting_angle:int=1, monomer_size:int=2):
        n_angles = (n_monomers - 1) * 2
        # monomer_size = 2

        angle_array = np.zeros((n_angles, 5), dtype=int)
        angle_array[:, 0] = np.arange(starting_angle, n_angles+starting_angle)
        angle_array[:, 1] = 3
        
        angle_array[:n_angles//2, 2] = np.arange(
            starting_index+1, # use the subatom as base
            starting_index+(n_monomers - 1)*monomer_size,
            monomer_size
        ) # atom1
        angle_array[:n_angles//2, 3] = angle_array[:n_angles//2, 2] - 1 # atom2, main atom of subatom monomer
        angle_array[:n_angles//2, 4] = angle_array[:n_angles//2, 2] + monomer_size - 1 # atom3, main atom of second monomer

        angle_array[n_angles//2:, 2] = angle_array[:n_angles//2, 2] - 1 # atom1, main atom of first monomer
        angle_array[n_angles//2:, 3] = angle_array[n_angles//2:, 2] + monomer_size # atom2, main atom of second monomer
        angle_array[n_angles//2:, 4] = angle_array[n_angles//2:, 2] + monomer_size + 1 # atom3, subatom of second monomer

        return angle_array
    
    @staticmethod
    def _subatom_angles_double_filaments(n_monomers:int, starting_index:int, starting_angle:int=1, monomer_size:int=2):
        n_angles = n_monomers 
        # monomer_size = 2

        angle_array = np.zeros((n_angles, 5), dtype=int)
        angle_array[:, 0] = np.arange(starting_angle, n_angles+starting_angle)
        angle_array[:, 1] = 3 #type
        
        angle_array[:-1, 2] = np.arange(
            starting_index+1, # use the subatom as base
            starting_index+(n_monomers - 1)*monomer_size,
            monomer_size
        ) # atom1
        angle_array[:-1, 3] = angle_array[:-1, 2] - 1 # atom2, main atom of subatom monomer
        angle_array[:-1, 4] = angle_array[:-1, 2] + monomer_size - 1 # atom3, main atom of second monomer

        angle_array[-1, 2] = starting_index+(n_monomers - 1)*monomer_size + 1 # atom1, main atom of first monomer
        angle_array[-1, 3] = starting_index+(n_monomers - 1)*monomer_size  # atom2, main atom of second monomer
        angle_array[-1, 4] = starting_index+(n_monomers - 1)*monomer_size - monomer_size # atom3, subatom of second monomer

        return angle_array

    @staticmethod
    def _subatom_dihedrals(n_monomers, starting_index, starting_dihedral, monomer_size=2):
        # add dihedrals (polar interaction) between monomers
        n_dihedrals = n_monomers - 1
        # monomer_size = 2

        dihedral_array = np.zeros((n_dihedrals, 6), dtype=int)
        dihedral_array[:, 0] = np.arange(starting_dihedral, n_dihedrals+starting_dihedral)
        dihedral_array[:, 1] = 2
        
        dihedral_array[:, 2] = np.arange(
            starting_index+1, # start from subatom
            starting_index+(n_monomers - 1)*monomer_size, 
            monomer_size
        ) # atom1
        dihedral_array[:, 3] = dihedral_array[:, 2] - 1 # atom2, main atom of first monomer
        dihedral_array[:, 4] = dihedral_array[:, 2] + monomer_size - 1 # atom3, main atom of second monomer
        dihedral_array[:, 5] = dihedral_array[:, 2] + monomer_size # atom4, subatom of second monomer

        return dihedral_array

    @staticmethod
    def get_helix_properties(helix, step_ab:int=1, verbose:bool=False) -> tuple[dict[int, float], float, float]:
        if helix.shape[0] % 2:
            raise ValueError(f"The helix provided must consist of 2 x n_beads points, the first n_beads being the main beads and the second n_beads being the minibeads, got {helix.shape[0]} beads.")
        n_monomers = helix.shape[0] // 2

        bond_array = DoubleBADUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=1,
            starting_bond=1,
            step_ab=step_ab,
            add_bad_bonds=False
        )

        bond_types_lengths = {}
        for bond_type in np.unique(bond_array[:, 1]):
            is_type = bond_array[:, 1] == bond_type
            bond_lengths = np.linalg.norm(
                helix[bond_array[is_type, 2]-1, :] - 
                helix[bond_array[is_type, 3]-1, :],
                axis=-1
            )
            mean_bond_length = np.mean(bond_lengths)
            bond_types_lengths[int(bond_type)] = float(mean_bond_length)
            if verbose:
                print(f"Bond type {bond_type}: mean bond length: {mean_bond_length}.")

        angle_indices = DoubleBADUtils._subatom_angles(n_monomers, 1, 1)
        angles = get_angles(
            helix[angle_indices[:, 2]-1, :] - helix[angle_indices[:, 3]-1, :], # BA
            helix[angle_indices[:, 4]-1, :] - helix[angle_indices[:, 3]-1, :] # BC
        )

        dihedral_indices = DoubleBADUtils._subatom_dihedrals(n_monomers, 1, 1)
        dihedrals = get_dihedrals(
            helix[dihedral_indices[:, 3]-1, :] - helix[dihedral_indices[:, 2]-1, :], # AB
            helix[dihedral_indices[:, 4]-1, :] - helix[dihedral_indices[:, 3]-1, :], # BC
            helix[dihedral_indices[:, 5]-1, :] - helix[dihedral_indices[:, 4]-1, :] # CD
        )
        mean_angle = np.mean(angles)
        mean_dihedral = np.mean(dihedrals)

        if verbose:
            print("Subatom angles: ", mean_angle)
            print("Subatom dihedrals: ", mean_dihedral)

        return bond_types_lengths, mean_angle, mean_dihedral
    
class DoubleBADTwoPatchesUtils:
    @staticmethod
    def _get_bond_array(
            n_monomers: int,
            starting_index:int=1,
            starting_bond:int=1, 
            bad_bond_type:int=1,
            dipole_bond_type: int = 3, chain_bond_type: int = 4, 
            dipole_second_bond_type: int = 5, second_chain_bond_type: int = 6, 
            step_ab:int=1,
            add_bad_bonds: bool=True,
            add_scaffolding: bool=False,
            monomer_size: int=3,
            add_chain_bonds: bool=False
        ):
        # array of format [bond_id, type, atom1, atom2], 1 based indexing

        if add_bad_bonds:
            bad_bonds = BADUtils._get_bond_array(
                n_monomers=n_monomers,
                starting_index=starting_index,
                starting_bond=starting_bond,
                bond_type=bad_bond_type,
                step_ab=step_ab,
                monomer_size=monomer_size
            )
            n_bad_bonds = bad_bonds.shape[0]
        else:
            bad_bonds = np.empty((0, 4), dtype=int)
            n_bad_bonds = 0
        
        # now add bonds between the chain of added minibeads
        # mostly useful for adding all subbeads
        if add_chain_bonds:
            chain_bonds = BADUtils._get_bond_array(
                n_monomers=n_monomers,
                starting_index=starting_index+1,
                starting_bond=starting_bond+n_bad_bonds,
                bond_type=chain_bond_type,
                step_ab=step_ab,
                monomer_size=monomer_size
            )
            n_chain_bonds = chain_bonds.shape[0]
            
        else:
            n_chain_bonds = 0

        # finally add the bond that makes this structure dipolar
        # i.e. bond between minibead and bead
        n_dipole_bonds = n_monomers
        dipole_start = starting_bond + n_bad_bonds + n_chain_bonds
        dipole_bond_array = np.zeros((n_dipole_bonds, 4), dtype=int)
        dipole_bond_array[:, 0] = np.arange(dipole_start, dipole_start+n_dipole_bonds)
        dipole_bond_array[:n_dipole_bonds, 1] = dipole_bond_type
        dipole_bond_array[:n_dipole_bonds, 2] = np.arange(starting_index, starting_index+monomer_size*n_monomers, monomer_size)
        dipole_bond_array[:n_dipole_bonds, 3] = np.arange(starting_index+1, starting_index+1+monomer_size*n_monomers, monomer_size)
        
        if add_chain_bonds:
            #bonds connecting second patches
            chain_bonds2 = BADUtils._get_bond_array(
                n_monomers=n_monomers,
                starting_index=starting_index+2,
                starting_bond=dipole_start+n_dipole_bonds,
                bond_type=second_chain_bond_type,
                step_ab=step_ab,
                monomer_size=monomer_size
            )
            n_chain_bonds2 = chain_bonds2.shape[0]
        else:
            n_chain_bonds2 = 0
            
        # add bonds between main filament and second patches
        n_dipole_bonds2 = n_monomers
        dipole_start2 = dipole_start + n_dipole_bonds + n_chain_bonds2
        dipole_bond_array2 = np.zeros((n_dipole_bonds2, 4), dtype=int)
        dipole_bond_array2[:, 0] = np.arange(dipole_start2, dipole_start2+n_dipole_bonds2)
        dipole_bond_array2[:n_dipole_bonds2, 1] = dipole_second_bond_type
        dipole_bond_array2[:n_dipole_bonds2, 2] = np.arange(starting_index, starting_index+monomer_size*n_monomers, monomer_size)
        dipole_bond_array2[:n_dipole_bonds2, 3] = np.arange(starting_index+2, starting_index+2+monomer_size*n_monomers, monomer_size)
        
            
        if add_chain_bonds:
            res = np.vstack((bad_bonds, chain_bonds, dipole_bond_array, chain_bonds2, dipole_bond_array2)) 
        else:
            res = np.vstack((bad_bonds, dipole_bond_array, dipole_bond_array2))

        return res
    
    @staticmethod
    def _subatom_angles(n_monomers:int, starting_index:int, starting_angle:int=1, monomer_size:int=3, angle_type_patch1:int=3,angle_type_patch2:int=4):
        n_angles = n_monomers * 2
        # monomer_size = 2
        half_angles = int(n_angles/2)

        angle_array = np.zeros((n_angles, 5), dtype=int)
        angle_array[:, 0] = np.arange(starting_angle, n_angles+starting_angle)
        angle_array[:half_angles, 1] = angle_type_patch1
        angle_array[half_angles:, 1] = angle_type_patch2

        #angles for first patch
        angle_array[:half_angles-1, 2] = np.arange(
            starting_index+1, # use the subatom as base
            starting_index+(n_monomers - 1)*monomer_size,
            monomer_size
        ) # atom1
        angle_array[:half_angles-1, 3] = angle_array[:half_angles-1, 2] - 1 # atom2, main atom of subatom monomer
        angle_array[:half_angles-1, 4] = angle_array[:half_angles-1, 2] + monomer_size - 1 # atom3, main atom of second monomer

        angle_array[half_angles-1, 2] = starting_index+(n_monomers - 1)*monomer_size + 1 # atom1, main atom of first monomer
        angle_array[half_angles-1, 3] = starting_index+(n_monomers - 1)*monomer_size  # atom2, main atom of second monomer
        angle_array[half_angles-1, 4] = starting_index+(n_monomers - 1)*monomer_size - monomer_size # atom3, subatom of second monomer
        
        #angles for second patch
        angle_array[half_angles:-1, 2] = np.arange(
            starting_index+2, # use the subatom as base
            starting_index+(n_monomers - 1)*monomer_size,
            monomer_size
        ) 
        
        angle_array[half_angles:-1, 3] = angle_array[half_angles:-1, 2] - 2 # atom2, main atom of subatom monomer
        angle_array[half_angles:-1, 4] = angle_array[half_angles:-1, 2] + monomer_size - 2
        
        angle_array[-1, 2] = starting_index+(n_monomers - 1)*monomer_size + 2 # atom1, main atom of first monomer
        angle_array[-1, 3] = starting_index+(n_monomers - 1)*monomer_size  # atom2, main atom of second monomer
        angle_array[-1, 4] = starting_index+(n_monomers - 1)*monomer_size - monomer_size # atom3, subatom of second monomer

        return angle_array
    
    @staticmethod
    def get_helix_properties(helix, step_ab:int=1, verbose:bool=False) -> tuple[dict[int, float], float, float]:
        if helix.shape[0] % 3:
            raise ValueError(f"The helix provided must consist of 3 x n_beads points, the first n_beads being the main beads, the second n_beads being the first patches, and the third n_beads being the second patches, got {helix.shape[0]} beads.")
        n_monomers = helix.shape[0] // 3

        bond_array = DoubleBADTwoPatchesUtils._get_bond_array(
            n_monomers=n_monomers, starting_index=1,
            starting_bond=1,
            step_ab=step_ab,
            add_bad_bonds=False
        )

        bond_types_lengths = {}
        for bond_type in np.unique(bond_array[:, 1]):
            is_type = bond_array[:, 1] == bond_type
            bond_lengths = np.linalg.norm(
                helix[bond_array[is_type, 2]-1, :] - 
                helix[bond_array[is_type, 3]-1, :],
                axis=-1
            )
            mean_bond_length = np.mean(bond_lengths)
            bond_types_lengths[int(bond_type)] = float(mean_bond_length)
            if verbose:
                print(f"Bond type {bond_type}: mean bond length: {mean_bond_length}.")

        angle_indices = DoubleBADTwoPatchesUtils._subatom_angles(n_monomers, 1, 1)
        #print(angle_indices)
        angles = get_angles(
            helix[angle_indices[:, 2]-1, :] - helix[angle_indices[:, 3]-1, :], # BA
            helix[angle_indices[:, 4]-1, :] - helix[angle_indices[:, 3]-1, :] # BC
        )

        mean_angle = np.mean(angles)

        if verbose:
            print("Subatom angles: ", mean_angle)

        return bond_types_lengths, mean_angle

class TripleBADUtils:
    @staticmethod
    def _get_bond_array(n_monomers: int, starting_bond: int=1, starting_index: int=1, dipole_bond_type: int=3, chain_bond_type: int=4, monomer_size: int=3, step_ab: int=1, triangle_scaffold: bool=False):
        # finally add the bond that makes this structure dipolar
        # i.e. bond between minibead and bead
        chain_bonds = BADUtils._get_bond_array(
            n_monomers=n_monomers,
            starting_index=starting_index+2,
            starting_bond=starting_bond,
            bond_type=chain_bond_type,
            step_ab=step_ab,
            monomer_size=monomer_size
        )
        n_chain_bonds = chain_bonds.shape[0]
        
        n_dipole_bonds = n_monomers
        dipole_start = starting_bond + n_chain_bonds
        dipole_bond_array = np.zeros((n_dipole_bonds, 4), dtype=int)
        dipole_bond_array[:, 0] = np.arange(dipole_start, dipole_start+n_dipole_bonds)
        dipole_bond_array[:n_dipole_bonds, 1] = dipole_bond_type
        dipole_bond_array[:n_dipole_bonds, 2] = np.arange(starting_index, starting_index+monomer_size*n_monomers, monomer_size)
        dipole_bond_array[:n_dipole_bonds, 3] = np.arange(starting_index+2, starting_index+2+monomer_size*n_monomers, monomer_size)

        print(dipole_bond_array)

        if triangle_scaffold:
            triangle_start = starting_bond + n_chain_bonds + n_dipole_bonds
            triangle_bond_array = np.zeros((n_dipole_bonds, 4), dtype=int)
            triangle_bond_array[:, 0] = np.arange(triangle_start, triangle_start+n_dipole_bonds)
            triangle_bond_array[:n_dipole_bonds, 1] = 6
            triangle_bond_array[:n_dipole_bonds, 2] = np.arange(starting_index+1, starting_index+1+monomer_size*n_monomers, monomer_size)
            triangle_bond_array[:n_dipole_bonds, 3] = np.arange(starting_index+2, starting_index+2+monomer_size*n_monomers, monomer_size)

            dipole_bond_array = np.vstack((triangle_bond_array, dipole_bond_array))

        return np.vstack((chain_bonds, dipole_bond_array))

    @staticmethod
    def _subatom_angles(n_monomers:int, starting_index:int, starting_angle:int=1, monomer_size:int=2, angle_type:int=4):
        n_angles = (n_monomers - 1) * 2

        angle_array = np.zeros((n_angles, 5), dtype=int)
        angle_array[:, 0] = np.arange(starting_angle, n_angles+starting_angle)
        angle_array[:, 1] = 4
        
        angle_array[:n_angles//2, 2] = np.arange(
            starting_index+2, # use the subatom as base
            starting_index+(n_monomers - 1)*monomer_size,
            monomer_size
        ) # atom1
        angle_array[:n_angles//2, 3] = angle_array[:n_angles//2, 2] - 2 # atom2, main atom of subatom monomer
        angle_array[:n_angles//2, 4] = angle_array[:n_angles//2, 2] + monomer_size - 2 # atom3, main atom of second monomer

        angle_array[n_angles//2:, 2] = angle_array[:n_angles//2, 2] - 2 # atom1, main atom of first monomer
        angle_array[n_angles//2:, 3] = angle_array[n_angles//2:, 2] + monomer_size # atom2, main atom of second monomer
        angle_array[n_angles//2:, 4] = angle_array[n_angles//2:, 2] + monomer_size + 2 # atom3, subatom of second monomer

        return angle_array

    @staticmethod
    def _triplet_angles(n_monomers: int, starting_index: int, starting_angle: int=1, monomer_size: int=3):
        subatom_angles = TripleBADUtils._subatom_angles(
            n_monomers=n_monomers,
            starting_index=starting_index,
            starting_angle=starting_angle,
            monomer_size=monomer_size
        )
        n_subatom_angles = subatom_angles.shape[0]
        
        n_angles = n_monomers
        # monomer_size = 3

        angle_array = np.zeros((n_angles, 5), dtype=int)
        angle_array[:, 0] = np.arange(starting_angle+n_subatom_angles, n_angles+starting_angle+n_subatom_angles)
        angle_array[:, 1] = 5
        
        angle_array[:, 2] = np.arange(
            starting_index+1, # use the subatom as base
            starting_index+n_monomers*monomer_size,
            monomer_size
        ) # atom1
        angle_array[:, 3] = angle_array[:, 2] - 1 # atom2
        angle_array[:, 4] = angle_array[:, 2] + 1 # atom3

        return np.vstack((subatom_angles, angle_array))
    
    @staticmethod
    def _subatom_dihedrals(n_monomers, starting_index, starting_dihedral, monomer_size=2):
        # add dihedrals (polar interaction) between monomers
        n_dihedrals = n_monomers - 1
        # monomer_size = 2

        dihedral_array = np.zeros((n_dihedrals, 6), dtype=int)
        dihedral_array[:, 0] = np.arange(starting_dihedral, n_dihedrals+starting_dihedral)
        dihedral_array[:, 1] = 3
        
        dihedral_array[:, 2] = np.arange(
            starting_index+2, # start from subatom
            starting_index+(n_monomers - 1)*monomer_size,
            monomer_size
        ) # atom1
        dihedral_array[:, 3] = dihedral_array[:, 2] - 2 # atom2, main atom of first monomer
        dihedral_array[:, 4] = dihedral_array[:, 2] + monomer_size - 2 # atom3, main atom of second monomer
        dihedral_array[:, 5] = dihedral_array[:, 2] + monomer_size # atom4, subatom of second monomer

        return dihedral_array

@dataclass
class TripleBeadPatchUtils:
    '''Utilities for filaments whose monomer is 3 collinear beads + 1 patch on the middle bead.'''
    @staticmethod
    def get_straight_filament(
            n_monomers: int,
            bead_distance: float,
            patch_distance: float,
            start: tuple[float, float, float] = (0., 0., 0.),
            centered: bool = False
        ) -> np.ndarray:
        """Straight filament along x, patches displaced by +patch_distance in y from the middle bead.

        Returns array of shape (4*n_monomers, 3) ordered per monomer as [b0, b1, b2, patch].
        Consecutive backbone beads (also across monomers) are bead_distance apart.
        """
        backbone_x = np.arange(3*n_monomers) * bead_distance
        if centered:
            backbone_x -= backbone_x.mean()

        positions = np.zeros((4*n_monomers, 3), dtype=float)
        for kk in range(n_monomers):
            positions[4*kk:4*kk+3, 0] = backbone_x[3*kk:3*kk+3]
            positions[4*kk+3, 0] = backbone_x[3*kk+1]
            positions[4*kk+3, 1] = patch_distance

        positions += np.asarray(start, dtype=float)
        return positions

class TripleBeadPatchDiffuseUtils:
    '''Utilities for filaments whose monomer is 3 collinear beads + 1 patch on the middle bead + 3 diffuse beads.'''
    @staticmethod
    def get_straight_filament(
            n_monomers: int,
            bead_distance: float,
            patch_distance: float,
            diffuse_distance: float,
            start: tuple[float, float, float] = (0., 0., 0.),
            centered: bool = False
        ) -> np.ndarray:
        """Straight filament along x; patch at +patch_distance in y from the middle bead, one diffuse bead at
        -diffuse_distance in y from every backbone bead.

        Returns array of shape (7*n_monomers, 3) ordered per monomer as [b0, b1, b2, patch, d0, d1, d2].
        """
        backbone_x = np.arange(3*n_monomers) * bead_distance
        if centered:
            backbone_x -= backbone_x.mean()

        positions = np.zeros((7*n_monomers, 3), dtype=float)
        for kk in range(n_monomers):
            positions[7*kk:7*kk+3, 0] = backbone_x[3*kk:3*kk+3]
            positions[7*kk+3, 0] = backbone_x[3*kk+1]
            positions[7*kk+3, 1] = patch_distance
            positions[7*kk+4:7*kk+7, 0] = backbone_x[3*kk:3*kk+3]
            positions[7*kk+4:7*kk+7, 1] = -diffuse_distance

        positions += np.asarray(start, dtype=float)
        return positions

class JinjaCoord:
    index: int
    x: float
    y: float
    z: float