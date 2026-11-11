# based on a Python script by Adam Prada
# Functions generating ESCRT geometries

from itertools import combinations, product

import numpy as np
import copy

HEAD_TYPES = [1, 2, 3]
BULK_TYPES = [4, 5, 6]
TAIL_TYPES = [7, 8, 9]

class _ESCRTUtils:
    @staticmethod
    def _unit_vector(vector):
        """ Returns the unit vector of the vector. """
        if max(abs(vector))>1e-12:
            return vector / np.linalg.norm(vector)
        else:
            return vector

    @staticmethod
    def _distance_between(v1, v2):
        """Returns the distance between position vectors 'v1' and 'v2' """
        return np.sqrt(np.sum(np.square(np.array(v1)-np.array(v2))))

    @staticmethod
    def mols_and_types(n_triplets: int, use_rigid: bool=False):
        mol_ids = np.zeros((n_triplets*3), dtype=int)

        # give each tuple molecule ids
        for ii in range(3):
            mol_ids[ii::3] = np.arange(1, n_triplets+1)

        # make atom_type array
        if use_rigid:
            # historically based on what Adam did in original script
            # this makes 0 sense to me, that's a lot of atom types
            # - Felix Wodaczek
            atom_types = np.zeros(n_triplets*3)
            atom_types[::3] = 1
            atom_types[1::3] = mol_ids+1
            atom_types[2::3] = mol_ids+1
        else:
            atom_types = np.tile(BULK_TYPES, n_triplets)
            atom_types[:3] = TAIL_TYPES # tail
            atom_types[-3:] = HEAD_TYPES # head

        return mol_ids, atom_types

    @staticmethod
    def get_coordinates(
        N, Nc,
        radius_multiplier: float=1, use_rigid: bool=False,
        position_offset=np.array([0, 0, 0], dtype=float)
    ):
        """Gives atoms of a helical filament consisting of triplets

        Args:
            N  = Number of particles of the helix
            Nc = number of beads at the circumference of the inscribed membrane
            radius_multiplier = factor, determines the quotient between the radii of the membrane tube and the helix
            use_rigid = if True, no additional bonds are created between triplets

        Returns:
            coordinates (np.ndarray[float]): (3 * N x 3) coordinates of atom triplets ordered by triplet
            atom_types (np.ndarray[int]): (3 * N) types of individual atoms
            mol_ids (np.ndarray[int]): (3 * N) molecule id of each atom, one molecule per triplet
        """
        if use_rigid:
            print("WARNING: Using the rigid generator is untested and may not work as expected! -Felix Wodaczek.")

        coordinates = np.zeros((N*3, 3), dtype=float)

        # Start calculation like for the membrane cylinder
        d = 1 # bead diameter
        d_overlap = 0.2*d # overlap of beads
        ##********************
        ## THIS IS TEMPORARY
        #d_overlap = 0
        ##********************
        d_spacing = 1.39*d # spacing of monomers (midpoints of bottom sides)
        # ^ This is deemed 3.2 nm by Lena p42, i.e. 1 sim. unit = 2.3 nm
        d_side = d - d_overlap
        d_height = d_side * np.sqrt(3)/2
        groove_size = 3
        pitch = groove_size/(2*np.pi) # 2*pi*pitch should be groove size

        # Calculating circumference of the membrane
        d_membrane = 1
        alpha_membrane = 2*np.pi/Nc
        R_membrane = (d_membrane/2)/np.sin(alpha_membrane/2)
        # Radius of the helix (midpoints of lower sides)
        # This number is difficult to pick, since it should be diameter of the relaxed membrane
        R = (R_membrane-d)*radius_multiplier
        # factor m determines the quotient between the radii of the membrane tube and the helix
        # Get the absolute difference between the radius of the membrane and the radius of the shrinked helix
        r_shift = (R_membrane-d)-R
        # Helix steepness angle
        tan_theta_hel = pitch/R
        theta_hel = np.arctan(tan_theta_hel)
        # Distance along cylinder axis
        dx = d_spacing*np.sin(theta_hel)
        # Right angle distance along circumference
        dc = d_spacing*np.cos(theta_hel)
        # angle around cylinder axis
        alpha = dc/R
        # Calculating 
        #alpha = 2*np.arcsin(d/(2*R))
        #Nc = 2*np.pi/alpha
        #Nc_max = np.floor(Nc)
        #helix_width = d*N_turns

        # Calculating centres of lower triangles
        N_turns = N*alpha/(2*np.pi)
        helix_width = N_turns*2*np.pi*pitch
        helix_centre = 0
        for i_n in range(N):
            # generate coordinates
            x = i_n*dx + helix_centre - helix_width/2
            angle = i_n*alpha
            y = R*np.sin(angle)
            z = R*np.cos(angle)
            coord = np.array([x,y,z])
            # now, x,y,z refer to the midpoint of the lower edge
            # upper atom
            # effective radius is d_height up from the central atom
            R_u = R-d_height
            coord_u = [
                x,
                R_u*np.sin(angle),
                R_u*np.cos(angle)+r_shift
            ]
            # adding the r_shift to the coords to shift escrt to the membrane surface again

            # Vector from the axis to the midpoint
            vec_rad = _ESCRTUtils._unit_vector(np.array([0,y,z]))
            # Vector tangent to the cylinder
            vec_tan = _ESCRTUtils._unit_vector(np.array([0,z,-y]))
            # Vector along the helix
            vec_tan = _ESCRTUtils._unit_vector(vec_tan + np.array([tan_theta_hel,0,0]))
            # Vector perpendicular, i.e. along the bottom edge of the triangle
            # in the direction of the positive x axis
            vec_triangle_base = _ESCRTUtils._unit_vector(np.cross(vec_tan,vec_rad))

            # lower left, looking from back along the filament
            coord_ll = coord + vec_triangle_base*d_side/2
            # adding the scaled vector components to the coords to shift escrt to the membrane surface again
            coord_ll[2] = coord_ll[2] + r_shift
            # lower right, looking from back along the filament
            coord_lr = coord - vec_triangle_base*d_side/2
            # adding the scaled vector components to the coords to shift escrt to the membrane surface again
            coord_lr[2] = coord_lr[2] + r_shift

            # Add optioinal offset to the coordinates
            coord_u = np.array(coord_u) + position_offset
            coord_ll = np.array(coord_ll) + position_offset
            coord_lr = np.array(coord_lr) + position_offset
            
            coordinates[i_n*3] = coord_u
            coordinates[i_n*3+1] = coord_ll
            coordinates[i_n*3+2] = coord_lr
        
        return coordinates

    @staticmethod
    def get_bonds(N: int, use_rigid: bool = False):
        """Get bonds for N triplets that connect all atoms triplet by triplet and within triplets if use_rigid

        Args:
            N (int): Number of triplets.
            use_rigid (bool, optional): Whether the system uses rigid bonds. If True, no intra-triplet bonds. Defaults to False.
        
        Returns:
            bond_list (list[int, int, int]): (3*N x 3) [bond_type, left_atom, right_atom]
        """
        bond_list: list[tuple[int, int, int]] = [] # TODO: figure out how many of those are needed
        # TODO: would be good if this accepted at least a starting bond type
        
        for i_n in range(N):
        # this used to be one bond type per bond, now one bond type per different bond
            bond_type = 1
            if not use_rigid:
            # Adding bonds within triplets since rigid cannot be used
                for left_atom, right_atom in combinations([0,1,2], 2): # 0-1, 0-2, 1-2
                    bond_list.append((bond_type, i_n*3+left_atom+1, i_n*3+right_atom+1))
                    bond_type += 1

            # Now adding bonds between triplets
            if i_n < (N - 1): # do for all but last triplet, since this doesn't have any more connections
                for ii_left_triplet, ii_right_triplet in product([0, 1, 2], [3, 4, 5]): # all combinations between triplets
                    bond_list.append((bond_type, i_n*3+ii_left_triplet+1, i_n*3+ii_right_triplet+1))
                    bond_type += 1

        return bond_list
    
    @staticmethod
    def get_bonds_double_patch(N: int):
        """Get bonds for N triplets that connect all atoms triplet by triplet and within triplets if use_rigid

        Args:
            N (int): Number of triplets.
        
        Returns:
            bond_list (list[int, int, int]): (3*N x 3) [bond_type, left_atom, right_atom]
            
        Assumes a topology of the double patch system - bonds between fil atom and patches is type 1, bond between patches in same fil atom type 2, bonds between iff fil atoms type 3 and bonds between patches of diff fil atom type 4
        """
        bond_list: list[tuple[int, int, int]] = [] # TODO: figure out how many of those are needed

        # --- Intra-triplet bonds ---
        for t in range(N):
            base = 3 * t
            a, b, c = base, base + 1, base + 2

            # Convert to 1-based indexing
            a1, b1, c1 = a+1, b+1, c+1

            # Bonds inside each triplet
            bond_list.append((1, a1, b1))  # 1-2
            bond_list.append((1, a1, c1))  # 1-3
            bond_list.append((2, b1, c1))  # 2-3

        # --- Inter-triplet bonds ---
        for t in range(N - 1):
            baseA = 3 * t
            baseB = 3 * (t + 1)

            a0 = baseA + 1      # +1 here too
            a2 = baseA + 3
            b0 = baseB + 1
            b1 = baseB + 2

            # type 3 bond between first atoms of each triplet
            bond_list.append((3, a0, b0))

            # type 4 bond between last of A (a2) and middle of B (b1)
            bond_list.append((4, a2, b1))

        return bond_list

    @staticmethod
    def get_bond_lengths(coordinates: np.ndarray, use_rigid=False):
        """Calculate the equilibrium lengths of triplet bonds based on 'coordinates'.

        Args:
            coordinates (np.ndarray): At least two triplets of an ESCRT filament.
            use_rigid (bool, optional): Whether the system uses rigid bonds. If True, no intra-triplet bonds. Defaults to False.

        Raises:
            ValueError: If coordinates is too short

        Returns:
            bond_lengths (list[float]): (n_bond_types) The equilibrium lengths of the bonds.
        """
        if len(coordinates) < 6:
            raise ValueError("Not enough coordinates given in get_bond_lengths, must be at least 2 triplets.")
        
        bond_lengths: list[float] = []

        if not use_rigid: # first add distance of bond interconnections
            for left_atom, right_atom in combinations([0,1,2], 2): # 0-1, 0-2, 1-2
                bond_lengths.append(
                    _ESCRTUtils._distance_between(
                        coordinates[left_atom], coordinates[right_atom]
                    )
                )

        for ii_left_triplet, ii_right_triplet in product([0, 1, 2], [3, 4, 5]):
            bond_lengths.append(
                _ESCRTUtils._distance_between(
                    coordinates[ii_left_triplet], coordinates[ii_right_triplet]
                )
            )

        return bond_lengths