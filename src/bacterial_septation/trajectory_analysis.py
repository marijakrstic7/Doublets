import re
import gzip
from io import StringIO
from pathlib import Path
import json
import itertools as it

import ase
from ase.io import read as ase_read
from scipy.spatial import distance
import numpy as np

import bacterial_septation.parameters as bsp


NATOMS_RE = re.compile(r'ITEM:\sNUMBER OF ATOMS\n([0-9]*)\n')
DATA_RE = re.compile(r'ITEM:\sENTRIES.*([\s\n0-9\.e\-\+]*)')
TRAJ_RE = re.compile(r'ITEM:\sATOMS.*([\s\n0-9\.e\-\+]*)')
TIMESTEP_RE = re.compile(r'ITEM:\sTIMESTEP\n([0-9]*)\n')
BOX_RE = re.compile(r'ITEM:\sBOX.*\n([\s\n0-9\.e\-\+]*)')

def safe_load(gzip_path: Path):
    file_read = gzip.open(gzip_path)

    content = ''
    while True:
        try:
            content += next(file_read).decode('utf-8')
        except EOFError:
            break

    last_match = None
    for match in re.finditer(TIMESTEP_RE, content):
        last_match = match

    if last_match is not None:
        content = content[:last_match.start()]
    else:
        raise ValueError(f'No TIMESTEP found in {gzip_path}')
    
    return content

class TrajectoryParser:
    def __init__(self, path: Path):
        self.path = path
        if self.path.suffix == '.gz':
            try:
                self.file_content = gzip.open(path).read().decode('utf-8')
            except EOFError:
                # If file is corrupted read as far as possible
                print(f"File {path} is corrupted, reading as far as possible.")
                self.file_content = safe_load(path)
        else:
            self.file_content = open(path).read()
        # self.ase_trajectory = ase_read(path, index=':')

    def get_timesteps(self):
        data_matches = re.finditer(TIMESTEP_RE, self.file_content)
        return np.asarray([int(timestep.group(1)) for timestep in data_matches], dtype=int)
    
    def get_box_size(self):
        data_matches = re.finditer(BOX_RE, self.file_content)
        box_size = np.asarray([np.loadtxt(StringIO(box.group(1)), dtype=float) for box in data_matches])
        return box_size
    
    def get_natoms(self, target_types: list[int] | None = None):
        # Case 1: total number of atoms per frame
        if target_types is None:
            data_matches = re.finditer(NATOMS_RE, self.file_content)
            return np.asarray(
                [int(nentries.group(1)) for nentries in data_matches],
                dtype=int
            )

        # Case 2: count atoms of specific types
        target_types = np.atleast_1d(target_types)
        types = self.types_iterator()

        return np.asarray([
            np.count_nonzero(
                np.any(frame_types[:, np.newaxis] == target_types, axis=1)
            )
            for frame_types in types
        ])

            

    def frame_iterator(self, usecols=None):
        data_iterator = re.finditer(TRAJ_RE, self.file_content)

        for data_block in data_iterator:
            yield np.loadtxt(
                StringIO(data_block.group(1)), usecols=usecols
            )
    
    def box_iterator(self):
        data_iterator = re.finditer(BOX_RE, self.file_content)

        for data_block in data_iterator:
            yield np.loadtxt(
                StringIO(data_block.group(1)), dtype=float
            )

    def get_box(self, step: int = 0) -> np.ndarray:
        res = next(it.islice(self.box_iterator(), step, None), None)

        if res is None:
            raise ValueError(f'Step {step} not found in trajectory.')
        
        return res

    def types_iterator(self):
        data_iterator = re.finditer(TRAJ_RE, self.file_content)

        for data_block in data_iterator:
            type_array = np.loadtxt(
                StringIO(data_block.group(1)), usecols=2, dtype=int
            )
            yield type_array

    def get_nfilaments(self, head_type: int = 3):
        data_iterator = re.finditer(TRAJ_RE, self.file_content)

        n_filaments = []
        for data_block in data_iterator:
            type_array = np.loadtxt(
                StringIO(data_block.group(1)), usecols=2, dtype=int
            )
            n_filaments.append(np.count_nonzero(type_array == head_type))
            
        return np.asarray(n_filaments, dtype=int)
            
class BondsParser:
    def __init__(self, path: Path):
        self.path = path
        if self.path.suffix == '.gz':
            try:
                self.file_content = gzip.open(path).read().decode('utf-8')
            except EOFError:
                # If file is corrupted read as far as possible
                self.file_content = safe_load(path)
        else:
            self.file_content = open(path).read()
    
    def bonds_iterator(self, usecols: int = -1):
        data_matches = re.finditer(DATA_RE, self.file_content)

        for datablock in data_matches:
            yield np.loadtxt(
                StringIO(datablock.group(1)), usecols=usecols
            )
    
    def mean_std_alignment(self, index=np.s_[:]):
        bond_steps = np.array([int(match) for match in re.findall(TIMESTEP_RE, self.file_content)])

        # datablock contains dx, dy, dz
        for ii_block, dr_block in enumerate(self.bonds_iterator(usecols=[-3, -2, -1])):
            pass
        
    def get_timesteps(self):
        """Get array of timesteps for each frame."""
        import re
        
        if not hasattr(self, '_timesteps'):
            # Extract timesteps from file content
            self._timesteps = [int(match) for match in re.findall(TIMESTEP_RE, self.file_content)]
        
        return self._timesteps
    
    def get_frame_data(self):
        """Get all frame data including timesteps and bond information."""
        frame_indices = list(self.bonds_iterator(usecols=(0, 1)))
        frame_deltas = list(self.bonds_iterator(usecols=(-3, -2, -1)))
        frame_types = list(self.bonds_iterator(usecols=2))
        timesteps = self.get_timesteps()
        
        return {
            'timesteps': timesteps,
            'indices': frame_indices,
            'deltas': frame_deltas,
            'types': frame_types
        }

    def nematic_order_mortal_filaments(self, bonds_max: int=1000):
        """Calculate the nematic order parameter based on @cvanhille's implementation in mortal filaments (10.1038/s41567-024-02597-8).
        This is quite slow as it requires correlation matrices to be calculated.
        Also scales wrong in 2D, is not between 0 and 1 but 1/4 and 1.

        Args:
            bonds_max (int, optional): Maximum number of bonds to consider. 
                If more than bonds_max bonds are present in the system, 
                pick bonds_max randomly selected bonds.
                Defaults to 1000.

        Returns:
            np.ndarray: Array of nematic order parameters for each frame.
        """

        frame_indices = self.bonds_iterator(usecols=(0, 1))
        frame_deltas = self.bonds_iterator(usecols=(-3, -2, -1))

        s_order_params = [0.]
        for ii_frame, (indices, deltas) in enumerate(zip(frame_indices, frame_deltas)):
            
            if ii_frame == 0: # skip first frame
                continue
            
            deltas = np.atleast_2d(deltas)
            indices = np.atleast_2d(indices)
            unit_vectors = deltas[indices[:, 0]!=1, :].copy()
            unit_vectors /= np.linalg.norm(unit_vectors, axis=-1)[:, np.newaxis]

            if len(unit_vectors) > bonds_max:
                unit_vectors = unit_vectors[np.random.choice(len(unit_vectors), bonds_max, replace=False)]

            pairwise = (-distance.pdist(unit_vectors, metric='cosine'))+1
            s_order_params.append(
                (3 * np.mean(pairwise**2) - 1) / 2
            )
        
        return np.array(s_order_params, dtype=np.float32)

    def nematic_order(self, min_particle_index: int = 2, bond_type: int = 2):
        """Nematic order parameter according to Freud's documentation.
        See https://freud.readthedocs.io/en/latest/gettingstarted/examples/module_intros/order.Nematic.html

        According to @jmajek, in 2D this has an incorrect scaling and is not between 0 and 1 but 1/4 and 1.
        Tested agains freud.order.Nematic and gives same results.
        """
        frame_indices = self.bonds_iterator(usecols=(0, 1))
        frame_deltas = self.bonds_iterator(usecols=(-3, -2, -1))
        frame_types = self.bonds_iterator(usecols=2)

        s_order_params = []
        for ii_frame, (indices, deltas, types) in enumerate(zip(frame_indices, frame_deltas, frame_types)):
            if len(deltas) == 0:
                # frame is empty, no bonds
                s_order_params.append(0.)
                continue
            
            indices = np.atleast_2d(indices)
            where_filament_bonds = np.logical_and(indices[:, 1] > min_particle_index, types == bond_type)
            if not np.any(where_filament_bonds):
                # no filament bonds, skip
                s_order_params.append(0.)
                continue
                
            
            deltas = np.atleast_2d(deltas)
            unit_vectors = deltas[where_filament_bonds, :].copy()
            unit_vectors /= np.linalg.norm(unit_vectors, axis=-1)[:, np.newaxis]
            
            m_matrix = np.einsum('ij,ik->jk', unit_vectors, unit_vectors) / len(unit_vectors)
            q_matrix = m_matrix - np.eye(3) / 3
            eigenvalues, _ = np.linalg.eig(q_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            s_order_params.append(1.5*eigenvalues[0])
        
        return np.array(s_order_params, dtype=np.float32)

    def cylinder_alignment(self, min_particle_index: int = 2, long_axis: int = 1, bond_type: int = 2):
        """ Calculate the cylinder alignment based on bond distances.
        
        Args:
            min_particle_index (int): Minimum particle index to consider for filament bonds.
                This is a temporary solution for filtering out eg. bonds between nucleators or other structures.
            long_axis (int): Axis along which the cylinder is aligned.
                0 for x, 1 for y, 2 for z. Defaults to 1 (y-axis).

        Returns:
            np.ndarray: Array of cylinder alignments for each frame.
        """
        if long_axis not in [0, 1, 2]:
            raise ValueError("long_axis must be 0, 1, or 2.")

        cylinder_alignments = []
        bond_distances = self.bonds_iterator(usecols=[-3, -2, -1])
        bond_partners = self.bonds_iterator(usecols=1)
        bond_types = self.bonds_iterator(usecols=2)

        for frame_bond_distances, frame_partners, frame_types in zip(
            bond_distances, bond_partners, bond_types
        ):
            if len(frame_bond_distances) == 0:
                # frame is empty, no bonds
                cylinder_alignments.append(0.)
                continue

            where_filament_bonds = np.logical_and(frame_partners > min_particle_index, frame_types == bond_type)
            if not np.any(where_filament_bonds):
                # no filament bonds, skip
                cylinder_alignments.append(0.)
                continue
            
            frame_bond_distances = np.atleast_2d(frame_bond_distances)
            filtered_bond_distances = np.atleast_2d(frame_bond_distances[where_filament_bonds].squeeze())
            filtered_bond_distances /= np.linalg.norm(filtered_bond_distances, axis=1)[:, np.newaxis]

            cylinder_alignments.append(np.mean(1 - np.abs(filtered_bond_distances[:, long_axis])))

        return np.array(cylinder_alignments, dtype=np.float32)

class AnglesParser:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.file_content = gzip.open(path).read().decode('utf-8')
        except EOFError:
            # If file is corrupted read as far as possible
            self.file_content = safe_load(path)
    
    def angles_iterator(self, usecols: int = -1):
        data_matches = re.finditer(DATA_RE, self.file_content)

        for datablock in data_matches:
            yield np.loadtxt(
                StringIO(datablock.group(1)), usecols=usecols
            )

    def mean_std(self, equilibrium_angle, index=np.s_[:], angle_type: int=1):
        # timesteps, just find all
        angle_steps = np.array([int(match) for match in re.findall(TIMESTEP_RE, self.file_content)])
        mean_angles = np.zeros(len(angle_steps), dtype=np.float32)
        std_angles = np.zeros(len(angle_steps), dtype=np.float32)

        angle_values = self.angles_iterator(usecols=-1)
        angle_types = self.angles_iterator(usecols=3)

        for ii_block, (datablock, angle_type) in enumerate(zip(angle_values, angle_types)):
            # skip empty blocks
            if not len(datablock):
                continue

            is_type = angle_type == angle_type
            
            datablock = np.atleast_1d(datablock[is_type])
            # skip if no angles of this type
            if not len(datablock):
                continue
            
            datablock[:] = np.abs(
                datablock[:]-equilibrium_angle
            )
            
            mean_angles[ii_block] = np.mean(datablock[index])
            std_angles[ii_block] = np.std(datablock[index])

        return angle_steps, mean_angles, std_angles

class DihedralsParser:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.file_content = gzip.open(path).read().decode('utf-8')
        except EOFError:
            # If file is corrupted read as far as possible
            self.file_content = safe_load(path)
    
    def dihedrals_iterator(self, usecols: int = -2):
        data_matches = re.finditer(DATA_RE, self.file_content)

        for datablock in data_matches:
            yield np.atleast_1d(np.loadtxt(
                StringIO(datablock.group(1)), usecols=usecols
            ))

    def mean_std(self, equilibrium_dihedral=0., center=True, index=np.s_[:], dihedral_type: int=1):
        # timesteps, just find all
        dihedral_steps = np.array([int(match) for match in re.findall(TIMESTEP_RE, self.file_content)])
        mean_dihedrals = np.zeros(len(dihedral_steps), dtype=np.float32)
        std_dihedrals = np.zeros(len(dihedral_steps), dtype=np.float32)

        dihedral_values = self.dihedrals_iterator(usecols=-2)
        dihedral_types = self.dihedrals_iterator(usecols=4)

        for ii_block, datablock in enumerate(self.dihedrals_iterator()):
            if not len(datablock):
                continue

            is_type = dihedral_types == dihedral_type
            datablock = np.atleast_1d(datablock[is_type])
            # skip if no dihedrals of this type
            if not len(datablock):
                continue

            if center:
                datablock[datablock>90] = datablock[datablock>90]-180
            datablock[:] = np.abs(
                datablock[:]-equilibrium_dihedral
            )

            mean_dihedrals[ii_block] = np.mean(datablock[index])
            std_dihedrals[ii_block] = np.std(datablock[index])
        
        return dihedral_steps, mean_dihedrals, std_dihedrals


class ReactionsParser():
    def __init__(
        self,
        path: Path, 
        n_nmer_reacts: int = 3,
        extra_nucleation: bool = False
    ):
        self.path: Path = Path(path).resolve()
        self.data: np.ndarray = np.loadtxt(path).astype(int)
        self.n_cols: int = self.data.shape[1]
        self.n_nmer_reacts: int = n_nmer_reacts
        self.extra_nucleation: bool = extra_nucleation

        self.addition_indices = np.r_[2, 4:4+self.n_nmer_reacts*2:2]
        self.deletion_indices = np.r_[3, 5:5+self.n_nmer_reacts*2:2]
        self.nucleation_indices = np.r_[0, 5+self.n_nmer_reacts*2] if self.extra_nucleation else np.r_[0]
        self.denucleation_indices = np.r_[1]

    @property
    def timesteps(self):
        return self.data[:, 0].copy()

    @property
    def additions(self):
        """Get the addition reactions."""
        return np.sum(np.atleast_2d(self.data[:, self.addition_indices]), axis=1)
    
    @property
    def deletions(self):
        """Get the deletion reactions."""
        return np.sum(np.atleast_2d(self.data[:, self.deletion_indices]), axis=1)
    
    @property
    def nucleations(self):
        """Get the nucleation reactions."""
        return np.sum(np.atleast_2d(self.data[:, self.nucleation_indices]), axis=1)
    
    @property
    def denucleations(self):
        """Get the denucleation reactions."""
        return np.sum(np.atleast_2d(self.data[:, self.denucleation_indices]), axis=1)
    
    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    @staticmethod
    def rate_per_time(n_reactions, reaction_time: float, n_window: int=1):
        if n_window == 1:
            return np.diff(n_reactions) / reaction_time
        return ReactionsParser.moving_average(np.diff(n_reactions), n=n_window) / reaction_time


class Utils:
    @staticmethod
    def load_param_dict(path: Path):
        path = Path(path).resolve()
        run_params: dict[str, str | float | dict[str, float]] = json.loads(path.read_text())

        param_dict = {}
        for key, value in run_params.items():
            if isinstance(value, dict):
                for subkey in list(value.keys()): # copy keys
                    # if the key is already in there, add it as a subkey
                    if subkey in param_dict.keys():
                        value[f'{key}.{subkey}'] = param_dict.pop(subkey, None)
                param_dict.update(value)
            else:
                param_dict[key] = value

        return param_dict