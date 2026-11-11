from dataclasses import dataclass, field
from os import path as osp
from pathlib import Path
import warnings

from numpy import ndarray
from numpy import pi as np_pi

HYDROLYSIS_MODES = ['mean-field', 'explicit']
KAPPA_FROM_MU = {2.5: 12.7, 3: 15.1, 3.5: 17.7}

@dataclass
class SimulationParameters:
    timestep: float # [tau]
    total_time: float # [tau]
    print_time: float # [tau]

    diffusivity: float = field(kw_only=True, default=2) # [sigma^2/tau]

    reaction_time: float = field(kw_only=True, default=None) # [tau]
    nucleation_time: float = field(kw_only=True, default=None) # [tau]

    tau_per_sec: float = field(kw_only=True, default=None) # tau/s
    sigma: float = field(kw_only=True, default=None) # nm/sigma

    seed: int = field(kw_only=True, default=1234)
    comm_range: float = field(kw_only=True, default=5.0) # sigma
    hydrolysis_mode: str = field(kw_only=True, default='mean-field')
    
    max_restart_number: int = field(kw_only=True, default=0) # maximum restart number for constriction or secnuc simulations

    def __post_init__(self):
        if self.nucleation_time is None:
            self.nucleation_time = self.reaction_time

        if self.hydrolysis_mode not in HYDROLYSIS_MODES:
            raise ValueError(f"hydrolysis_mode must be one of {HYDROLYSIS_MODES}, got {self.hydrolysis_mode}.")

    @property
    def reaction_steps(self):
        return int(self.reaction_time / self.timestep)
    
    @property
    def nucleation_steps(self):
        return int(self.nucleation_time / self.timestep)
    
    @property
    def print_steps(self):
        return int(self.print_time / self.timestep)
    
    @property
    def total_steps(self):
        return int(self.total_time / self.timestep)
    
    @property 
    def damp(self):
        return self.diffusivity / 2
        
    def set_diffusivity_nms(self, diffusivity):
        if self.tau_per_sec is None or self.sigma is None:
            raise ValueError('tau_per_sec and sigma must be set to calculate diffusivity from physical units.')
        
        self.diffusivity = diffusivity / (self.tau_per_sec * self.sigma**2)

@dataclass
class TreadmillingParameters:
    r_on: float
    tau_hyd: float
    r_nuc: float
    r_off: float|None = field(kw_only=True, default=None)
    template_path: Path | str | None = field(kw_only=True, default=None)
    are_patches_interacting: bool = field(kw_only=True, default=False)
    
    add_pol_modulation: bool = field(kw_only=True, default=False)
    add_depol_modulation: bool = field(kw_only=True, default=False)
    r_on_modulation: float = field(kw_only=True,default=0.)
    r_off_modulation: float = field(kw_only=True,default=0.)
    tau_hyd_modulation: float = field(kw_only=True,default=0.)
    infinite_growth: bool = field(kw_only=True, default=False)
    infinite_growth_box_size: float | None = field(kw_only=True, default=None)
    add_min_treadmilling_profile: bool = field(kw_only=True, default=False)
    treadmilling_profile_width: float | None = field(kw_only=True, default=None)
    add_alignment_force: float | None = field(kw_only=True, default=None)
    
    add_two_oligomer_templates: bool = field(kw_only=True, default=False)
    reservoir: bool = field(kw_only=True, default=False)
    prob_secnuc: float | None = field(kw_only=True, default=None)
    prob_changetype: float | None = field(kw_only=True, default=None)
    add_sec_nuc_modulation: bool = field(kw_only=True, default=False)
    r_sec_nuc: float | None = field(kw_only=True, default=None)
    
    add_stacking_modulation: bool = field(kw_only=True, default=False)
    prob_stacking: float | None = field(kw_only=True, default=None)
    prob_changetype_stacking: float | None = field(kw_only=True, default=None)
    r_stacking: float | None = field(kw_only=True, default=None)
    
    vary_nucleation: bool = field(kw_only=True, default=False)
    max_atoms: int | None = field(kw_only=True, default=None)
    
    add_fuzzy_interaction: bool = field(kw_only=True, default=False)
    fuzzy_prob: float | None = field(kw_only=True, default=None)
    fuzzy_reaction_step: int | None = field(kw_only=True, default=None)
    enable_templates_for9_and_14: bool = field(kw_only=True, default=False)
    template_path_for9_and_14: Path | str | None = field(kw_only=True, default=None)
    

    def __post_init__(self):
        if self.template_path is None:
            self.template_path = str(
                osp.join(
                    Path(__file__).resolve().parent.parent.parent 
                    / 'lammps_chemistry_templates' / 
                    'nn_treadmilling_templates',
                    ''
                )
            )
        if self.template_path_for9_and_14 is None:
            self.template_path_for9_and_14 = str(
                osp.join(
                    Path(__file__).resolve().parent.parent.parent 
                    / 'lammps_chemistry_templates' / 
                    'nn_treadmilling_templates_for9_and_14',
                    ''
                )
            )
            
        if isinstance(self.template_path_for9_and_14, Path):
            self.template_path_for9_and_14 = osp.join(self.template_path_for9_and_14, '')
        
        if isinstance(self.template_path_for9_and_14, str):
            if not self.template_path_for9_and_14.endswith('/'):
                raise ValueError("template_path_for9_and_14 must end with a '/' character as it must be a directory path.")

        if isinstance(self.template_path, Path):
            self.template_path = osp.join(self.template_path, '')

        if isinstance(self.template_path, str):
            if not self.template_path.endswith('/'):
                raise ValueError("template_path must end with a '/' character as it must be a directory path.")
            


@dataclass
class SurfaceTreadmillingParameters(TreadmillingParameters):
    surf_min: float
    surf_max: float

@dataclass
class ModulatedSurfaceTreadmillingParameters(SurfaceTreadmillingParameters):
    profile_width: float

    def __post_init__(self):
        super().__post_init__()

@dataclass
class SwitchTreadmillingParameters(ModulatedSurfaceTreadmillingParameters):
    switch_time: float

    r_on_post: float | None = field(kw_only=True, default=None)
    tau_hyd_post: float | None = field(kw_only=True, default=None)
    r_nuc_post: float | None = field(kw_only=True, default=None)
    profile_width_post: float | None = field(kw_only=True, default=None)

    def __post_init__(self):
        super().__post_init__()

        if self.r_on_post is None:
            self.r_on_post = self.r_on
        if self.tau_hyd_post is None:
            self.tau_hyd_post = self.tau_hyd
        if self.r_nuc_post is None:
            self.r_nuc_post = self.r_nuc
        if self.profile_width_post is None:
            self.profile_width_post = self.profile_width

@dataclass 
class FilamentParameters:
    '''Parameters of the filaments.

    nth_neigh_bond_distance: bond distance of nth neighbours.
        Can be added additionally for stability over longer distances.
        But does not change runs much and usually not necessary.
    nn_equilibrium_angle: Equilibrium angle between nearest neighbours.
        Needed when defining angles and dihedrals to be over multiple monomers
        to give filament stability.
    '''
    mass: float
    bond_distance: float
    bond_rigidity: float

    equilibrium_angle: float
    angle_rigidity: float

    equilibrium_dihedral: float
    dihedral_rigidity: float

    repulsion_energy: float
    repulsion_distance: float
    attraction_distance: float = field(kw_only=True, default=0.)

    nth_neigh_bond_distance: float = field(kw_only=True, default=0.)
    nth_neigh_bond_rigidity: float = field(kw_only=True, default=0.)
    nn_equilibrium_angle: float = field(kw_only=True, default=0.)
    nn_angle_rigidity: float = field(kw_only=True, default=0.)

@dataclass
class DoubleBADFilamentParameters(FilamentParameters):
    patch_distance: float
    patch_rigidity: float
    patch_equilibrium_angle: float
    patch_angle_rigidity: float
    patch_equilibrium_dihedral: float
    patch_dihedral_rigidity: float

    interpatch_bond_rigidity: float = field(kw_only=True, default=0.)
    interpatch_bond_distance: float = field(kw_only=True, default=0.)

    dihedral_rigidity: float = field(kw_only=True, default=0.)
    equilibrium_dihedral: float = field(kw_only=True, default=0.)

    def __post_init__(self):
        if self.dihedral_rigidity != 0.:
            warnings.warn("It is recommended to set dihedral_rigidity to 0 in DoubleBADFilaments and use patch_dihedral_rigidity instead to control filament geometry.", UserWarning)

        if self.interpatch_bond_rigidity != 0.:
            warnings.warn(f"interpatch_bond_rigidity set to {self.interpatch_bond_rigidity}.Bonds between patches are only used to guide chemistry and not necessary when properly setting patch angles and dihedrals.", UserWarning)
            
@dataclass
class DoubleBADTwoPatchesFilamentParameters(DoubleBADFilamentParameters):

    patch2_distance: float
    patch2_rigidity: float
    patch2_equilibrium_angle: float
    patch2_angle_rigidity: float
    
    interpatch2_bond_rigidity: float = field(kw_only=True, default=0.)
    interpatch2_bond_distance: float = field(kw_only=True, default=0.)

@dataclass
class ESCRTFilamentParameters:
    bond_lengths: list[float]
    bond_rigidities: list[float]

    # also store which types of bonds the rigidities and lengths are saved for
    # if not given assumed 1 ... n_lengths + 1
    mass: float = field(kw_only=True, default=1.)
    bond_types: list[int] | None = field(kw_only=True, default=None)

    def __post_init__(self):
        if isinstance(self.bond_lengths, ndarray):
            self.bond_lengths = self.bond_lengths.tolist()

        if isinstance(self.bond_rigidities, ndarray):
            self.bond_rigidities = self.bond_rigidities.tolist()

        if self.bond_types is None:
            self.bond_types = list(range(1, len(self.bond_lengths) + 1))

@dataclass
class MembraneParameters:
    attraction_epsilon: float

@dataclass
class CylinderMembraneParameters(MembraneParameters):
    cell_radius: float

@dataclass
class CosSqMembraneParameters(CylinderMembraneParameters):
    attraction_min: float
    attraction_max: float
    repulsion_epsilon: float

    sigma: float = field(kw_only=True, default=1.)

    @property
    def equilibrium_distance(self):
        print("Equilibrium distance in CosSqMembraneParameters not properly configured.")
        # TODO: implement equilibrium distance calculation properly
        return self.attraction_min

@dataclass
class CosSqCylinderMembraneParameters(CosSqMembraneParameters):
    @property
    def max_attraction_radius(self):
        return self.cell_radius - self.attraction_min - (self.attraction_max - self.attraction_min)/2
    
    @property
    def equilibrium_radius(self):
        return self.cell_radius - self.equilibrium_distance
    
@dataclass
class LJMembraneParameters(CylinderMembraneParameters):
    equilibrium_distance: float
    cutoff: float = field(kw_only=True, default=None)

    @property
    def sigma(self):
        return self.equilibrium_distance/(2.**(1./6.))
    
    @property
    def equilibrium_radius(self):
        return self.cell_radius - self.equilibrium_distance
    
@dataclass
class YLZParameters:
    type: int
    cutoff: float
    mu: float
    zeta: float
    eps: float
    sigma: float
    
@dataclass
class YLZCylinderParameters(YLZParameters):
    target_radius: float
    height: float
    n_particles: int

    def barostat_from_radius(self, target_radius: float|None = None):
        """Compute barostat value for YLZ membrane simulation, which when using fix nph/asphere x/y/z is a stress tensor component, sigma_xx/yy/zz.
        Based on r_eq = sqrt(kappa / 2 * gamma) -> gamma = kappa / (2 * r_eq^2)
        P_zz = - 2 * a * gamma -> Pzz = - a * kappa / r_eq^2
        Finally, sigma_zz = P_zz * n_particles.
        """
        if target_radius is None:
            target_radius = self.target_radius

        if self.mu not in KAPPA_FROM_MU:
            raise ValueError(f"Unsupported mu value {self.mu}, supported values: {list(KAPPA_FROM_MU.keys())}")
        
        kappa = KAPPA_FROM_MU[self.mu]
        area_per_particle = 4 * np_pi * target_radius * self.height / self.n_particles
        # return r_target ** 2 / (kappa * area_per_particle)
        return - area_per_particle * kappa * self.n_particles / target_radius**2

@dataclass
class SurfaceNucleationParameters:
    surface_group: str
    Nevery: int
    seed: int

    prob: float|None = None
    Rsurf: float|None = None
    Roverlap: float|None = None
    nowarn: bool|None = None
    noffset: int|None = None
    insert_sigma: float|None = None
    lifetime: str|None = None