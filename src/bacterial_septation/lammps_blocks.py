from dataclasses import dataclass, field
from pathlib import Path
import warnings

import jinja2

from bacterial_septation import parameters as bsp

BLOCKS_DIR = Path(__file__).parent / 'lammps_blocks'
LIFETIME_ARGS = ['no', 'yes', 'hydrolysis']
VARIABLE_PROFILE_TYPES = ['uniform', 'gaussian']
NUCTYPES = ['plane', 'cylinder', 'double_cylinder_mod']

class BaseBlock:
    def __init__(self, template_path: Path):
        self.template_path: Path = Path(template_path).resolve()
        self.template = jinja2.Template(
            self.template_path.read_text()
        )
        
    def write(self):
        """Pass own namespace to template and render it.

        Returns:
            str: filled template string
        """
        return self.template.render(
            **self.__dict__
        )

class SystemDumpBlock(BaseBlock):
    def __init__(
        self,
        simulation_parameters: bsp.SimulationParameters,
        filename: str = None,
        compress: bool = True,
        extra_thermo_list: list[str] | None = None,
        extra_dump_list: list[str] | None = None,
        dump_ellipsoids: bool = False
    ):
        super().__init__(BLOCKS_DIR / 'system_dump_block.lmp')
        
        self.simulation_parameters: bsp.SimulationParameters = simulation_parameters
        if filename is None:
            filename = 'trajectory'
        self.filename: str = filename
        self.fsuffix = '.gz' if compress else '.txt'
        self.dump_ellipsoids: bool = dump_ellipsoids

        self.dump_type = 'custom'
        if compress:
            self.dump_type += '/gz'

        self.extra_thermo: str = ''
        if extra_thermo_list is not None:
            self.extra_thermo = ' '.join(extra_thermo_list)
        
        self.extra_dump: str = ''
        if extra_dump_list is not None:
            self.extra_dump = ' '.join(extra_dump_list)

        if self.dump_ellipsoids:
            self.extra_dump = ' '.join(['c_cEllipsoid[*]', self.extra_dump])

class BADFilamentDumpBlock(BaseBlock):
    def __init__(self, simulation_parameters: bsp.SimulationParameters, compress=True, extra_outputs=None, write_bonds: bool=True, write_angles: bool=True, write_dihedrals: bool=True):
        """Dump block for BAD filament.

        Args:
            simulation_parameters (bsp.SimulationParameters): Simulation parameters.
        """
        super().__init__(BLOCKS_DIR / 'bad_filament_dump_block.lmp')

        self.simulation_parameters: bsp.SimulationParameters = simulation_parameters
        self.fsuffix = '.gz' if compress else '.txt'
        
        self.write_bonds = write_bonds
        self.write_angles = write_angles
        self.write_dihedrals = write_dihedrals

        self.dump_type: str = 'local'
        if compress:
            self.dump_type += r'/gz'
            
class BADFilamentDumpBlockConstriction(BaseBlock):
    def __init__(self, simulation_parameters: bsp.SimulationParameters, compress=True, extra_outputs=None, write_bonds: bool=True, write_angles: bool=True, write_dihedrals: bool=True, filenameextra: str='_constriction'):
        """Dump block for BAD filament.

        Args:
            simulation_parameters (bsp.SimulationParameters): Simulation parameters.
        """
        super().__init__(BLOCKS_DIR / 'bad_filament_dump_block_constriction.lmp')

        self.simulation_parameters: bsp.SimulationParameters = simulation_parameters
        self.fsuffix = '.gz' if compress else '.txt'
        
        self.write_bonds = write_bonds
        self.write_angles = write_angles
        self.write_dihedrals = write_dihedrals
        self.filenameextra = filenameextra

        self.dump_type: str = 'local'
        if compress:
            self.dump_type += r'/gz'

class BADFilamentBlock(BaseBlock):
    def __init__(self, filament_parameters: bsp.FilamentParameters):
        """Bonds, Angles, Dihedrals based 3D chiral filament.

        Args:
            simulation_parameters (bsp.SimulationParameters): _description_
        """
        super().__init__(BLOCKS_DIR / 'bad_filament_block.lmp')

        self.filament_parameters: bsp.FilamentParameters = filament_parameters

class DoubleBADFilamentBlock(BaseBlock):
    def __init__(self, filament_parameters: bsp.DoubleBADFilamentParameters):
        """A double BAD filament consisting of a BAD filament with patches.

        Args:
            filament_parameters (bsp.DoubleBADFilamentParameters): Rigidities and lengths of bonds, angles and dihedrals within double BAD filament.
        """
        super().__init__(BLOCKS_DIR / 'double_bad_filament_block.lmp')

        if not isinstance(filament_parameters, bsp.DoubleBADFilamentParameters):
            raise ValueError(
                'filament_parameters must be of type DoubleBADFilamentParameters or child'
            )

        self.filament_parameters: bsp.DoubleBADFilamentParameters = filament_parameters
        
class DoubleBADDoubleFilamentBlock(BaseBlock):
    def __init__(self, filament_parameters: bsp.DoubleBADFilamentParameters):
        """A double BAD filament consisting of a BAD filament with patches.

        Args:
            filament_parameters (bsp.DoubleBADFilamentParameters): Rigidities and lengths of bonds, angles and dihedrals within double BAD filament.
        """
        super().__init__(BLOCKS_DIR / 'double_filament2d_bad_filament_block.lmp')

        if not isinstance(filament_parameters, bsp.DoubleBADFilamentParameters):
            raise ValueError(
                'filament_parameters must be of type DoubleBADFilamentParameters or child'
            )

        self.filament_parameters: bsp.DoubleBADFilamentParameters = filament_parameters
        
class DoubleBADDoubleFilamentTwoPatchesBlock(BaseBlock):
    def __init__(self, filament_parameters: bsp.DoubleBADFilamentParameters):
        """A double BAD filament consisting of a BAD filament with patches.

        Args:
            filament_parameters (bsp.DoubleBADFilamentParameters): Rigidities and lengths of bonds, angles and dihedrals within double BAD filament.
        """
        super().__init__(BLOCKS_DIR / 'double_filament2d_2patches_bad_filament_block.lmp')

        if not isinstance(filament_parameters, bsp.DoubleBADFilamentParameters):
            raise ValueError(
                'filament_parameters must be of type DoubleBADFilamentParameters or child'
            )

        self.filament_parameters: bsp.DoubleBADFilamentParameters = filament_parameters

class ESCRTFilamentBlock(BaseBlock):
    def __init__(self, filament_parameters: bsp.ESCRTFilamentParameters):
        """An ESCRT filament consisting of triplets that are inter- and maybe intra-connectively bonded.

        Args:
            filament_parameters (bsp.ESCRTFilamentParameters): Rigidities and lengths of bonds within ESCRT filament.
        """
        super().__init__(BLOCKS_DIR / 'escrt_filament_block.lmp')

        self.filament_parameters: bsp.ESCRTFilamentParameters = filament_parameters

class TreadmillingVariableModulatedBlock(BaseBlock):
    def __init__(
            self,
            simulation_parameters: bsp.SimulationParameters,
            treadmilling_parameters: bsp.ModulatedSurfaceTreadmillingParameters,
            initial_profile: str = 'uniform'
        ):
        """Variable block for treadmilling.

        Args:
            simulation_parameters (bsp.SimulationParameters): Simulation parameters.
            treadmilling_parameters (bsp.TreadmillingParameters): Treadmilling parameters.
            initial_profile (str, optional): Initial profile for the treadmilling. Defaults to 'uniform'.
        """
        super().__init__(BLOCKS_DIR / 'treadmilling_variable_modulated_block.lmp')

        self.simulation_parameters: bsp.SimulationParameters = simulation_parameters

        if not isinstance(treadmilling_parameters, bsp.ModulatedSurfaceTreadmillingParameters):
            raise ValueError(
                'treadmilling_parameters must be of type ModulatedSurfaceTreadmillingParameters or child'
            )

        if initial_profile not in VARIABLE_PROFILE_TYPES:
            raise ValueError(
                f'initial_profile must be one of {VARIABLE_PROFILE_TYPES}, got {initial_profile}'
            )

        self.treadmilling_parameters: bsp.ModulatedSurfaceTreadmillingParameters = treadmilling_parameters
        self.initial_profile: str = initial_profile

class _TreadmillingDumpBlock(BaseBlock):
    def __init__(
            self, 
            simulation_parameters: bsp.SimulationParameters, 
            max_nmer: int = 0,
            modulation: bool = False
        ):
        super().__init__(BLOCKS_DIR / 'treadmilling_dump_block.lmp')

        self.simulation_parameters: bsp.SimulationParameters = simulation_parameters
        self.max_nmer: int = max_nmer
        self.modulation: bool = modulation

class SurfaceNucleationBlock(BaseBlock):
    def __init__(
        self,
        surface_nucleation_parameters: bsp.SurfaceNucleationParameters|None = None,
        *args, **kwargs
    ):
        if surface_nucleation_parameters is not None:
            self.surface_nucleation_parameters: bsp.SurfaceNucleationParameters = surface_nucleation_parameters
        else:
            self.surface_nucleation_parameters = bsp.SurfaceNucleationParameters(*args, **kwargs)

        super().__init__(BLOCKS_DIR / 'surface_nucleation_block.lmp')

class TreadmillingBlock(BaseBlock):
    def __init__(
            self, 
            simulation_parameters: bsp.SimulationParameters, 
            treadmilling_parameters: bsp.TreadmillingParameters,
            modulation: bool = False,
            nuctype: str = 'plane',
            membrane_parameters: bsp.CylinderMembraneParameters | None = None,
            max_nmer: int = 0,
            lifetime_arg: str = 'hydrolysis',
            include_dump: bool = True,
            surface_treadmilling: bool = False,
            stabilize_steps: int = 60,
            nooverlap: list[int] | None = None
        ):
        """
        fix bond/react blocks containing everything necessary for treadmilling.
        """
        super().__init__(BLOCKS_DIR / 'treadmilling_block.lmp')

        self.simulation_parameters: bsp.SimulationParameters = simulation_parameters
        self.treadmilling_parameters: bsp.TreadmillingParameters = treadmilling_parameters
        self.modulation: bool = modulation
        self.nuctype: str = nuctype
        self.membrane_parameters: bsp.CylinderMembraneParameters | None = membrane_parameters
        self.max_nmer: int = max_nmer
        self.lifetime_arg: str = lifetime_arg
        self.stabilize_steps: int = stabilize_steps
        self.nooverlap: list[int] | None = nooverlap

        # default values for when growth and shrinkage are allowed
        self.xmin_on, self.xmin_off = 0, 0
        self.xmax_on, self.xmax_off = 1.5, 1.5

        if self.nuctype not in NUCTYPES:
            # TODO: maybe split up nuctypes into different classes?
            raise ValueError(
                f'nuctype must be one of {NUCTYPES}, got {nuctype}'
            )
        
        if self.nuctype == 'cylinder' and self.membrane_parameters is None:
            raise ValueError(
                'membrane_parameters of type CylinderMembraneParameters or child must be provided when nuctype is "cylinder"'
            )
        
        if self.nuctype == 'double_cylinder_mod':
            if self.membrane_parameters is None:
                raise ValueError(
                    'membrane_parameters of type CylinderMembraneParameters or child must be provided when nuctype is "double_cylinder_mod"'
                )
            if not isinstance(treadmilling_parameters, bsp.SwitchTreadmillingParameters):
                raise ValueError(
                    'treadmilling_parameters must be of type SwitchTreadmillingParameters or child when nuctype is "double_cylinder_mod"'
                )
            if not modulation:
                raise ValueError(
                    'modulation must be True when nuctype is "double_cylinder_mod"'
                )

        if self.modulation:
            try:
                profile_width = self.treadmilling_parameters.profile_width
            except AttributeError:
                raise ValueError(
                    'treadmilling_parameters must have profile_width attribute when modulation is True'
                )

        self._dump_block: _TreadmillingDumpBlock | None = None
        if include_dump:
            self._dump_block = _TreadmillingDumpBlock(
                simulation_parameters=self.simulation_parameters,
                max_nmer=self.max_nmer,
                modulation=self.modulation
            )

        if not lifetime_arg in LIFETIME_ARGS:
            raise ValueError(
                f'lifetime_arg must be one of {LIFETIME_ARGS}, got {lifetime_arg}'
            )
        
        # check if hydrolysis mode and lifetime arg are consistent
        if self.simulation_parameters.hydrolysis_mode == 'explicit' and lifetime_arg != 'hydrolysis':
            warnings.warn(
                "Using explicit hydrolysis in simulation_parameters "
                "automatically sets 'fix bond/react ... lifetime hydrolysis'. "
                "Set lifetime_arg='hydrolysis' to avoid this warning.",
                UserWarning
            )
            self.lifetime_arg = 'hydrolysis'
        
        # check if hydrolysis mode and r_off are consistent
        if self.simulation_parameters.hydrolysis_mode == 'explicit' and self.treadmilling_parameters.r_off is None:
            raise ValueError(
                "Using explicit hydrolysis in simulation_parameters "
                "requires r_off to be set in treadmilling_parameters."
            )
        
        if self.simulation_parameters.hydrolysis_mode == 'mean-field' and self.treadmilling_parameters.r_off is not None:
            warnings.warn(
                "Using mean-field hydrolysis in simulation_parameters "
                "ignores r_off in treadmilling_parameters. "
                "Set r_off=None to avoid this warning.",
                UserWarning
            )
        
        # post process lifetime arg to contain seed for hydrolysis
        if self.lifetime_arg == 'hydrolysis':
            self.lifetime_arg += f' {self.simulation_parameters.seed}'

        if surface_treadmilling:
            if not isinstance(treadmilling_parameters, bsp.SurfaceTreadmillingParameters):
                raise ValueError(
                    'treadmilling_parameters must be of type SurfaceTreadmillingParameters or child when surface_treadmilling is True'
                )
            self.xmin_on, self.xmax_on = treadmilling_parameters.surf_min, treadmilling_parameters.surf_max

    def dump_block(self) -> str:
        """Return the dump block as a string."""
        if self._dump_block is None:
            raise ValueError(
                'Dump block is not initialized. Set include_dump=True when creating TreadmillingBlock.'
            )
        return self._dump_block.write()
    
class TripleBeadTreadmillingBlock(TreadmillingBlock):
    GROWTH_OVERLAP_STRING = 'modify_create overlap 0.9 fit 2'

    def __init__(
            self,
            *args,
            growth_overlap: float = 0.25,
            growth_overlap_types: dict[int, float] | None = None,
            growth_skip_template: bool = False,
            reservoir_max_monomers: int | None = None,
            nucleation_overlap: float | None = None,
            nucleation_overlap_types: dict[int, float] | None = None,
            yband_denominator: float = 100.,
            **kwargs
        ):
        """TreadmillingBlock (treadmilling_block.lmp) with two additions for TripleBeadPatchFilaments.

        1. A different overlap check for the growth (rOn / rOn<n>mer) reactions only.
           The bond/react overlap check compares every created atom with ALL existing atoms, including the
           head of the filament it attaches to. For monomers whose beads are closer than 0.9 (e.g. r_zero/3)
           the default 'overlap 0.9' would abort every growth step. Nucleation keeps the default 0.9.

        2. r_off modulation for explicit hydrolysis (hydrolysis_mode='explicit' and add_depol_modulation=True),
           as in treadmilling_block_volume_exclusion.lmp:
               koff = ((r_off_modulation - r_off)*reaction_time*c_cPatchNeighbors + r_off*reaction_time)
           The off templates evaluate rxnsum(v_koff, 2), fragment 2 being the patch of the second monomer from
           the shrinking end; c_cPatchNeighbors (compute coord/atom cutoff 0.4 6 in template.lmp) counts its
           type-6 neighbours. (With mean-field hydrolysis treadmilling_block.lmp already modulates tau_hyd instead.)

        Args:
            growth_overlap: general overlap cutoff for growth reactions, must be smaller than the distance
                between the new bead and the current head bead.
            growth_overlap_types: optional {existing_atom_type: cutoff} using the 'overlap R types N t1 R1 ...'
                syntax of saricgrp_lammps fix bond/react. A typed cutoff only adds to the general one,
                e.g. {1: 0.9, 2: 0.9} keeps 0.9 against bulk/tail beads while heads (type 3) use growth_overlap.
            growth_skip_template: adds 'modify_create overlap_skip_template yes' (needs the patched
                saricgrp_lammps fix bond/react): created atoms are not checked against the atoms of the reacting
                template (the filament end they attach to), only against everything else. Then growth_overlap can
                be 0.9 for all types, which also stops monomers being created on top of other filaments' heads.
            reservoir_max_monomers: monomer reservoir. If set, kon is multiplied by 0 as soon as the system holds
                >= reservoir_max_monomers monomers (backbone beads of types 1-3 divided by 3), else by 1.
                Only growth is switched off; nucleation and shrinking continue.
            nucleation_overlap, nucleation_overlap_types: if set, replace the nucleation overlap check
                ('overlap 0.9') by 'overlap nucleation_overlap [types ...]', same syntax as growth_overlap(_types).
            yband_denominator: y-band profile exp(-y*y/yband_denominator) used in kon (treadmilling_block.lmp has 100).
        """
        super().__init__(*args, **kwargs)
        self.growth_overlap: float = growth_overlap
        self.growth_overlap_types: dict[int, float] = growth_overlap_types or {}
        self.growth_skip_template: bool = growth_skip_template
        self.reservoir_max_monomers: int | None = reservoir_max_monomers
        self.nucleation_overlap: float | None = nucleation_overlap
        self.nucleation_overlap_types: dict[int, float] = nucleation_overlap_types or {}
        self.yband_denominator: float = yband_denominator

    def _modulate_explicit_koff(self, rendered: str) -> str:
        tp = self.treadmilling_parameters
        rt = self.simulation_parameters.reaction_time
        constant_koff = f'variable koff atom "{tp.r_off}*{rt}"'
        if constant_koff not in rendered:
            raise RuntimeError(f"Could not find '{constant_koff}' in treadmilling_block.lmp; koff can not be modulated.")

        modulated_koff = (
            f'variable koff atom (({tp.r_off_modulation}-{tp.r_off})*{rt}*c_cPatchNeighbors+{tp.r_off}*{rt})'
            ' # r_off modulated by type-6 neighbours of the patch'
        )
        return rendered.replace(constant_koff, modulated_koff)

    def _apply_reservoir(self, rendered: str) -> str:
        # exactly one active (non-comment) kon definition is expected in treadmilling_block.lmp
        kon_lines = [ii for ii, line in enumerate(rendered.split('\n')) if line.startswith('variable kon atom ')]
        if len(kon_lines) != 1:
            raise RuntimeError(f'Expected exactly one active "variable kon atom" line, found {len(kon_lines)}.')

        lines = rendered.split('\n')
        kon_expr = lines[kon_lines[0]][len('variable kon atom '):]
        lines[kon_lines[0]] = '\n'.join([
            f'## monomer reservoir: kon = 0 once the system holds >= {self.reservoir_max_monomers} monomers',
            'compute cReservoirCount all count/type atom',
            'variable vNMonomers equal (c_cReservoirCount[1]+c_cReservoirCount[2]+c_cReservoirCount[3])/3',
            f'variable vReservoirOpen equal "v_vNMonomers < {self.reservoir_max_monomers}"',
            f'variable konNoReservoir atom {kon_expr}',
            'variable kon atom v_konNoReservoir*v_vReservoirOpen',
        ])
        return '\n'.join(lines)

    def write(self):
        rendered = super().write()

        if self.simulation_parameters.hydrolysis_mode == 'explicit' and self.treadmilling_parameters.add_depol_modulation:
            rendered = self._modulate_explicit_koff(rendered)

        if self.reservoir_max_monomers is not None:
            rendered = self._apply_reservoir(rendered)

        if self.yband_denominator != 100. and self.treadmilling_parameters.add_min_treadmilling_profile:
            if 'exp(-y*y/100)' not in rendered:
                raise RuntimeError("Could not find 'exp(-y*y/100)' in treadmilling_block.lmp; the y-band can not be rescaled.")
            rendered = rendered.replace('exp(-y*y/100)', f'exp(-y*y/{self.yband_denominator:.6g})')

        if self.nucleation_overlap is not None:
            nucleation_string = 'fit all modify_create overlap 0.9 &'
            if nucleation_string not in rendered:
                raise RuntimeError(f"Could not find '{nucleation_string}' in treadmilling_block.lmp.")
            nuc_types = ''
            if self.nucleation_overlap_types:
                pairs = ' '.join(f'{t} {c}' for t, c in self.nucleation_overlap_types.items())
                nuc_types = f' types {len(self.nucleation_overlap_types)} {pairs}'
            rendered = rendered.replace(nucleation_string, f'fit all modify_create overlap {self.nucleation_overlap}{nuc_types} &')

        if self.GROWTH_OVERLAP_STRING not in rendered:
            raise RuntimeError(
                f"Could not find '{self.GROWTH_OVERLAP_STRING}' in treadmilling_block.lmp; "
                "the growth overlap can not be replaced."
            )

        types_string = ''
        if self.growth_overlap_types:
            pairs = ' '.join(f'{atom_type} {cutoff}' for atom_type, cutoff in self.growth_overlap_types.items())
            types_string = f' types {len(self.growth_overlap_types)} {pairs}'

        return rendered.replace(
            self.GROWTH_OVERLAP_STRING,
            f'modify_create overlap {self.growth_overlap}{types_string}'
            f'{" overlap_skip_template yes" if self.growth_skip_template else ""} fit 2'
        )

class TreadmillingBlockVolumeExclusion(BaseBlock):
    def __init__(
            self, 
            simulation_parameters: bsp.SimulationParameters, 
            treadmilling_parameters: bsp.TreadmillingParameters,
            modulation: bool = False,
            nuctype: str = 'plane',
            membrane_parameters: bsp.CylinderMembraneParameters | None = None,
            max_nmer: int = 0,
            lifetime_arg: str = 'hydrolysis',
            include_dump: bool = True,
            surface_treadmilling: bool = False,
            stabilize_steps: int = 60,
            nooverlap: list[int] | None = None,
            nooverlap_atoms_secnuc: list[int] | None = None,
            overlap_cutoffs: list[float] | None = None
        ):
        """
        fix bond/react blocks containing everything necessary for treadmilling.
        """
        super().__init__(BLOCKS_DIR / 'treadmilling_block_volume_exclusion.lmp')

        self.simulation_parameters: bsp.SimulationParameters = simulation_parameters
        self.treadmilling_parameters: bsp.TreadmillingParameters = treadmilling_parameters
        self.modulation: bool = modulation
        self.nuctype: str = nuctype
        self.membrane_parameters: bsp.CylinderMembraneParameters | None = membrane_parameters
        self.max_nmer: int = max_nmer
        self.lifetime_arg: str = lifetime_arg
        self.stabilize_steps: int = stabilize_steps
        self.nooverlap: list[int] | None = nooverlap
        self.nooverlap_atoms_secnuc: list[int] | None = nooverlap_atoms_secnuc
        self.overlap_cutoffs: list[float] | None = overlap_cutoffs

        # default values for when growth and shrinkage are allowed
        self.xmin_on, self.xmin_off = 0, 0
        self.xmax_on, self.xmax_off = 1.5, 1.5

        if self.nuctype not in NUCTYPES:
            # TODO: maybe split up nuctypes into different classes?
            raise ValueError(
                f'nuctype must be one of {NUCTYPES}, got {nuctype}'
            )
        
        if self.nuctype == 'cylinder' and self.membrane_parameters is None:
            raise ValueError(
                'membrane_parameters of type CylinderMembraneParameters or child must be provided when nuctype is "cylinder"'
            )
        
        if self.nuctype == 'double_cylinder_mod':
            if self.membrane_parameters is None:
                raise ValueError(
                    'membrane_parameters of type CylinderMembraneParameters or child must be provided when nuctype is "double_cylinder_mod"'
                )
            if not isinstance(treadmilling_parameters, bsp.SwitchTreadmillingParameters):
                raise ValueError(
                    'treadmilling_parameters must be of type SwitchTreadmillingParameters or child when nuctype is "double_cylinder_mod"'
                )
            if not modulation:
                raise ValueError(
                    'modulation must be True when nuctype is "double_cylinder_mod"'
                )

        if self.modulation:
            try:
                profile_width = self.treadmilling_parameters.profile_width
            except AttributeError:
                raise ValueError(
                    'treadmilling_parameters must have profile_width attribute when modulation is True'
                )

        self._dump_block: _TreadmillingDumpBlock | None = None
        if include_dump:
            self._dump_block = _TreadmillingDumpBlock(
                simulation_parameters=self.simulation_parameters,
                max_nmer=self.max_nmer,
                modulation=self.modulation
            )

        if not lifetime_arg in LIFETIME_ARGS:
            raise ValueError(
                f'lifetime_arg must be one of {LIFETIME_ARGS}, got {lifetime_arg}'
            )
        
        # check if hydrolysis mode and lifetime arg are consistent
        if self.simulation_parameters.hydrolysis_mode == 'explicit' and lifetime_arg != 'hydrolysis':
            warnings.warn(
                "Using explicit hydrolysis in simulation_parameters "
                "automatically sets 'fix bond/react ... lifetime hydrolysis'. "
                "Set lifetime_arg='hydrolysis' to avoid this warning.",
                UserWarning
            )
            self.lifetime_arg = 'hydrolysis'
        
        # check if hydrolysis mode and r_off are consistent
        if self.simulation_parameters.hydrolysis_mode == 'explicit' and self.treadmilling_parameters.r_off is None:
            raise ValueError(
                "Using explicit hydrolysis in simulation_parameters "
                "requires r_off to be set in treadmilling_parameters."
            )
        
        if self.simulation_parameters.hydrolysis_mode == 'mean-field' and self.treadmilling_parameters.r_off is not None:
            warnings.warn(
                "Using mean-field hydrolysis in simulation_parameters "
                "ignores r_off in treadmilling_parameters. "
                "Set r_off=None to avoid this warning.",
                UserWarning
            )
        
        # post process lifetime arg to contain seed for hydrolysis
        if self.lifetime_arg == 'hydrolysis':
            self.lifetime_arg += f' {self.simulation_parameters.seed}'

        if surface_treadmilling:
            if not isinstance(treadmilling_parameters, bsp.SurfaceTreadmillingParameters):
                raise ValueError(
                    'treadmilling_parameters must be of type SurfaceTreadmillingParameters or child when surface_treadmilling is True'
                )
            self.xmin_on, self.xmax_on = treadmilling_parameters.surf_min, treadmilling_parameters.surf_max

    def dump_block(self) -> str:
        """Return the dump block as a string."""
        if self._dump_block is None:
            raise ValueError(
                'Dump block is not initialized. Set include_dump=True when creating TreadmillingBlock.'
            )
        return self._dump_block.write()
    
    
class CylinderTetherBlock(BaseBlock):
    def __init__(
        self,
        cylinder_membrane_parameters: bsp.CylinderMembraneParameters,
        target_group: str = "all",
        dynamic_radius: bool = False
    ):
        """Cylinder tether block for applying surface attraction forces.

        Args:
            membrane_parameters (bsp.CylinderMembraneParameters): Membrane parameters containing
                cell_radius and attraction_epsilon.
            target_group (str, optional): The group of atoms to apply the tether force to. 
                Defaults to "all".
            dynamic_radius (bool, optional): whether to pick a static radius from 
                CylinderMembraneParameters or to always attract to the mean radius of the target group.
                Defaults to False.
        """
        super().__init__(BLOCKS_DIR / 'cylinder_tether.lmp')
        
        self.cylinder_membrane_parameters: bsp.CylinderMembraneParameters = cylinder_membrane_parameters
        self.target_group: str = target_group
        self.dynamic_radius: str = dynamic_radius

class RightingForceBlock(BaseBlock):
    def __init__(
        self,
        epsilon: float,
        delta_r: float,
        r_zero: float,
        target_group: str,
        define_variables: bool = False,
    ):
        """Surface righting force that attracts a group of particles inwards with strength epsilon at r_zero and 0 at r_zero - delta_r.

        Args:
            epsilon (float): Strength of the righting force.
            delta_r (float): Distance over which the force is applied.
            r_zero (float): Distance from 0 at which the force is maximum.
            target_group (str): The group of atoms to apply the righting force to, usually lower end of dimer beads.
            define_variables (bool, optional): Redefine the distance variables, not necessary if run with a tethering block. Defaults to False.
        """
        super().__init__(BLOCKS_DIR / 'righting_force_block.lmp')

        self.epsilon: float = epsilon
        self.delta_r: float = delta_r
        self.r_zero: float = r_zero
        self.target_group: str = target_group
        self.define_variables: bool = define_variables