# Bacterial Septation

Working repository for work on bacterial septation.
Currently includes:
- ODE for septal closure taken from [original paper](https://doi.org/10.1242/jcs.166926)

## Installation

The most straightforward way is using `conda` and `pip`.
A small environment file is included as `environment.yml`, running `conda env create -f environment.yml` creates the environment `bacterial_septation`.
Run `conda activate bacterial_septation` and then
```bash
pip install -e .
```
to install an editable version of this package.
After doing so, all notebooks should work.

Additionally, in the future, analysis notebooks may depend on the Ovito Python package.
Run `conda activate bacterial_septation` and then either follow Ovito's [Instructions](https://docs.ovito.org/python/introduction/installation.html) or
```bash
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
```


## Lammps

Running the associated `lammps` scripts obviously requires a version of `lammps`.

The `bacterial_septation` `conda` environment comes with a version of `libomp`, which may be needed on all systems.
If you're not sure if you have `libomp` installed, after [Installation](#installation) run
```bash
conda activate bacterial_septation
conda install conda-forge::openmp
```
and then follow the rest of these instructions.
If you are on MacOS use `brew` to install this, as `conda` only has installations for Windows, Linux and Intel MacOs.

Clone [https://github.com/Saric-Group/lammps] and find the changes originally added by Chris and modified to the new lammps version (currently in branch `feature/update-treadmilling/stable`).
The new (testing) version of `lammps` can be built via
```bash
# module load openmpi # Uncomment when compiling on an HPC
mkdir build; cd build
cmake ../cmake -D PKG_COMPRESS=yes -D PKG_MOLECULE=yes -D PKG_EXTRA-MOLECULE=yes -D PKG_REACTION=yes -D PKG_EXTRA-PAIR=yes -D PKG_ASPHERE=yes -D PKG_OPENMP=yes
cmake --build .
```

### Known Issues

- When using `mpirun`, moleculeIDs are not updated properly upon particle insertion, so you cannot rely on the molecule ID to count the number of filaments in a system.
- `fix set` does not update atom variables properly when atoms are inserted and deleted.
    Whenever a particle is inserted, there already is already a value in the `atom/property` vector, that seems somehow pre-computed.
    This is also the reason why lifetime tracking via `fix set` does not work and it has to be done in `fix bond/react`.