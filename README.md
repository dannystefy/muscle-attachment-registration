# Muscle Attachment Transfer

Automatically transfers muscle attachment points from a reference bone to a patient-specific bone using non-rigid surface registration (Fast-RNRR).

Given a reference bone with known muscle attachment regions (stored as VTK point sets) and a patient bone of the same type, the pipeline deforms the patient mesh to match the reference, maps the attachment points via nearest-vertex lookup, then projects them back onto the original patient geometry. Results are saved as VTK files and visualised with Open3D.

---

## Requirements

- **Python 3.11** — this project was tested with Python 3.11
- **Windows** — the prebuilt `Fast_RNRR.exe` and `OpenMeshCore.dll` in `win/` are Windows binaries

Install Python dependencies:

```bash
pip install -r requirements.txt
```
---

## Repository structure

```
├── data/
│   ├── reference_data/          # Reference bone meshes (.obj) + VTK attachment files
│   └── source_data/
│       └── 002/                 # Example patient data (pelvis + femur)
├── external/
│   └── FAST_RNRR/               # Non-rigid registration library (git submodule)
├── scripts/
│   └── run.py                   # Main entry point
├── src/
│   ├── __init__.py
│   ├── muscle_transfer.py       # VTK I/O, nearest-vertex lookup
│   ├── registration.py          # Fast-RNRR wrapper
│   └── visualization.py         # Open3D visualisation
├── win/
│   ├── Fast_RNRR.exe            # Prebuilt Windows binary
│   └── OpenMeshCore.dll
└── requirements.txt
```

---

## Installation

```bash
git clone --recurse-submodules https://github.com/dannystefy/muscle-attachment-registration
cd muscle-attachment-registration
pip install -r requirements.txt
```

The `--recurse-submodules` flag is needed to pull the Fast-RNRR source in `external/`. If you cloned without it, run:

```bash
git submodule update --init --recursive
```

On **Windows** the prebuilt binary in `win/` is used automatically.

On **Linux** you need to build Fast-RNRR from source. First install the required dependencies:

```bash
# Ubuntu / Debian
sudo apt install libeigen3-dev
sudo apt install libopenmesh-dev
```

Then build:

```bash
cd external/FAST_RNRR
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## Usage

All commands are run from the repository root.

### Demo — built-in patient data

Runs the full pipeline on the example data in `data/source_data/002/`:

```bash
python scripts/run.py --demo --pelvis
python scripts/run.py --demo --femur
```

### Custom patient bone

Provide a path to your own `.obj` file:

```bash
python scripts/run.py --pelvis --patient path/to/Pelvis.obj
python scripts/run.py --femur  --patient path/to/Femur_R.obj
```

### Additional flags

| Flag | Description |
|------|-------------|
| `--no-window` | Skip the interactive viewer, save PNG snapshot only |
| `--force` | Ignore cached registration results and re-run from scratch |

---

## Output

All outputs are written to `muscle_mapping_<bone>/` next to the patient `.obj` file:

```
source_data/002/
└── muscle_mapping_pelvis/
    ├── result1_pelvis.obj       # Patient deformed to reference space
    ├── result2_pelvis.obj       # Result deformed back to patient space
    ├── snapshot_pelvis.png      # Visualisation screenshot
    ├── _tmp1_pelvis/            # Centred meshes for pass 1
    ├── _tmp2_pelvis/            # Centred meshes for pass 2
    └── vtk/
        ├── R_Gluteus_Maximus_Ori_transferred.vtk
        ├── R_Gluteus_Medius_Ori_transferred.vtk
        ├── R_Iliacus_Ori_transferred.vtk
        └── R_Adductor_Brevis_Ori_transferred.vtk
```

VTK files are in the original patient coordinate system and can be loaded directly in 3D Slicer, ParaView, or any other tool that supports VTK POLYDATA.

---

## How it works

1. **Pass 1 — patient → reference:** the patient mesh is non-rigidly registered to the reference bone using Fast-RNRR. Both meshes are centred before registration and the result is shifted back into the reference coordinate system.
2. **Attachment mapping:** for each VTK attachment point on the reference, the nearest vertex on the registered mesh is found via KD-tree lookup.
3. **Pass 2 — result → patient:** the registered mesh is mapped back onto the original patient bone, carrying the attachment vertex indices with it.
4. **Export:** final attachment positions are written as VTK POLYDATA files and displayed in an Open3D window.

---

## Citation

This project uses [Fast-RNRR](https://github.com/Juyong/Fast_RNRR) for non-rigid registration. If you use this tool in your research, please also cite the original paper:

```bibtex
@InProceedings{Yao_2020_CVPR,
    author    = {Yao, Yuxin and Deng, Bailin and Xu, Weiwei and Zhang, Juyong},
    title     = {Quasi-Newton Solver for Robust Non-Rigid Registration},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2020}
}
```

> **Note:** Fast-RNRR is protected under patent and may only be used for research purposes.
> For commercial use, contact juyong@ustc.edu.cn.

---

## About

This tool was developed as part of a bachelor's project at the University of West Bohemia, Faculty of Applied Sciences, Department of Computer Science and Engineering.

---

## Licence

The source code in this repository (`src/`, `scripts/`) is released under the **MIT licence** — see [`LICENSE`](LICENSE).

However, the overall project depends on third-party components with additional restrictions. **Commercial use of this project as a whole is not permitted** due to the following:

| Component | Licence | Restriction |
|-----------|---------|-------------|
| `src/`, `scripts/` | MIT | None |
| `data/reference_data/` | [CC BY-NC-SA 2.0 BE](https://creativecommons.org/licenses/by-nc-sa/2.0/be/deed.fr) | Non-commercial only |
| `data/source_data/` | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Non-commercial only |
| `external/FAST_RNRR/` | Patent-protected | Research use only — contact juyong@ustc.edu.cn for commercial licensing |