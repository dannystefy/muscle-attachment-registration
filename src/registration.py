import subprocess
import numpy as np
from pathlib import Path
import platform

if platform.system() == "Windows":
    RNRR_BINARY = Path(__file__).parent.parent / "win/Fast_RNRR.exe"
else:
    RNRR_BINARY = Path(__file__).parent.parent / "external/FAST_RNRR/build/Fast_RNRR"

RNRR_OUTPUT_SUFFIX = "res.obj"


def load_obj(path: Path) -> tuple[np.ndarray, list, list]:
    vertices, faces, other = [], [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                faces.append(line)
            else:
                other.append(line)
    return np.array(vertices), faces, other


def save_obj(path: Path, vertices: np.ndarray, faces: list, other: list) -> None:
    with open(path, "w") as f:
        for line in other:
            f.write(line)
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(face)


# ── Centring ─────────────────────────────────────────────────────────────────

def center_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = vertices.mean(axis=0)
    return vertices - centroid, centroid


def create_centered_copy(input_path: Path, output_path: Path) -> np.ndarray:
    """Save a centred copy of the mesh and return its centroid."""
    verts, faces, other = load_obj(input_path)
    centered, centroid = center_vertices(verts)
    save_obj(output_path, centered, faces, other)
    return centroid


# ── Registration pipeline ─────────────────────────────────────────────────────

def register_meshes(source: Path, target: Path, output: Path, landmarks: Path = None, tmp_dir: Path = None) -> tuple[Path, np.ndarray, np.ndarray]:
    """
    Centre both meshes, run Fast_RNRR, then shift the result back into the
    target coordinate system.

    source  – patient bone (deformed to match target)
    target  – reference bone carrying the attachment points
    output  – deformed source in the target's original coordinate system

    Returns
    -------
    output       : Path   – result OBJ in the target's original space
    src_centroid : ndarray – centroid of the source (patient) mesh
    tgt_centroid : ndarray – centroid of the target (reference) mesh
    """
    if not RNRR_BINARY.exists():
        raise RuntimeError(
            f"Fast_RNRR binary not found: {RNRR_BINARY}\n"
            "Run scripts/build_rnrr.sh to build it first."
        )

    source, target, output = Path(source), Path(target), Path(output)
    tmp_dir = Path(tmp_dir) if tmp_dir else output.parent / "_tmp_rnrr"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Centre source and target
    src_centered    = tmp_dir / "source_centered.obj"
    tgt_centered    = tmp_dir / "target_centered.obj"
    rnrr_out_stem   = tmp_dir / "registered_centered"   
    rnrr_out_actual = Path(str(rnrr_out_stem) + RNRR_OUTPUT_SUFFIX)

    src_centroid = create_centered_copy(source, src_centered)
    tgt_centroid = create_centered_copy(target, tgt_centered)

    # 2. Run Fast_RNRR on the centred meshes
    cmd = [str(RNRR_BINARY), str(src_centered), str(tgt_centered), str(rnrr_out_stem)]
    if landmarks:
        cmd.append(str(landmarks))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Fast_RNRR failed:\n{result.stderr}")
    if not rnrr_out_actual.exists():
        raise RuntimeError(
            f"Fast_RNRR produced no output. Expected: {rnrr_out_actual}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # 3. Shift result back into the target's original coordinate system
    verts, faces, other = load_obj(rnrr_out_actual)
    save_obj(output, verts + tgt_centroid, faces, other)

    return output, src_centroid, tgt_centroid