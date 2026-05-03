import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registration import register_meshes
from src.muscle_transfer import load_vtk_points, find_nearest_vertices, save_vtk_points
from src.visualization import visualize_all_attachments
from src.visualization import save_snapshot


# Path
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
REF_DIR    = DATA_DIR / "reference_data"
SRC_DIR    = DATA_DIR / "source_data" / "002"

# Config
CONFIGS = {
    "pelvis": {
        "label":        "Pelvis",
        "ref_mesh":     REF_DIR / "Pelvis.obj",
        "patient_mesh": SRC_DIR / "Pelvis.obj",
        "vtk_files": [
            REF_DIR / "R_Gluteus_Maximus_Ori.vtk",
            REF_DIR / "R_Gluteus_Medius_Ori.vtk",
            REF_DIR / "R_Iliacus_Ori.vtk",
            REF_DIR / "R_Adductor_Brevis_Ori.vtk",
        ],
    },
    "femur": {
        "label":        "Femur",
        "ref_mesh":     REF_DIR / "R_Femur.obj",
        "patient_mesh": SRC_DIR / "Femur_R.obj",
        "vtk_files": [
            REF_DIR / "R_Gluteus_Maximus_Ins.vtk",
            REF_DIR / "R_Gluteus_Medius_Ins.vtk",
            REF_DIR / "R_Iliacus_Ins.vtk",
            REF_DIR / "R_Adductor_Brevis_Ins.vtk",
        ],
    },
}

COLOR_PALETTE = [
    (0.93, 0.35, 0.18),  # coral  – Gluteus Maximus
    (0.23, 0.55, 0.83),  # blue   – Gluteus Medius
    (0.13, 0.62, 0.46),  # teal   – Iliacus
    (0.91, 0.62, 0.15),  # amber  – Adductor Brevis
]



def centered_mesh(mesh: o3d.geometry.TriangleMesh, centroid: np.ndarray,) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    verts = np.asarray(mesh.vertices) - centroid
    m = deepcopy(mesh)
    m.vertices = o3d.utility.Vector3dVector(verts)
    return m, verts


def load_mesh(path: Path, label: str) -> o3d.geometry.TriangleMesh:
    if not path.exists():
        raise FileNotFoundError(f"{label}: file not found: {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(np.asarray(mesh.vertices)) == 0:
        raise ValueError(f"{label}: mesh contains no vertices: {path}")
    return mesh


def centroid_from_original(original_path: Path) -> np.ndarray:
    verts = []
    with open(original_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts).mean(axis=0)


def cache_valid(result_path: Path, tmp_dir: Path) -> bool:
    """Returns True if the registration result exists and tmp_dir contains the centered OBJ files."""
    return (
        result_path.exists()
        and (tmp_dir / "target_centered.obj").exists()
        and (tmp_dir / "source_centered.obj").exists()
    )


# ── Pipeline ─────────────────────────────────────────────────────────────────
# --demo is just a shortcut that calls run() with the default SRC_DIR patient path.
# No separate function needed.

def run(bone: str, patient_path: Path, no_window: bool = False, force: bool = False, demo: bool = False) -> None:
    """
    Full two-pass registration pipeline.
    Transfers attachments from the reference bone onto the patient bone.
    All outputs (result1/2, tmp folders, snapshot) are saved to
    {patient_path.parent}/muscle_mapping/.

    demo=True just prints [DEMO] in the header; no other behavioural difference.
    """
    cfg   = CONFIGS[bone]
    label = cfg["label"]

    demo_tag = "  [DEMO]" if demo else ""
    print(f"\n{'═'*60}")
    print(f"  {label}{demo_tag}  [patient: {patient_path}]")
    print(f"{'═'*60}")

    # Output folder next to the patient source file
    out_dir = patient_path.parent / f"muscle_mapping_{bone}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result1_path = out_dir / f"result1_{bone}.obj"
    result2_path = out_dir / f"result2_{bone}.obj"
    tmp1_dir     = out_dir / f"_tmp1_{bone}"
    tmp2_dir     = out_dir / f"_tmp2_{bone}"
    tmp1_dir.mkdir(parents=True, exist_ok=True)
    tmp2_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Output folder: {out_dir}")

    ref_mesh     = load_mesh(cfg["ref_mesh"], "Reference")
    patient_mesh = load_mesh(patient_path,   "Patient")

    ref_centroid = np.asarray(ref_mesh.vertices).mean(axis=0)
    pat_centroid = np.asarray(patient_mesh.vertices).mean(axis=0)

    # ── Pass 1: patient → reference ──────────────────────────────────────
    if not force and cache_valid(result1_path, tmp1_dir):
        print(f"[1/2] Registration: using cache  ({result1_path.name})")
        tgt_centroid1 = centroid_from_original(cfg["ref_mesh"])
    else:
        msg = "(--force, overwriting cache)" if force else ""
        print(f"[1/2] Registration: patient → reference {msg}...")
        result1_path, _, tgt_centroid1 = register_meshes(
            source=patient_path,
            target=cfg["ref_mesh"],
            output=result1_path,
            tmp_dir=tmp1_dir,
        )

    result1_mesh = load_mesh(result1_path, "Result1")
    result1_mesh_c, res1_verts_c = centered_mesh(result1_mesh, tgt_centroid1)

    # ── Pass 2: result1 → patient ─────────────────────────────────────────
    if not force and cache_valid(result2_path, tmp2_dir):
        print(f"[2/2] Registration: using cache  ({result2_path.name})")
        tgt_centroid2 = centroid_from_original(str(patient_path))
    else:
        msg = "(--force, overwriting cache)" if force else ""
        print(f"[2/2] Registration: result1 → patient {msg}...")
        result2_path, _, tgt_centroid2 = register_meshes(
            source=result1_path,
            target=patient_path,
            output=result2_path,
            tmp_dir=tmp2_dir,
        )

    result2_mesh = load_mesh(result2_path, "Result2")
    result2_mesh_c, res2_verts_c = centered_mesh(result2_mesh, tgt_centroid2)
    patient_mesh_c, pat_verts_c  = centered_mesh(patient_mesh, pat_centroid)

    # ── Attachment transfer ───────────────────────────────────────────────────
    results = {}

    for vtk_path in cfg["vtk_files"]:
        name = vtk_path.stem
        print(f"\n── {name}")

        if not vtk_path.exists():
            print(f"  ERROR: file not found: {vtk_path}")
            continue

        vtk_points = load_vtk_points(str(vtk_path))
        if vtk_points is None or len(vtk_points) == 0:
            print(f"  ERROR: no points in VTK file: {vtk_path}")
            continue

        vtk_points_c = vtk_points - ref_centroid

        result1_tendon_ids = find_nearest_vertices(result1_mesh_c, vtk_points_c)
        result1_tendon_pos = res1_verts_c[result1_tendon_ids]

        hd = max(
            directed_hausdorff(result1_tendon_pos, vtk_points_c)[0],
            directed_hausdorff(vtk_points_c, result1_tendon_pos)[0],
        )
        n    = min(len(result1_tendon_pos), len(vtk_points_c))
        mse  = np.mean(np.linalg.norm(result1_tendon_pos[:n] - vtk_points_c[:n], axis=1) ** 2)
        rmse = np.sqrt(mse)

        print(f"  Points:                  {len(vtk_points)}")
        print(f"  Hausdorff after reg.:    {hd:.4f} mm")
        print(f"  MSE after reg.:          {mse:.4f} mm²")
        print(f"  RMSE after reg.:         {rmse:.4f} mm")

        result1_tendon_ids = np.asarray(result1_tendon_ids)
        tendon_pos_c       = res2_verts_c[result1_tendon_ids]
        patient_tendon_ids = np.asarray(find_nearest_vertices(patient_mesh_c, tendon_pos_c))

        results[name] = {
            "patient_tendon_ids": patient_tendon_ids,
            "result1_tendon_ids": result1_tendon_ids,
            "vtk_points_c":       vtk_points_c,
            "hausdorff":          hd,
            "mse":                mse,
            "rmse":               rmse,
        }

    # ── Souhrn ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SUMMARY – {label}")
    print(f"{'─'*60}")
    print(f"{'Muscle':<38} {'Hausdorff':>12} {'RMSE':>10}")
    print(f"{'─'*60}")
    for name, r in results.items():
        print(f"{name:<38} {r['hausdorff']:>10.4f} mm {r['rmse']:>8.4f} mm")
    print(f"{'═'*60}\n")

    # ── Visualisation ─────────────────────────────────────────────────────────
    verts = np.asarray(patient_mesh_c.vertices)
    muscles = {
        name: verts[r["patient_tendon_ids"]]
        for name, r in results.items()
    }

    # ── Save transferred VTK files ────────────────────────────────────────────
    # Points are stored back in the original (non-centred) patient coordinate
    # system so they align with the patient OBJ file in downstream tools.
    vtk_out_dir = out_dir / "vtk"
    vtk_out_dir.mkdir(parents=True, exist_ok=True)
    for name, pts_c in muscles.items():
        pts_world = pts_c + pat_centroid          # undo centring
        vtk_path  = vtk_out_dir / f"{name}_transferred.vtk"
        save_vtk_points(vtk_path, pts_world)
        print(f"  VTK saved: {vtk_path.relative_to(out_dir.parent)}")

    snapshot_path = out_dir / f"snapshot_{bone}.png"

    if no_window:
        save_snapshot(
            patient_mesh=patient_mesh_c,
            muscles=muscles,
            output_path=snapshot_path,
            radius=2.0,
        )
    else:
        visualize_all_attachments(
            patient_mesh=patient_mesh_c,
            muscles=muscles,
            window_title=f"{label} – transferred attachments",
            radius=2.0,
            snapshot_path=snapshot_path,
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Muscle attachment transfer to a patient bone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "\n"
            "  Demo – full pipeline with built-in SRC_DIR patient data:\n"
            "    python run.py --demo --pelvis\n"
            "    python run.py --demo --femur\n"
            "    python run.py --demo --pelvis --no-window\n"
            "    python run.py --demo --pelvis --force\n"
            "\n"
            "  User – full pipeline with a custom patient bone:\n"
            "    python run.py --pelvis                                     # default path from CONFIGS\n"
            "    python run.py --pelvis --patient path/to/Pelvis.obj\n"
            "    python run.py --femur  --patient path/to/Femur_R.obj\n"
            "    python run.py --pelvis --patient path/to/Pelvis.obj --no-window\n"
            "    python run.py --pelvis --patient path/to/Pelvis.obj --force\n"
            "\n"
            "  All outputs (result1/2, tmp folders, snapshot) are saved to\n"
            "  muscle_mapping/ next to the patient OBJ file.\n"
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Demo mode: run the full two-pass registration using the built-in "
            f"SRC_DIR patient data ({SRC_DIR}). "
            "Cannot be combined with --patient."
        ),
    )

    bone_group = parser.add_mutually_exclusive_group(required=True)
    bone_group.add_argument(
        "--pelvis",
        action="store_true",
        help="Map pelvis attachments (Pelvis.obj).",
    )
    bone_group.add_argument(
        "--femur",
        action="store_true",
        help="Map femur attachments (Femur_R.obj).",
    )

    parser.add_argument(
        "--patient",
        metavar="PATH",
        default=None,
        help=(
            "Path to the patient OBJ file. "
            "If omitted, the default path from CONFIGS is used. "
            "Ignored in --demo mode."
        ),
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Skip the interactive window and only save a PNG snapshot.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-run registration (overwrites result1/result2).",
    )

    args = parser.parse_args()

    bone = "pelvis" if args.pelvis else "femur"

    if args.demo:
        if args.patient is not None:
            parser.error("--patient cannot be combined with --demo.")
        patient_path = CONFIGS[bone]["patient_mesh"]
        print(f"  [DEMO] Using built-in patient path: {patient_path}")
        run(bone, patient_path=patient_path, no_window=args.no_window, force=args.force, demo=True)
    else:
        if args.patient is not None:
            patient_path = Path(args.patient)
        else:
            patient_path = CONFIGS[bone]["patient_mesh"]
            print(f"  (Using default patient path from CONFIGS: {patient_path})")

        run(bone, patient_path=patient_path, no_window=args.no_window, force=args.force)


if __name__ == "__main__":
    main()