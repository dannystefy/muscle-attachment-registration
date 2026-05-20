import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registration import register_meshes, interpolate_deformation, load_obj
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



# ── Pipeline ─────────────────────────────────────────────────────────────────
# --demo is just a shortcut that calls run() with the default SRC_DIR patient path.
# No separate function needed.

def run(bone: str, patient_path: Path, no_window: bool = False, demo: bool = False) -> None:
    cfg   = CONFIGS[bone]
    label = cfg["label"]

    demo_tag = "  [DEMO]" if demo else ""
    print(f"\n{'═'*60}")
    print(f"  {label}{demo_tag}  [patient: {patient_path}]")
    print(f"{'═'*60}")

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

    patient_mesh_c, _ = centered_mesh(patient_mesh, pat_centroid)

    ref_verts_raw, _, _     = load_obj(cfg["ref_mesh"])
    patient_verts_raw, _, _ = load_obj(patient_path)

    # ── Sbírání VTK bodů ──────────────────────────────────────────────────────
    vtk_info = {}
    all_vtk  = []
    for vtk_path in cfg["vtk_files"]:
        name = vtk_path.stem
        if not vtk_path.exists():
            print(f"  ERROR: file not found: {vtk_path}")
            continue
        pts = load_vtk_points(str(vtk_path))
        if pts is None or len(pts) == 0:
            print(f"  ERROR: no points in VTK file: {vtk_path}")
            continue
        s = len(all_vtk)
        all_vtk.extend(pts.tolist())
        vtk_info[name] = (s, s + len(pts))
    all_vtk = np.array(all_vtk)

    # ── Pass 1: reference → patient ───────────────────────────────────────────
    print("[1/2] Registration: reference → patient ...")
    result1_path, _, _ = register_meshes(
        source=cfg["ref_mesh"],
        target=patient_path,
        output=result1_path,
        tmp_dir=tmp1_dir,
    )
    result1_verts_raw, _, _ = load_obj(result1_path)
    transferred_vtk = interpolate_deformation(ref_verts_raw, result1_verts_raw, all_vtk)

    # ── Pass 2: patient → reference (round-trip metrika) ──────────────────────
    print("[2/2] Registration: patient → reference (round-trip) ...")
    result2_path, _, _ = register_meshes(
        source=patient_path,
        target=cfg["ref_mesh"],
        output=result2_path,
        tmp_dir=tmp2_dir,
    )
    result2_verts_raw, _, _ = load_obj(result2_path)
    roundtrip_vtk = interpolate_deformation(patient_verts_raw, result2_verts_raw, transferred_vtk)

    # ── Přenos úponů + round-trip metriky ─────────────────────────────────────
    results = {}
    for vtk_path in cfg["vtk_files"]:
        name = vtk_path.stem
        if name not in vtk_info:
            continue
        s, e = vtk_info[name]

        orig_vtk        = all_vtk[s:e]
        trans_vtk_c     = transferred_vtk[s:e] - pat_centroid
        roundtrip_pos   = roundtrip_vtk[s:e]
        roundtrip_pos_c = roundtrip_pos - ref_centroid

        n    = min(len(orig_vtk), len(roundtrip_pos))
        hd   = max(directed_hausdorff(roundtrip_pos[:n], orig_vtk[:n])[0],
                   directed_hausdorff(orig_vtk[:n], roundtrip_pos[:n])[0])
        mse  = np.mean(np.linalg.norm(roundtrip_pos[:n] - orig_vtk[:n], axis=1) ** 2)
        rmse = np.sqrt(mse)

        ref_scale = np.linalg.norm(ref_verts_raw.max(axis=0) - ref_verts_raw.min(axis=0))
        hd_norm   = hd   / ref_scale * 100
        rmse_norm = rmse / ref_scale * 100

        print(f"\n── {name}")
        print(f"  Points:                  {n}")
        print(f"  Hausdorff (round-trip):  {hd:.4f} mm  ({hd_norm:.2f} %)")
        print(f"  MSE (round-trip):        {mse:.4f} mm²")
        print(f"  RMSE (round-trip):       {rmse:.4f} mm  ({rmse_norm:.2f} %)")

        patient_tendon_ids = np.asarray(find_nearest_vertices(patient_mesh_c, trans_vtk_c))

        results[name] = {
            "patient_tendon_ids": patient_tendon_ids,
            "transferred_vtk_c":  trans_vtk_c,
            "vtk_points_c":       orig_vtk - ref_centroid,
            "roundtrip_vtk_c":    roundtrip_pos_c,
            "hausdorff": hd, "mse": mse, "rmse": rmse,
        }

    # ── Souhrn ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SUMMARY – {label}")
    print(f"{'─'*60}")
    print(f"{'Muscle':<38} {'Hausdorff':>12} {'RMSE':>10}")
    print(f"{'─'*60}")
    for name, r in results.items():
        print(f"{name:<38} {r['hausdorff']:>10.4f} mm {r['rmse']:>8.4f} mm")
    print(f"{'═'*60}\n")

    # ── Uložení VTK ───────────────────────────────────────────────────────────
    vtk_out_dir = out_dir / "vtk"
    vtk_out_dir.mkdir(parents=True, exist_ok=True)
    muscles = {}
    verts   = np.asarray(patient_mesh_c.vertices)
    for name, r in results.items():
        muscles[name] = verts[r["patient_tendon_ids"]]
        pts_world = r["transferred_vtk_c"] + pat_centroid
        vtk_path  = vtk_out_dir / f"{name}_transferred.vtk"
        save_vtk_points(vtk_path, pts_world)
        print(f"  VTK saved: {vtk_path.relative_to(out_dir.parent)}")

    # ── Vizualizace ───────────────────────────────────────────────────────────
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
            "\n"
            "  User – full pipeline with a custom patient bone:\n"
            "    python run.py --pelvis                                     # default path from CONFIGS\n"
            "    python run.py --pelvis --patient path/to/Pelvis.obj\n"
            "    python run.py --femur  --patient path/to/Femur_R.obj\n"
            "    python run.py --pelvis --patient path/to/Pelvis.obj --no-window\n"
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

    args = parser.parse_args()

    bone = "pelvis" if args.pelvis else "femur"

    if args.demo:
        if args.patient is not None:
            parser.error("--patient cannot be combined with --demo.")
        patient_path = CONFIGS[bone]["patient_mesh"]
        print(f"  [DEMO] Using built-in patient path: {patient_path}")
        run(bone, patient_path=patient_path, no_window=args.no_window, demo=True)
    else:
        if args.patient is not None:
            patient_path = Path(args.patient)
        else:
            patient_path = CONFIGS[bone]["patient_mesh"]
            print(f"  (Using default patient path from CONFIGS: {patient_path})")

        run(bone, patient_path=patient_path, no_window=args.no_window)


if __name__ == "__main__":
    main()