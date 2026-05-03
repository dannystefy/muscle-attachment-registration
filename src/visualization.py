"""
visualization.py
================
Visualise transferred muscle attachments on a patient mesh.
----------
visualize_all_attachments()  – interactive Open3D window; S = snapshot
save_snapshot()              – silent offscreen render (batch / --no-window)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d



COLOR_PALETTE = [
    [0.93, 0.35, 0.18],  # coral  – Gluteus Maximus
    [0.23, 0.55, 0.83],  # blue   – Gluteus Medius
    [0.13, 0.62, 0.46],  # teal   – Iliacus
    [0.91, 0.62, 0.15],  # amber  – Adductor Brevis
    [0.60, 0.29, 0.72],  # purple – spare
    [0.20, 0.73, 0.73],  # cyan   – spare
]

_MESH_COLOR = [0.75, 0.75, 0.75]
_BG_COLOR   = [0.12, 0.12, 0.12]
_SNAP_W     = 1920
_SNAP_H     = 1080



def _build_geometries(patient_mesh: o3d.geometry.TriangleMesh, muscles: dict, radius: float) -> list:
    geoms = []

    mesh = o3d.geometry.TriangleMesh(patient_mesh)
    mesh.paint_uniform_color(_MESH_COLOR)
    mesh.compute_vertex_normals()
    geoms.append(mesh)

    for i, (name, pts) in enumerate(muscles.items()):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        if radius > 0:
            for pt in pts:
                s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                s.translate(pt)
                s.paint_uniform_color(color)
                s.compute_vertex_normals()
                geoms.append(s)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color(color)
            geoms.append(pcd)

    return geoms


# ── Interactive window ────────────────────────────────────────────────────────

def visualize_all_attachments(patient_mesh: o3d.geometry.TriangleMesh, 
                                muscles: dict, window_title: str = "Muscle attachment",
                                radius: float = 2.0,snapshot_path: Optional[Path] = None,
                            ) -> None:
    """
    Open an interactive Open3D window.

    Key bindings
    ------------
    S  –  save a PNG snapshot of the current view
           (first press: snapshot.png, subsequent: snapshot_1.png, …)
    Q  –  close the window
    """
    snap_base = Path(snapshot_path) if snapshot_path is not None else Path("snapshot.png")
    snap_base.parent.mkdir(parents=True, exist_ok=True)

    counter = [0]

    def _next_path() -> Path:
        n = counter[0]
        counter[0] += 1
        if n == 0:
            return snap_base
        return snap_base.with_stem(f"{snap_base.stem}_{n}")

    geoms = _build_geometries(patient_mesh, muscles, radius)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{window_title}   |   S = save snapshot   |   Q = quit",
        width=1280,
        height=800,
    )

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color    = np.array(_BG_COLOR)
    opt.point_size          = 8.0
    opt.mesh_show_back_face = True

    def _on_s(v):
        out = _next_path()
        v.capture_screen_image(str(out), do_render=True)
        print(f"  [S] Snapshot saved: {out}")
        return False

    vis.register_key_callback(ord("S"), _on_s)
    vis.register_key_callback(ord("s"), _on_s)

    print()
    print("  ┌──────────────────────────────────────────┐")
    print("  │  Adjust the view and press  S = snapshot  │")
    print("  │  Each S press saves a new file            │")
    print("  │  Close the window or press  Q  to quit    │")
    print("  └──────────────────────────────────────────┘")
    print()

    vis.run()
    vis.destroy_window()


# ── Silent offscreen render ───────────────────────────────────────────────────

def save_snapshot(patient_mesh: o3d.geometry.TriangleMesh, muscles: dict, output_path: Path,
                  radius: float = 2.0, width: int = _SNAP_W, height: int = _SNAP_H,
                  ) -> None:
    """
    Save a PNG without an interactive window.
    The window briefly appears and closes by itself (Windows limitation).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    geoms = _build_geometries(patient_mesh, muscles, radius)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="snapshot", width=width, height=height, visible=False)

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color    = np.array(_BG_COLOR)
    opt.point_size          = 8.0
    opt.mesh_show_back_face = True

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_path), do_render=True)
    vis.destroy_window()
    print(f"  Snapshot saved: {output_path}")