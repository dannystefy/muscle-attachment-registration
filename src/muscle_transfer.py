import numpy as np
import open3d as o3d
from pathlib import Path


def load_vtk_points(vtk_path: str) -> np.ndarray:
    points = []
    in_points = False
    n_points = 0

    with open(vtk_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("POINTS"):
                parts = line.split()
                n_points = int(parts[1])
                in_points = True
                continue
            if in_points:
                nums = line.split()
                i = 0
                while i + 2 < len(nums):
                    points.append([float(nums[i]), float(nums[i+1]), float(nums[i+2])])
                    i += 3
                if len(points) >= n_points:
                    break

    return np.array(points)


def save_vtk_points(path: Path, points: np.ndarray) -> None:
    """
    Save an (N, 3) array of 3-D points as a VTK POLYDATA file.

    The output format mirrors the reference VTK files used as input:
      - POINTS section with float coordinates
      - VERTICES section (one vertex cell per point)
      - CELL_DATA / POINT_DATA with a bit SCALARS block (all ones)
    """
    n = len(points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        # ── Header ───────────────────────────────────────────────────────────
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # ── Points ───────────────────────────────────────────────────────────
        f.write(f"POINTS {n} float\n")
        for i, (x, y, z) in enumerate(points):
            f.write(f"{x:.6g} {y:.6g} {z:.6g}")
            # three triplets per line (matches reference style)
            f.write("\n" if (i + 1) % 3 == 0 else " ")
        if n % 3 != 0:
            f.write("\n")

        # ── Vertices (one cell per point) ────────────────────────────────────
        f.write(f"VERTICES {n} {n * 2}\n")
        for i in range(n):
            f.write(f"1 {i} \n")

        # ── Scalar data (all ones) ────────────────────────────────────────────
        f.write(f"CELL_DATA {n}\n")
        f.write(f"POINT_DATA {n}\n")
        f.write("SCALARS scalars bit\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n):
            f.write("1")
            f.write("\n" if (i + 1) % 8 == 0 else " ")
        if n % 8 != 0:
            f.write("\n")


def find_nearest_vertices(mesh: o3d.geometry.TriangleMesh, query_points: np.ndarray) -> list[int]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    tree = o3d.geometry.KDTreeFlann(pcd)

    vertex_ids = []
    for pt in query_points:
        _, idx, _ = tree.search_knn_vector_3d(pt, 1)
        vertex_ids.append(idx[0])
    return vertex_ids


def project_points_to_mesh(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    tree = o3d.geometry.KDTreeFlann(pcd)

    verts = np.asarray(mesh.vertices)
    projected = []
    for pt in points:
        _, idx, _ = tree.search_knn_vector_3d(pt, 1)
        projected.append(verts[idx[0]])
    return np.array(projected)