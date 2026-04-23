import numpy as np
import pyvista as pv
from scipy.spatial import KDTree


# ---------- 读取 OFF 文件 ----------
def load_off(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert lines[0] == "OFF", "Not a valid OFF file."

    n_vertices, n_faces, _ = map(int, lines[1].split())

    vertices = np.array([
        list(map(float, lines[2 + i].split()))
        for i in range(n_vertices)
    ])

    faces = []
    start = 2 + n_vertices
    for i in range(n_faces):
        data = list(map(int, lines[start + i].split()))
        faces.extend(data)

    faces = np.array(faces, dtype=np.int64)
    return vertices, faces


# ---------- 计算每个 face 的 centroid ----------
def compute_centroids(vertices, faces):
    # 假设每个 face 都是三角形，OFF 格式为: 3 v0 v1 v2
    faces_tri = faces.reshape(-1, 4)[:, 1:]
    centroids = vertices[faces_tri].mean(axis=1)
    return centroids


# ---------- 找近邻 faces ----------
def find_near_faces(centroids, seed_face, k=28):
    tree = KDTree(centroids)
    _, idx = tree.query(centroids[seed_face], k=k)
    return np.atleast_1d(idx)


# ---------- 计算 MBB ----------
def compute_mbb(vertices, faces, near_faces):
    faces_tri = faces.reshape(-1, 4)[:, 1:]
    selected_vertices = vertices[faces_tri[near_faces]].reshape(-1, 3)

    bounds = (
        selected_vertices[:, 0].min(), selected_vertices[:, 0].max(),
        selected_vertices[:, 1].min(), selected_vertices[:, 1].max(),
        selected_vertices[:, 2].min(), selected_vertices[:, 2].max()
    )
    return pv.Box(bounds=bounds).outline()


# ---------- 构建带颜色的 mesh ----------
def build_mesh(vertices, faces, groups):
    n_faces = len(faces.reshape(-1, 4))
    colors = np.zeros((n_faces, 3), dtype=float)
    colors[:] = [0.75, 0.75, 0.75]  # 默认灰色

    for face_ids, color in groups:
        colors[face_ids] = color

    mesh = pv.PolyData(vertices, faces)
    mesh.cell_data["colors"] = colors
    return mesh


# ---------- 主程序 ----------
def main():
    vertices, faces = load_off("nuclei.pt")
    centroids = compute_centroids(vertices, faces)

    k = 10

    # ===== 物体 A =====
    seed_a1, seed_a2 = 50, 60
    near_a1 = find_near_faces(centroids, seed_a1, k=k)
    near_a2 = find_near_faces(centroids, seed_a2, k=k)

    box_a1 = compute_mbb(vertices, faces, near_a1)
    box_a2 = compute_mbb(vertices, faces, near_a2)

    mesh_a = build_mesh(
        vertices, faces,
        groups=[
            (near_a1, [0.0, 0.8, 0.0]),   # 深绿
            (near_a2, [1, 0.5, 0]),   # 浅绿
        ]
    )

    # ===== 物体 B：复制 A，并平移到右边 =====
    seed_b1, seed_b2 = 170, 150
    near_b1 = find_near_faces(centroids, seed_b1, k=k)
    near_b2 = find_near_faces(centroids, seed_b2, k=k)

    box_b1 = compute_mbb(vertices, faces, near_b1)
    box_b2 = compute_mbb(vertices, faces, near_b2)

    mesh_b = build_mesh(
        vertices, faces,
        groups=[
            (near_b1, [0.0, 0.8, 0.0]),   # 蓝色
            (near_b2, [1, 0.5, 0]),   # 浅蓝
        ]
    )

    # 平移距离：按模型宽度自动估计
    dx = mesh_a.bounds[1] - mesh_a.bounds[0]
    offset = (dx * 1.4, 0.0, 0.0)

    mesh_b = mesh_b.copy(deep=True)
    mesh_b.translate(offset, inplace=True)

    box_b1 = box_b1.copy(deep=True)
    box_b2 = box_b2.copy(deep=True)
    box_b1.translate(offset, inplace=True)
    box_b2.translate(offset, inplace=True)

    # ---------- 可视化 ----------
    plotter = pv.Plotter(window_size=(1600, 900))

    # 物体 A
    plotter.add_mesh(mesh_a, scalars="colors", rgb=True, show_edges=True)
    plotter.add_mesh(box_a1, color="green", line_width=4, render_lines_as_tubes=True)
    plotter.add_mesh(box_a2, color="orange", line_width=4, render_lines_as_tubes=True)

    # 物体 B
    plotter.add_mesh(mesh_b, scalars="colors", rgb=True, show_edges=True)
    plotter.add_mesh(box_b1, color="green", line_width=4, render_lines_as_tubes=True)
    plotter.add_mesh(box_b2, color="orange", line_width=4, render_lines_as_tubes=True)

    plotter.add_text(f"A: {seed_a1}, {seed_a2}    B: {seed_b1}, {seed_b2}", font_size=12)
    plotter.show()


if __name__ == "__main__":
    main()