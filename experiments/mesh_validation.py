
from pathlib import Path
from pprint import pprint

import numpy as np
import trimesh

from contactforge.assets.mesh import inspect_mesh


output_dir = Path("examples/meshes")
output_dir.mkdir(parents=True, exist_ok=True)

# A closed rectangular solid.
box = trimesh.creation.box(extents=[0.2, 0.3, 0.4])
box.export(output_dir / "valid_box.stl")

# Deliberately remove one triangle from the surface.
open_box = box.copy()

keep_faces = np.ones(len(open_box.faces), dtype=bool)
keep_faces[0] = False

open_box.update_faces(keep_faces)
open_box.remove_unreferenced_vertices()
open_box.export(output_dir / "open_box.stl")

for filename in ["valid_box.stl", "open_box.stl"]:
    print(f"\n--- {filename} ---")
    pprint(inspect_mesh(output_dir / filename))
