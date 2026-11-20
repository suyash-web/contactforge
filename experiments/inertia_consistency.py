
from pprint import pprint

from contactforge.assets.consistency import compare_inertia
from contactforge.assets.inertia import box_inertia


mesh_path = "examples/meshes/valid_box.stl"

mass = 2.0

# Physical dimensions: 0.2 x 0.3 x 0.4 meters.
correct_inertia = box_inertia(
    mass=mass,
    width=0.2,
    depth=0.3,
    height=0.4
)

# Deliberately incorrect but still positive-definite.
incorrect_inertia = correct_inertia * 10.0

print("\n--- Physically consistent model ---")
pprint(compare_inertia(mesh_path, mass, correct_inertia))

print("\n--- Physically inconsistent model ---")
pprint(compare_inertia(mesh_path, mass, incorrect_inertia))
