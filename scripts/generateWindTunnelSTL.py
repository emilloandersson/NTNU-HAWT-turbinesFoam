import numpy as np

# -----------------------------
# Input geometry
# -----------------------------
domainHeight_X = np.array([0.000, 1.281, 5.621, 8.435, 11.150])
domainHeight_Z = np.array([1.801, 1.801, 1.813, 1.842, 1.851])

half_width = 1.355
hubHeight = 0.817

# Apply vertical offset so turbine origin is at z = 0
floor_z = -hubHeight
corrected_ceiling_Z = domainHeight_Z - hubHeight

verts = []
for x, zc in zip(domainHeight_X, corrected_ceiling_Z):
    verts.append([
        [x, -half_width, floor_z],   # bottom-left
        [x,  half_width, floor_z],   # bottom-right
        [x, -half_width, zc],        # top-left
        [x,  half_width, zc],        # top-right
    ])
verts = np.array(verts)
N = verts.shape[0]

# Helper: quad → two triangles
def quad(a, b, c, d):
    return [
        (a, b, c),
        (c, b, d)
    ]

# Collect triangles per region
inlet_tris = []
outlet_tris = []
bottom_tris = []
top_tris = []
walls_tris = []

# Inlet (x = 0)
a, b, c, d = verts[0]
inlet_tris += quad(a, b, c, d)

# Outlet (x = end)
a, b, c, d = verts[-1]
outlet_tris += quad(a, b, c, d)

# Bottom (floor)
for i in range(N - 1):
    a = verts[i, 0]
    b = verts[i, 1]
    c = verts[i+1, 0]
    d = verts[i+1, 1]
    bottom_tris += quad(a, b, c, d)

# Top (ceiling)
for i in range(N - 1):
    a = verts[i, 2]
    b = verts[i, 3]
    c = verts[i+1, 2]
    d = verts[i+1, 3]
    top_tris += quad(a, b, c, d)

# Walls (left + right)
for i in range(N - 1):
    # left wall
    a = verts[i, 0]
    b = verts[i, 2]
    c = verts[i+1, 0]
    d = verts[i+1, 2]
    walls_tris += quad(a, b, c, d)

    # right wall
    a = verts[i, 1]
    b = verts[i, 3]
    c = verts[i+1, 1]
    d = verts[i+1, 3]
    walls_tris += quad(a, b, c, d)

# -----------------------------
# Write ASCII STL with regions
# -----------------------------
def write_region(f, name, tris):
    f.write(f"solid {name}\n")
    for tri in tris:
        a, b, c = tri
        # normal is dummy; snappyHexMesh recomputes it
        f.write("  facet normal 0 0 0\n")
        f.write("    outer loop\n")
        f.write(f"      vertex {a[0]} {a[1]} {a[2]}\n")
        f.write(f"      vertex {b[0]} {b[1]} {b[2]}\n")
        f.write(f"      vertex {c[0]} {c[1]} {c[2]}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
    f.write(f"endsolid {name}\n\n")

with open("wind_tunnel.stl", "w") as f:
    write_region(f, "inlet", inlet_tris)
    write_region(f, "outlet", outlet_tris)
    write_region(f, "bottom", bottom_tris)
    write_region(f, "top", top_tris)
    write_region(f, "walls", walls_tris)

print("Saved wind_tunnel.stl with regions: inlet, outlet, bottom, top, walls")
