import numpy as np

def write_ceiling_stl(filename, x_points, h_points, y_min=-1.35, y_max=1.35, n_samples=200):
    """
    Generate a smooth ceiling STL by linear interpolation of height(x).

    Parameters:
        filename (str): Output STL file name.
        x_points (list): List of x-coordinates defining the ceiling profile.
        h_points (list): List of ceiling heights corresponding to x_points.
        y_min (float): Minimum y coordinate (left wall).
        y_max (float): Maximum y coordinate (right wall).
        n_samples (int): Number of interpolated points along x.
    """

    # Interpolate ceiling height
    xs = np.linspace(min(x_points), max(x_points), n_samples)
    hs = np.interp(xs, x_points, h_points)

    # Build STL triangles
    # Two strips: one at y_min, one at y_max
    z_offset = 0.817  # blockMesh bottom offset
    strip_low  = np.column_stack([xs, np.full_like(xs, y_min), hs - z_offset])
    strip_high = np.column_stack([xs, np.full_like(xs, y_max), hs - z_offset])


    def facet(n, v1, v2, v3):
        """Return STL facet string."""
        return (
            f"  facet normal {n[0]} {n[1]} {n[2]}\n"
            "    outer loop\n"
            f"      vertex {v1[0]} {v1[1]} {v1[2]}\n"
            f"      vertex {v2[0]} {v2[1]} {v2[2]}\n"
            f"      vertex {v3[0]} {v3[1]} {v3[2]}\n"
            "    endloop\n"
            "  endfacet\n"
        )

    with open(filename, "w") as f:
        f.write("solid ceiling\n")

        # Create triangles between strips
        for i in range(n_samples - 1):
            # Quad split into two triangles
            v1 = strip_low[i]
            v2 = strip_low[i+1]
            v3 = strip_high[i]
            v4 = strip_high[i+1]

            # Triangle 1: v1, v2, v3
            n1 = np.cross(v2 - v1, v3 - v1)
            f.write(facet(n1, v1, v2, v3))

            # Triangle 2: v2, v4, v3
            n2 = np.cross(v4 - v2, v3 - v2)
            f.write(facet(n2, v2, v4, v3))

        f.write("endsolid ceiling\n")


# Example usage:
x_points = [0.000, 1.281, 5.621, 8.435, 11.150]
h_points = [1.801, 1.801, 1.813, 1.842, 1.851]

write_ceiling_stl("ceiling.stl", x_points, h_points)
