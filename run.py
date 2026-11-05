#!/usr/bin/env python
"""Script for running the NTNU HAWT case."""

import argparse
import os
import subprocess
from subprocess import call, check_output
import numpy as np
import pandas as pd
import glob
import foampy
from foampy.dictionaries import replace_value
import shutil
from pynhtf import processing as pr
import time

# These vertices must fully enclose wind_tunnel.stl
VERTICES_STL = [
    (11.5, -1.5, -0.9),
    (11.5,  1.5, -0.9),
    (-0.5,  1.5, -0.9),
    (-0.5, -1.5, -0.9),
    (11.5, -1.5,  1.1),
    (11.5,  1.5,  1.1),
    (-0.5,  1.5,  1.1),
    (-0.5, -1.5,  1.1)
]

# These vertices should be as close to the actual dimensions as possible
VERTICES_BLOCKMESH = [
    (11.150, -1.355, -0.817),
    (11.150,  1.355, -0.817),
    (0.000,   1.355, -0.817),
    (0.000,  -1.355, -0.817),
    (11.150, -1.355,  1.083),
    (11.150,  1.355,  1.083),
    (0.000,   1.355,  1.083),
    (0.000,  -1.355,  1.083)
]

GEOM_STL = """
    windTunnel
    {
        type triSurfaceMesh;
        file "wind_tunnel.stl";

        regions
        {
            inlet  { name inlet; }
            outlet { name outlet; }
            walls  { name walls; }
            top    { name top; }
            bottom { name bottom; }
        }
    }
"""

GEOM_BLOCKMESH = """
    // No STL geometry used
    // Only refinement boxes are active
"""

REFINE_STL = """
    windTunnel
    {
        level (1 2);
        regions
        {
            inlet  { level (1 2); patchInfo { type patch; } }
            outlet { level (1 2); patchInfo { type patch; } }
            walls  { level (1 2); patchInfo { type wall; } }
            top    { level (1 2); patchInfo { type wall; } }
            bottom { level (1 2); patchInfo { type wall; } }
        }
    }
"""

REFINE_NONE = """
    // No STL refinement surfaces
"""

FEATURES_STL = """
    {
        file "wind_tunnel.eMesh";
        level 0;
    }
"""

FEATURES_NONE = """
    // No STL feature edges
"""


def get_mesh_dims():
    """Get mesh dimensions by grepping `blockMeshDict`."""
    raw = check_output("grep blocks system/blockMeshDict -A3",
                       shell=True).decode().split("\n")[3]
    raw = raw.replace("(", "").replace(")", "").split()
    # block ordering from vertex ordering
    # (the vertex ordering should be fixed)
    return {"nx": int(raw[1]), "ny": int(raw[0]), "nz": int(raw[2])}


def get_dt():
    """Read ``deltaT`` from ``controlDict``."""
    return foampy.dictionaries.read_single_line_value("controlDict",
                                                      keyword="deltaT")


def get_solver():
    """Return the solver specified in controlDict."""
    return foampy.dictionaries.read_single_line_value(
        "controlDict",
        keyword="application",
        dtype=str
    )


def get_final_iteration(logfile):
    with open(logfile) as f:
        for line in reversed(f.readlines()):
            if line.startswith("Time ="):
                return line.split("=")[1].strip()

    raise RuntimeError(f"No time found in {logfile}")


def get_nacelle_ano_vals():
    """Lookup sampled nacelle anemometer values."""
    res = {}
    df = pr.load_nacelle_sets()
    for idx, row in df.iterrows():
        res[f"vel_mag_{idx}"] = row.vel_mag
        res[f"vel_dir_{idx}"] = row.vel_dir
    return res


def log_results(param="turbine1_yaw", append=True, verbose=True):
    """Log results to CSV file."""
    if not os.path.isdir("processed"):
        os.mkdir("processed")
    fpath = "processed/{}_sweep.csv".format(param)
    if append and os.path.isfile(fpath):
        df = pd.read_csv(fpath)
    else:
        df = pd.DataFrame()
    solver = get_solver()
    avg_revs = 5 # average over last 5 iterations for pimpleFoam
    d = pr.calc_perf(avg_revs=avg_revs, solver=solver)
    d.update(get_mesh_dims())
    d["dt"] = get_dt()
    d["yaw"] = foampy.read_single_line_value(dictpath="system/fvOptions",
                                             keyword="yawAngle")
    # Add nacelle anemometer params and results
    d.update(get_nacelle_ano_vals())
    if verbose:
        print("Logging results:")
        for k, v in d.items():
            print(f"    {k}: {v}")
    # df = df.append(d, ignore_index=True) # OLD (broken in pandas 2.x)
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True) # NEW 
    df.to_csv(fpath, index=False)


def get_vertices(use_stl=False):
    """Return correct blockMesh vertices depending on STL usage,
    including vertex index comments."""
    verts = VERTICES_STL if use_stl else VERTICES_BLOCKMESH
    txt = ""
    for i, v in enumerate(verts):
        txt += f"    ({v[0]} {v[1]} {v[2]}) // {i}\n"
    return txt


def set_blockmesh(nx=96, ny=None, nz=None, use_stl=False):
    """Set mesh resolution in ``blockMeshDict``.

    If only ``nx`` is provided, the default resolutions for other dimensions are
    scaled proportionally.
    """
    defaults = {"nx": 96, "ny": 32, "nz": 24}
    if ny is None:
        ny = int(nx*defaults["ny"]/defaults["nx"])
    if nz is None:
        nz = int(nx*defaults["nz"]/defaults["nx"])

    print(f"Setting blockMesh resolution to ({nx} {ny} {nz}), use_stl={use_stl}")

    vertices_txt = get_vertices(use_stl)

    foampy.fill_template(
        "system/blockMeshDict.template",
        nx=nx, ny=ny, nz=nz,
        vertices=vertices_txt
    )


def set_dt(dt=0.01, tsr=None, tsr_0=6.0, write_interval=None, les=False):
    """Set ``deltaT`` in ``controlDict``. Will scale proportionally if ``tsr``
    and ``tsr_0`` are supplied, such that steps-per-rev is consistent with
    ``tsr_0``.
    """
    if tsr is not None:
        dt = dt*tsr_0/tsr
        print("Setting deltaT = dt*tsr_0/tsr = {:.3f}".format(dt))
    dt = str(dt)
    if write_interval is None:
        if les:
            write_interval = 0.01
        else:
            write_interval = 0.05
    foampy.fill_template("system/controlDict.template", dt=dt,
                         write_interval=write_interval)

def set_nproc(nproc=1):
    foampy.fill_template("system/decomposeParDict.template",
                         nproc=nproc)


def gen_sets_file(origin=(0.1, 0.0, 0.04), step=0.01, yaw=None):
    """Generate ``sets`` file for post-processing nacelle anemometer
    locations.
    """
    if yaw is None:
        yaw = foampy.read_single_line_value(dictpath="system/fvOptions",
                                            keyword="yawAngle")
    points_txt = ""
    points = [list(origin)]
    for s in [-step, step]:
        p = list(origin)
        p[1] += s
        points.append(p)
    # Rotate all points about origin by ``yaw`` degrees
    for p in points:
        x, y = pr.rotate_vector((p[0], p[1]), np.radians(yaw))
        p[0] = x
        p[1] = y
    for p in points:
        points_txt += f"            ({p[0]} {p[1]} {p[2]})\n"
    foampy.fill_template("system/sets.template", points=points_txt)


def post_process(parallel=False, tee=False, reconstruct=False, overwrite=True):
    """Execute all post-processing."""
    gen_sets_file()
    foampy.run("postProcess", args="-func -vorticity", parallel=parallel,
               logname="log.vorticity", tee=tee, overwrite=overwrite)
    foampy.run("postProcess", args="-dict system/controlDict.recovery "
               " -latestTime", parallel=parallel, logname="log.recovery",
               tee=tee, overwrite=overwrite)
    # Reconstruct if necessary so sampling isn't run in parallel
    if reconstruct:
        foampy.run("reconstructPar", args="-latestTime", overwrite=overwrite,
                   logname="log.reconstructPar-latestTime", tee=tee)
    foampy.run("postProcess", args="-func sets -latestTime",
               logname="log.sample", parallel=False, overwrite=overwrite,
               tee=tee)


def param_sweep(param="turbine1_yaw", start=-20, stop=21, step=5,
                dtype=float, append=False, parallel=True, tee=False,
                store_postProcessing=False, **kwargs):
    """Run multiple simulations, varying ``param``.

    ``stop`` is not included.
    """
    sweep_start = time.time()
    print("Running {} sweep".format(param))
    fpath = "processed/{}_sweep.csv".format(param)
    if not append:
        if os.path.isfile(fpath):
            os.remove(fpath)
        subprocess.call(f"rm processed/log.*", shell=True)
        subprocess.call(f"rm -rf processed/postProcessing.*", shell=True)
        subprocess.call(f"rm -rf processed/fields/*", shell=True)
    if param == "nx":
        dtype = int
    if store_postProcessing:
        os.makedirs("processed/fields", exist_ok=True)
    param_list = np.arange(start, stop, step, dtype=dtype)
    for p in param_list:
        print("Running with {} = {}".format(param, p))
        if p == param_list[0] or param in ("nx", "ny", "nz"):
            foampy.clean(remove_zero=True)
            mesh = True
        else:
            mesh = False
        # Update kwargs for this value
        kwargs.update({param: p})
        run(parallel=parallel, tee=tee, mesh=mesh, reconstruct=False,
            post=False, overwrite=True, **kwargs)
        solver = get_solver()
        os.rename(f"log.{solver}", f"log.{solver}.{p}")
        if store_postProcessing:
            subprocess.call(f"cp log.{solver}.{p} processed/", shell=True)
            subprocess.call(f"cp -r postProcessing processed/postProcessing.{p}", shell=True)
            print(f"Saved log and postProcessing for parameter {p} in processed/")
            # reconstruct the final field and store w.r.t. parameter name
            convergenceTime = get_final_iteration(f"log.{solver}.{p}")
            subprocess.call(
                f"reconstructPar -time {convergenceTime} >> log.sweepReconstruct 2>&1",
                shell=True
            )
            if not os.path.isdir(convergenceTime):
                raise RuntimeError(
                    f"reconstructPar did not create {convergenceTime}"
                )
            stored_time = f"processed/fields/{10000 + p}"
            if os.path.isdir(stored_time):
                shutil.rmtree(stored_time)
            os.rename(convergenceTime, stored_time)
            print(f"Saved reconstructed final time {convergenceTime} as {stored_time}")
        log_results(param=param, append=True)
        foampy.clean(leave_mesh=True, remove_zero=True)
    if store_postProcessing:
        subprocess.call("cp -r constant processed/fields/constant", shell=True)
    total_elapsed = time.time() - sweep_start
    print("\n==============================")
    print(f"Completed {param} sweep in {total_elapsed/60:.2f} minutes")
    print("==============================")


def get_snappy_geometry(use_stl=False):
    return GEOM_STL if use_stl else GEOM_BLOCKMESH


def get_refinement_surfaces(use_stl=False):
    return REFINE_STL if use_stl else REFINE_NONE


def get_features_block(use_stl=False):
    return FEATURES_STL if use_stl else FEATURES_NONE


def make_refinement_boxes(turbine_x, turbine_y,
                          turbine_D, active,
                          dx1_frac=1.5, dx2_frac=5.5,
                          dy1_frac=1.5, dy2_frac=1.5,
                          dz1_frac=1.0, dz2_frac=1.0):
    """Return pmin and pmax strings for refinement box."""
    if active == "off":
        return " -1e6 -1e6 -1e6 ", " -1e6 -1e6 -1e6 "

    x1 = turbine_x - dx1_frac * turbine_D
    x2 = turbine_x + dx2_frac * turbine_D
    y1 = turbine_y - dy1_frac * turbine_D
    y2 = turbine_y + dy2_frac * turbine_D
    z1 =         0 - dz1_frac * turbine_D
    z2 =         0 + dz2_frac * turbine_D

    pmin = f" {x1} {y1} {z1} "
    pmax = f" {x2} {y2} {z2} "
    return pmin, pmax


def make_refinement_cylinders(turbine_x, turbine_y, turbine_D,
                              turbine_yaw, active,
                              dx1_frac=0.75, dx2_frac=1.0,
                              dr_frac=1.5):
    """Return p1, p2, r for refinement cylinder."""
    if active == "off":
        return " -1.1e6 -1.1e6 -1.1e6 ", " -1e6 -1e6 -1e6 ", 0.1

    yaw_rad = np.deg2rad(turbine_yaw)

    dx = np.cos(yaw_rad)
    dy = np.sin(yaw_rad)

    x1 = turbine_x - dx * dx1_frac * turbine_D
    y1 = turbine_y - dy * dx1_frac * turbine_D

    x2 = turbine_x + dx * dx2_frac * turbine_D
    y2 = turbine_y + dy * dx2_frac * turbine_D

    r  = dr_frac * 0.5 * turbine_D

    p1_str = f" {x1} {y1} 0 "
    p2_str = f" {x2} {y2} 0 "
    return p1_str, p2_str, r


def set_turbine_params(turbine1_tsr=6, turbine1_active="on",
                       turbine1_x=1.788, turbine1_y=0,
                       turbine1_D=0.944,
                       turbine2_tsr=4, turbine2_active="on",
                       turbine2_x=4.470, turbine2_y=0,
                       turbine2_D=0.894,
                       turbine1_yaw=0, turbine2_yaw=0,
                       use_stl = False, verbose=True):
    """Write file defining turbine operating parameters.

    ``tsr_phase`` is in radians.
    """
    geom_txt = get_snappy_geometry(use_stl)
    refine_txt = get_refinement_surfaces(use_stl)
    features_txt = get_features_block(use_stl)

    # Even though a turbine is inactive, topoSet should not affect the
    # simulation if it does not intersect with any cells with turbine forces
    args_topoSet = {
        "turbine1_upstream_x": turbine1_x - 0.25,
        "turbine1_downstream_x": turbine1_x + 0.25,
        "turbine1_rotor_y": turbine1_y,
        "turbine1_tower_x": turbine1_x + 0.48,
        "turbine1_tower_y": turbine1_y,
        "turbine2_upstream_x": turbine2_x - 0.25,
        "turbine2_downstream_x": turbine2_x + 0.25,
        "turbine2_rotor_y": turbine2_y,
        "turbine2_tower_x": turbine2_x + 0.14,
        "turbine2_tower_y": turbine2_y
    }
    foampy.fill_template("system/topoSetDict.template", **args_topoSet)

    t1_pmin, t1_pmax = make_refinement_boxes(
        turbine_x=turbine1_x, turbine_y=turbine1_y,
        turbine_D=turbine1_D, active=turbine1_active
    )
    t2_pmin, t2_pmax = make_refinement_boxes(
        turbine_x=turbine2_x, turbine_y=turbine2_y,
        turbine_D=turbine2_D, active=turbine2_active
    )

    t1_cyl_p1, t1_cyl_p2, t1_cyl_r = make_refinement_cylinders(
        turbine_x=turbine1_x, turbine_y=turbine1_y,
        turbine_D=turbine1_D, turbine_yaw=turbine1_yaw,
        active=turbine1_active
    )
    t2_cyl_p1, t2_cyl_p2, t2_cyl_r = make_refinement_cylinders(
        turbine_x=turbine2_x, turbine_y=turbine2_y,
        turbine_D=turbine2_D, turbine_yaw=turbine2_yaw,
        active=turbine2_active
    )


    args_snappyHexMesh = {
        "turbine1_refBox_pmin": t1_pmin,
        "turbine1_refBox_pmax": t1_pmax,
        "turbine1_refCyl_p1": t1_cyl_p1,
        "turbine1_refCyl_p2": t1_cyl_p2,
        "turbine1_refCyl_r":  t1_cyl_r,

        "turbine2_refBox_pmin": t2_pmin,
        "turbine2_refBox_pmax": t2_pmax,
        "turbine2_refCyl_p1": t2_cyl_p1,
        "turbine2_refCyl_p2": t2_cyl_p2,
        "turbine2_refCyl_r":  t2_cyl_r,

        "geometry_block": geom_txt,
        "refinement_surfaces": refine_txt,
        "features_block": features_txt
    }
    foampy.fill_template("system/snappyHexMeshDict.template", **args_snappyHexMesh)

    params = {
        "turbine1_tsr": turbine1_tsr,
        "turbine1_active": turbine1_active,
        "turbine1_x": turbine1_x,
        "turbine1_y": turbine1_y,
        "turbine1_yaw": turbine1_yaw,
        "turbine2_tsr": turbine2_tsr,
        "turbine2_active": turbine2_active,
        "turbine2_x": turbine2_x,
        "turbine2_y": turbine2_y,
        "turbine2_yaw": turbine2_yaw
    }
    if verbose:
        print("Setting turbine params as:")
        for k, v in params.items():
            print("    " + k + ":", v)
    foampy.fill_template("system/fvOptions.template", **params)


def run(turbine1_tsr=6, turbine1_active="on",
        turbine1_x=1.788, turbine1_y=0, turbine1_D=0.944,
        turbine2_tsr=4, turbine2_active="on",
        turbine2_x=4.470, turbine2_y=0, turbine2_D=0.894,
        turbine1_yaw=0, turbine2_yaw=0, nx=96, ny=None, nz=None,
        mesh=True, parallel=False, tee=False, reconstruct=True,
        overwrite=False, post=False, write_interval=None, nproc=1, use_stl=False):
    """Run simulation once."""

    start_time = time.time()
    if nproc == 1:
        parallel = False
    set_turbine_params(turbine1_tsr=turbine1_tsr,
                       turbine1_active=turbine1_active,
                       turbine1_x=turbine1_x,
                       turbine1_y=turbine1_y,
                       turbine1_D=turbine1_D,
                       turbine2_tsr=turbine2_tsr,
                       turbine2_active=turbine2_active,
                       turbine2_x=turbine2_x,
                       turbine2_y=turbine2_y,
                       turbine2_D=turbine2_D,
                       turbine1_yaw=turbine1_yaw,
                       turbine2_yaw=turbine2_yaw,
                       use_stl=use_stl)
    if mesh:
        set_blockmesh(nx=nx, ny=ny, nz=nz, use_stl=use_stl)
        foampy.run("blockMesh", tee=tee)
    # Copy over initial conditions
    subprocess.call("cp -rf 0.orig 0 > /dev/null 2>&1", shell=True)
    if parallel:
        set_nproc(nproc)
    if parallel and not glob.glob("processor*"):
        foampy.run("decomposePar", tee=tee, overwrite=overwrite)
        subprocess.call("for PROC in processor*; do cp -rf 0.orig/* $PROC/0; "
                        " done", shell=True)
    if mesh:
        foampy.run("snappyHexMesh", args="-overwrite", tee=tee,
                   parallel=parallel)
        foampy.run("topoSet", parallel=parallel, tee=tee)
        if parallel:
            foampy.run("reconstructParMesh", args="-constant -time 0", tee=tee)
    solver = get_solver()
    foampy.run(solver, parallel=parallel, tee=tee, overwrite=overwrite)
    # Sample nacelle values
    gen_sets_file()
    foampy.run("postProcess", args="-func sets -latestTime",
               logname="log.sample", parallel=parallel, overwrite=overwrite,
               tee=tee)
    if parallel and reconstruct:
        foampy.run("reconstructPar", tee=tee, overwrite=overwrite)
    if post:
        post_process(overwrite=overwrite, parallel=parallel,
                     reconstruct=not reconstruct)
    elapsed = time.time() - start_time
    print(f"Run completed in {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NTNU HAWT ALM case")
    parser.add_argument("--turbine1-active", default="on")
    parser.add_argument("--turbine1-x", default=1.788, type=float) # Default: 2*D2
    parser.add_argument("--turbine1-y", default=0, type=float)
    parser.add_argument("--turbine1-D", default=0.944, type=float) # Default: D1=0.944
    parser.add_argument("--turbine1-tsr", default=6.0, type=float)
    parser.add_argument("--turbine1-yaw", default=0.0, type=float,
                        help="Yaw angle in degrees")
    parser.add_argument("--turbine2-active", default="on")
    parser.add_argument("--turbine2-x", default=4.470, type=float) # Default: (2+3)*D2
    parser.add_argument("--turbine2-y", default=0, type=float)
    parser.add_argument("--turbine2-D", default=0.894, type=float) # Default: D2=0.894
    parser.add_argument("--turbine2-tsr", default=4.0, type=float)
    parser.add_argument("--turbine2-yaw", default=0.0, type=float,
                        help="Yaw angle in degrees")
    parser.add_argument("--leave-mesh", "-l", default=False,
                        action="store_true", help="Leave existing mesh")
    parser.add_argument("--nx", type=int, default=96,
                    help="Number of cells in x-direction for blockMesh")
    parser.add_argument("--ny", type=int, default=None,
                    help="Number of cells in y-direction for blockMesh")
    parser.add_argument("--nz", type=int, default=None,
                    help="Number of cells in z-direction for blockMesh")
    parser.add_argument("--no-reconstruct", default=False, action="store_true",
                        help="Do not reconstruct mesh and fields")
    parser.add_argument("--post", "-P", default=False, action="store_true",
                        help="Run post-processing (done by default at end of "
                             " run)")
    parser.add_argument("--param-sweep", "-p",
                        help="Run multiple simulations varying a parameter",
                        choices=["turbine1_tsr", "turbine2_tsr",
                                 "turbine1_yaw", "turbine2_yaw"])
    parser.add_argument("--start", default=-30, type=float)
    parser.add_argument("--stop", default=31, type=float)
    parser.add_argument("--step", default=5, type=float)
    parser.add_argument("--serial", "-S", default=False, action="store_true")
    parser.add_argument("--append", "-a", default=False, action="store_true")
    parser.add_argument("--tee", "-T", default=False, action="store_true",
                        help="Print log files to terminal while running")
    parser.add_argument("--overwrite", "-f", default=False, action="store_true",
                        help="Clean case automatically before running")
    parser.add_argument("--nproc", "-n", default=2, type=int,
                    help="Number of subdomains for decomposePar")
    parser.add_argument("--use-stl", default=False, action="store_true",
                    help="Use STL wind tunnel geometry instead of blockMesh-only domain")
    parser.add_argument("--store-postProcessing", default=False, action="store_true",
                    help="Store the postProcessing folder for each parameter sweep step")

    args = parser.parse_args()

    if args.param_sweep:
        param_sweep(args.param_sweep, args.start, args.stop, args.step,
                    append=args.append,
                    parallel=not args.serial,
                    tee=args.tee,
                    turbine1_active=args.turbine1_active,
                    turbine1_tsr=args.turbine1_tsr,
                    turbine1_x=args.turbine1_x,
                    turbine1_y=args.turbine1_y,
                    turbine1_D=args.turbine1_D,
                    turbine1_yaw=args.turbine1_yaw,
                    turbine2_active=args.turbine2_active,
                    turbine2_tsr=args.turbine2_tsr,
                    turbine2_x=args.turbine2_x,
                    turbine2_y=args.turbine2_y,
                    turbine2_D=args.turbine2_D,
                    turbine2_yaw=args.turbine2_yaw,
                    nx=args.nx,
                    ny=args.ny,
                    nz=args.nz,
                    nproc=args.nproc,
                    use_stl=args.use_stl,
                    store_postProcessing=args.store_postProcessing)
    elif not args.post:
        run(turbine1_active=args.turbine1_active,
            turbine1_tsr=args.turbine1_tsr,
            turbine1_x=args.turbine1_x,
            turbine1_y=args.turbine1_y,
            turbine1_D=args.turbine1_D,
            turbine1_yaw=args.turbine1_yaw,
            turbine2_active=args.turbine2_active,
            turbine2_tsr=args.turbine2_tsr,
            turbine2_x=args.turbine2_x,
            turbine2_y=args.turbine2_y,
            turbine2_D=args.turbine2_D,
            turbine2_yaw=args.turbine2_yaw,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            reconstruct=not args.no_reconstruct,
            nproc=args.nproc,
            parallel=not args.serial,
            tee=args.tee,
            mesh=not args.leave_mesh,
            overwrite=args.overwrite,
            use_stl=args.use_stl)
    if args.post:
        post_process(parallel=not args.serial, tee=args.tee,
                     overwrite=args.overwrite)
