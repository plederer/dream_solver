from config import (get_geometrical_coordinates, get_connected_mesh, get_imex_meshes)
import argparse
from dream.io import IOConfiguration
from pathlib import Path
import numpy as np

mesh = {
    'geometrical': get_geometrical_coordinates,
}

parser = argparse.ArgumentParser(description='Export meshes')
parser.add_argument('mesh', type=str, help='Mesh type', choices=list(mesh))
parser.add_argument('--Nr', type=int, help='Number of radial elements', default=32)
parser.add_argument('--Nphi', type=int, help='Number of radial elements', default=32)

parser.add_argument('--Ni', type=int, nargs="+", help='Time steps', default=(16,))
parser.add_argument('--wake', type=float, nargs="+", help='Additional phis in the wake', default=[])
parser.add_argument('--dr0', type=float, help='Initial radial spacing', default=0.04)
parser.add_argument('--Ro', type=float, help='Outer radius', default=50.0)
parser.add_argument('--Ri', type=float, help='Inner radius', default=0.5)
parser.add_argument('--path', type=str, help='Path for the mesh export', default='default')
parser.add_argument('--curve', action=argparse.BooleanOptionalAction, default=False)

USER = vars(parser.parse_args())
Nr = USER['Nr']
Nphi = USER['Nphi']
Ni = USER['Ni']
dr0 = USER['dr0']
Ro = USER['Ro']
Ri = USER['Ri']
path = USER['path']
curve = USER['curve']
wake = np.deg2rad(np.array(USER['wake']))

filename = "gmesh"

if curve:
    filename += "_curved"

if wake.size > 0:
    filename += "_wake"

if path == 'default':
    path = Path.cwd().joinpath(f"{Nr}x{Nphi}_drmin{dr0}", "meshes")

if USER['mesh'] == 'geometrical':
    ri, re, phi = get_geometrical_coordinates(Nr=Nr, Nphi=Nphi, Ni=Ni[0], dr0=dr0, Ro=Ro, Ri=Ri, wake=wake)

mesh = get_connected_mesh(ri, re, phi, curve_all=curve)

io = IOConfiguration(None)
io.path = path
io.path.mkdir(parents=True, exist_ok=True)
saver = io.ngsmesh

# Export main mesh
saver.mesh = mesh
saver.filename = filename
saver.save_pre_time_routine()

# Export IMEX meshes
for ni in Ni:

    if USER['mesh'] == 'geometrical':
        ri, re, phi = get_geometrical_coordinates(Nr=Nr, Nphi=Nphi, Ni=ni, dr0=dr0, Ro=Ro, Ri=Ri, wake=wake)
    implicit_mesh, explicit_mesh = get_imex_meshes(ri, re, phi, curve_all=curve)

    saver.mesh = implicit_mesh
    saver.filename = f"{filename}_{ni}_implicit"
    saver.save_pre_time_routine()

    saver.mesh = explicit_mesh
    saver.filename = f"{filename}_{ni}_explicit"
    saver.save_pre_time_routine()
