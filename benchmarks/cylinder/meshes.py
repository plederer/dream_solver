from config import (get_geometrical_coordinates, get_single_mesh)
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

parser.add_argument('--dr0', type=float, help='Initial radial spacing', default=0.04)
parser.add_argument('--dphi0', type=float, help='Initial circumferential spacing', default=0.0)
parser.add_argument('--Ro', type=float, help='Outer radius', default=50.0)
parser.add_argument('--Ri', type=float, help='Inner radius', default=0.5)
parser.add_argument('--path', type=str, help='Path for the mesh export', default='default')
parser.add_argument('--curve', action=argparse.BooleanOptionalAction, default=False)

USER = vars(parser.parse_args())
Nr = USER['Nr']
Nphi = USER['Nphi']
dr0 = USER['dr0']
dphi0 = USER['dphi0']
Ro = USER['Ro']
Ri = USER['Ri']
path = USER['path']
curve = USER['curve']

if np.allclose(dr0, 0.0):
    dr0 = None

if np.allclose(dphi0, 0.0):
    dphi0 = None

filename = "mesh"
foldername = f"{Nr}x{Nphi}"

if dr0 is not None:
    foldername += f"_dr{dr0}"

if dphi0 is not None:
    foldername += f"_dphi{dphi0}"

if curve:
    foldername += "_curved"

if path == 'default':
    path = Path.cwd().joinpath(foldername)

if USER['mesh'] == 'geometrical':
    r, phi = get_geometrical_coordinates(Nr=Nr, Nphi=Nphi, dr0=dr0, dphi0=np.pi*dphi0, Ro=Ro, Ri=Ri)
mesh = get_single_mesh(r, phi, curve_all=curve)

io = IOConfiguration(None)
io.path = path
io.path.mkdir(parents=True, exist_ok=True)
saver = io.ngsmesh

# Export main mesh
saver.mesh = mesh
saver.filename = filename
saver.save_pre_time_routine()
