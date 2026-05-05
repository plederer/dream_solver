import argparse
from dream.io import IOConfiguration
from pathlib import Path
import numpy as np
from config import (get_single_mesh, get_twosided_geometrical_coordinates)

parser = argparse.ArgumentParser(description='Export meshes')
parser.add_argument('--Nr', type=int, help='Number of radial elements', default=64)
parser.add_argument('--Ni', type=int, help='Number of implicit elements', default=4)
parser.add_argument('--Nphi', type=int, help='Number of radial elements', default=32)

parser.add_argument('--dri0', type=float, help='Initial radial spacing for implicit region', default=0.001)
parser.add_argument('--dre0', type=float, help='Initial radial spacing for explicit region', default=0.05)
parser.add_argument('--dphi0', type=float, help='Initial circumferential spacing', default=0.0)
parser.add_argument('--Ro', type=float, help='Outer radius', default=100.0)
parser.add_argument('--Ri', type=float, help='Inner radius', default=0.5)
parser.add_argument('--path', type=str, help='Path for the mesh export', default='default')

USER = vars(parser.parse_args())
Nr = USER['Nr']
Ni = USER['Ni']
Nphi = USER['Nphi']
dri0 = USER['dri0']
dre0 = USER['dre0']
dphi0 = USER['dphi0']
Ro = USER['Ro']
Ri = USER['Ri']
path = USER['path']

if np.allclose(dri0, 0.0):
    dri0 = None

if np.allclose(dre0, 0.0):
    dre0 = None

if np.allclose(dphi0, 0.0):
    dphi0 = None

filename = "mesh"
foldername = f"{Nr}x{Ni}x{Nphi}"

if dre0 is not None:
    foldername += f"_dre{dre0}"

if dphi0 is not None:
    foldername += f"_dphi{dphi0}"

if path == 'default':
    path = Path.cwd().joinpath(foldername + f"/dri{dri0}")


if dphi0 is None:
    r, phi = get_twosided_geometrical_coordinates(Nr, Ni, Nphi, dri0, dre0, dphi0=None, Ro=Ro, Ri=Ri)
else:
    r, phi = get_twosided_geometrical_coordinates(Nr, Ni, Nphi, dri0, dre0, dphi0=np.pi*dphi0, Ro=Ro, Ri=Ri)
mesh = get_single_mesh(r, phi)

io = IOConfiguration(None)
io.path = path
io.path.mkdir(parents=True, exist_ok=True)
saver = io.ngsmesh

# Export main mesh
saver.mesh = mesh
saver.filename = filename
saver.save_pre_time_routine()
