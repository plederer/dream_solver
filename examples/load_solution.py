from ngsolve import *
import pickle

import os
import time 
import sys

sys.path.insert(1, '../code')
sys.path.insert(1, '../utils')

from HDGSolver import compressibleHDGsolver


example_data = "visc_cylinder_data"
files_path = os.path.join(os.path.abspath(os.getcwd()), example_data)

if not os.path.exists(files_path):
    raise Exception("Example data does not exist")

picklefile = open(os.path.join(files_path, "config_file"), "rb")
bnd_data, ff_data, order = pickle.load(picklefile)
picklefile.close()

picklefile = open(os.path.join(files_path, "gfu_file"), "rb")
gfu = pickle.load(picklefile)
picklefile.close()
mesh = gfu.space.mesh
mesh.Curve(order)


hdgsolver = compressibleHDGsolver(mesh, 
                                  order=order,
                                  ff_data=ff_data,
                                  bnd_data=bnd_data)


# hdgsolver.gfu = gfu
hdgsolver.SetDirName(example_data)
hdgsolver.SetUp()
hdgsolver.LoadState(0)
hdgsolver.DrawSolutions()


tend = 500 
s = 0
t = 0

# For time loops:

# with TaskManager():
#     while t < tend:
#         t += hdgsolver.FU.dt.Get()
#         s += 1
#         print("step = {}, time = {}".format(s, t), end='\r')
#         if s % 5 == 0:
#             hdgsolver.LoadState(s)
#             Redraw(blocking=True)
#             time.sleep(0.1)
