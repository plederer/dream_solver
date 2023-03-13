import sys
sys.path.append('')
from configuration import *

solver = angular_sponge_setup()
#time_advancing_routine(solver, False, False, True, time_periods=(0, 500, 600), fine_time_step=0.1, draw=True)
saving_routine(solver, (600, 700), False,  draw=True)

