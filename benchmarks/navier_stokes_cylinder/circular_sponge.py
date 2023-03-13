import sys
sys.path.append('')
from configuration import *

solver = circular_sponge_setup()
#time_advancing_routine(solver, True, True, True, time_periods=(0, 500, 600), fine_time_step=0.1, draw=True)
saving_routine(solver, (600, 700), False,  draw=True)