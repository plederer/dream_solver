from main import *
import sys
sys.path.append('.')


if __name__ == '__main__':

    for Qform in [True, False]:
        farfield_boundary(Qform)

    for Ut in [True, False]:
        for Qform in [True, False]:
            for sigma in [1, cfg.Mach_number.Get(), 1e-3]:
                for glue in [True, False]:
                    for pressure_relaxation in [True, False]:
                        gfarfield_boundary(sigma, Ut, pressure_relaxation, glue, Qform)
