from main import *
import sys
sys.path.append('.')


if __name__ == '__main__':    

    outflow_boundary()

    for Qform in [True, False]:
        farfield_boundary(Qform)

    for glue in [True, False]:
        yoo_boundary(glue)

    for glue in [True, False]:
        for sigma in [State(4, pressure=0.28, temperature=4),
                    State(4, pressure=1e-3, temperature=4),
                    State(4, pressure=5, temperature=4)]:
            poinsot_boundary(sigma, glue)