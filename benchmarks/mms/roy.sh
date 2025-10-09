#!/bin/sh

for Ma in 0.1 0.3 0.5 0.7; do
    for Re in 1 10 100 1000; do
        for riemann_solver in upwind lax_friedrich; do
            python3 roy.py $Ma $Re $riemann_solver hdg

            for IP in 1 5 10 20; do
                python3 roy.py $Ma $Re $riemann_solver dg --IP=$IP
            done
        done
    done
done

# Test case
# for Ma in 0.1; do
#     for Re in 1; do
#         for riemann_solver in upwind lax_friedrich; do
#             python3 roy.py $Ma $Re $riemann_solver hdg

#             for IP in 1; do
#                 python3 roy.py $Ma $Re $riemann_solver dg --IP=$IP
#             done
#         done
#     done
# done

