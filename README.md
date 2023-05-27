# EDIRN

We provide the Python 3.7 scripts used to reproduce the results in the main text. 
They are written and tested on a computer with Intel i7-13700KF (CPU), RTX 3060ti (GPU) and 32 GB (RAM). 
No particular hardware is required to run them.
However, depending on the CPU performance, the code may take several days to run for large time steps. 

# Software Requirements

1. Windows 11
2. Python 3.7

# Python Dependencies

numpy (1.24.3)
networkx (2.8.4)
pandas (1.5.3)

# Simulation Guide
With evolution_of_reciprocity.py, we simulate the evolution of direct and indirect reciprocity on networks and calculate the fixation probability of a cooperator (Fig. 2 and Extended Fig. 1).

With calculate_bc_ratio.py, we calculate the critical benefit-to-cost ratios of one million random networks (Fig. 3).

With bc_ratio_dc_error.py, we reproduce the results in Fig. 4.
