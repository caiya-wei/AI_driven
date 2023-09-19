# EDIRN

We provide the Python 3.7 scripts used to reproduce the results in the main text. 
They are written and tested on a computer with Intel i7-13700KF (CPU), RTX 3060ti (GPU) and 32 GB (RAM). 
No particular hardware is required to run them.
However, depending on the CPU performance, the code may take several days to run for large time steps. 

# Software Requirements

1. Windows 11
2. Python 3.7

# Python Dependencies

numpy 

networkx

pandas 

scipy


# Simulation Guide
With bc_ratio_random_networks.py, we calculate the critical benefit-to-cost ratios of random networks, reproducing the results of Fig. 2 and Fig. 3.

With fixation_probability.py, we calculate the fixation probability of one of such reciprocity strategies invading N - 1 defectors, which reproduces the result of Fig. 4.

With bc_ratio_empirical_network.py, we calculate the critical benefit-to-cost ratios of six empirical networks, which reproduces the result of Fig. 5.
