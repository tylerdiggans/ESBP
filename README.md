# ESBP
Code for finding and approximating solutions to the Essential Synchronization Backbone Problem of Networked Oscillators

This is companion code for the paper: 
"The Essential Synchronization Backbone Problem" (to be published soon)

There is an exhaustive search algorithm for finding an ESB of a system of osciallators with linear diffusive coupling on small graphs, and two algorithms for approximating an ESB, meaning finding a small (not necessarily minimal) volume spanning subgraph whose synchronization manifold has essentially equivalent stability to the system on the original network. We call these approximations Greedy-ESB or GESB.

The HybridGESB.py was the first developed algorithm, but includes an exhasutive search of all possible greedy choices at each iteration, and so can take a long time for larger graphs. As such, we only used this for some comparisons later in the paper.

The GreedyESB.py is an efficient method of finding a GESB of a linearly diffusive coupled synchronizing oscillator system. It uses the heuristic described in Habgerg 2008 to make the greedy choice based on the valuation of the eigenvector corresponding to the largest eigenvalue of the graph Laplacian.

Included in this repository are codes that can be used to create adjacency matrix files using the networkx package in python and some plotting functions for synchronization of particular initial conditions as well as visualizations of how the GESB is obtained in rotating GIFs of the backbone emerging from the original graph.

Please direct any questions about this (within reason) to the main author at the correspondence address on the published version.


