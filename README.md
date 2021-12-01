# incompressible-ns
This repository contains a C++ implementation of a 2D incompressible Navier Stokes solver. Please see associated report for more detail.
The videos mentioned in the report are displayed below:

# Kelvin-Helmholtz instability

https://user-images.githubusercontent.com/32913492/144205914-26fe5083-e961-43cb-8824-176f29f94b66.mp4

# Instability of periodic array of vortices

https://user-images.githubusercontent.com/32913492/144206584-6f3f8cd1-9d74-439e-9deb-6a3687b65dfc.mp4


# Two-dimensional isotropic turbulence (inverse cascade !)

https://user-images.githubusercontent.com/32913492/144206236-b5692910-f03f-4bd3-8fbe-fe6f35885432.mp4

# Code documentation
Local compile instructions for the code: 

```g++ -I ../eigen-3.4.0 -I ./io-tools -I /opt/homebrew/opt/fftw/include -L /opt/homebrew/opt/fftw/lib -lfftw3 -lm main.cpp -o main```

File structure:
main.cpp: contains the Simulation class which performs the numerical simulation.
plot.py: plots the output of the simulation to obtain the visualizations. 
