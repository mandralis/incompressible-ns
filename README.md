# incompressible-ns
This repository contains a C++ implementation of a 2D incompressible Navier Stokes solver. Please see associated report for more detail.
The videos mentioned in the report are displayed below:

# Kelvin-Helmholtz instability

https://user-images.githubusercontent.com/32913492/144205914-26fe5083-e961-43cb-8824-176f29f94b66.mp4

# 


Local compile instructions: 

```g++ -I ../eigen-3.4.0 -I ./io-tools -I /opt/homebrew/opt/fftw/include -L /opt/homebrew/opt/fftw/lib -lfftw3 -lm main.cpp -o main```
