This is a demo for an Agent-based epidemiology code built using the AMReX framework.

For more information about AMReX:
    website: https://amrex-codes.github.io/
    documentation: https://amrex-codes.github.io/amrex/docs_html/
    source code: https://github.com/AMReX-Codes/amrex

## Building the code

This demo uses CMake version 3.14 or higher. To build it:
     mkdir build
     cd build
     cmake ..
     make -j8

## Running the code

Navigate to build/bin and run the executable using one of the "inputs" files in "examples".

For example:
    cd build/bin
    ./agent ../../examples/inputs

## Visualizing the output

Running the code succesfully will create a number of "particles?????" files. You can visualize
the using the notebook at etc/Visualization.ipynb.