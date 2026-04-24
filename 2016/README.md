Date: 21/04/2026

# ising.py

Ising.py contains the Ising class, which runs simulations of the 2D Ising model for an antiferromagnet. Running Ising.py displays an indefinite animation of the time evolution of the Ising model system, along with the current values of magnetisation and staggered magnetisation. 

## Arguments

    -h, --help            
        -Show help message and exit

    -L, --size SIZE       
        -System size 
        -Default: 50

    -T, --temperature TEMPERATURE
        -Thermal energy 
        -Default: 2

    -J, --coupling COUPLING  
        -Coupling constant
        -Default: -1

    -H, --field FIELD     
        -External magnetic field 
        -Default: 0

## Usage

The program should be executed through the terminal. Example usage is shown below.


# Package Dependencies

All code has been developed and tested using Python 3.14.3. Additional package dependencies are listed below.

* numpy 2.4.4
* numba 0.65.0
* matplotlib 3.10.8
* pandas 3.0.2

Optional package dependencies:

* scienceplots 2.2.1 
    - only required to reproduce plots