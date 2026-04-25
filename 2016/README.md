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

    -P, --spatialperiod SPATIALPERIOD
            -Spatial period for space/time-dependent external magnetic field 
            -Default: 25

    -t, --task {animation,c,d}
        -Task to run: 'animation' for animation, 'c' for task c, or 'd' for task d 
        -Default: 'animation'

    --collect             
        -Collect data for a given task

    --plot                
        -Display plots for a given task

## Usage

The program should be executed through the terminal. Example usage is shown below.

### Run default simulation with animation (no field)

```bash
$ python ising.py
```

### Run simulation with animation and time-dependent field, choosing custom values of h0 and P

```bash
$ python ising.py -H 10 -P 20
```

### Collect the data for task c

```bash
$ python ising.py -t c --collect
```

### Plot the data for task c

```bash
$ python ising.py -t c --plot
```

### Collect the data for task d

```bash
$ python ising.py -t d --collect
```

### Plot the data for task d

```bash
$ python ising.py -t d --plot
```

# Package Dependencies

All code has been developed and tested using Python 3.14.3. Additional package dependencies are listed below.

* numpy 2.4.4
* numba 0.65.0
* matplotlib 3.10.8
* pandas 3.0.2

Optional package dependencies:

* scienceplots 2.2.1 
    - only required to reproduce plots