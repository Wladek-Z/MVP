Name: Wladek Zawadzki<br>	
Date: 06/03/2026

# cahn.py

Simulation of the Cahn-Hilliard equation for liquid-liquid phase separation.

## Arguments
    
    --animation           
        -Run the animation

    --data                
        -Collect data for task 5
    
    --plot                
        -Plot the free energy for task 5

    -L SIZE, --size SIZE  
        -Systems size 
        -type: integer
        -default: 50

    -dt TIMESTEP, --timestep TIMESTEP
        -Time step 
        -type: float
        -default: 0.01

    -phi0 INITIALPHI, --initialphi INITIALPHI
        -Initial state of the order parameter field 
        -type: float
        -default: 0.0

## Usage

Note: all tasks must be executed through the command line

### Run simulation with animation

```bash
python cahn.py --animation
```

### Run simulation with animation using custom step size, initial order parameter field, and system size

```bash
python cahn.py --animation -dt 0.005 -phi0 -0.5 -L 100
```

### Collect free energy data for task 5

```bash
python cahn.py --data
```

### Plot free energy data for task 5

```bash
python cahn.py --plot
```

# poisson.py

## Arguments

## Usage

# Dependencies

* Python (3.12.13)
* numba (0.64.0)
* numpy (1.26.4)
* matplotlib (3.10.8)
* scipy (1.17.1)
