Name: Wladek Zawadzki<br>	
UUN: s2280450<br>
Date: 26/03/2026

# cahn.py

Code for solving the Cahn-Hilliard equation to simulate liquid-liquid phase separation in a system with periodic boundary conditions.

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

Iterative solver of Poisson's equation for an electrostatic or magnetic problem with Dirichlet boundary conditions.

## Arguments

    -L SIZE, --size SIZE  
        -System size 
        -type: integer
        -default: 49

    -tol TOLERANCE, --tolerance TOLERANCE
        -Accuracy of final solution 
        -type: float
        -default: 1e-6

    -t, --task {monopole,wire,7,9,10}
        -Choose which task to perform 
        -type: string
        -default: monopole

    -m {Jacobi,Gauss-Seidel,SOR}, --method {Jacobi,Gauss-Seidel,SOR}
        -Method for solving Poisson's equation 
        -type: string
        -default: Jacobi

    -w RELAXATION, --relaxation RELAXATION
        -Relaxation parameter for SOR method 
        -type: float
        -default: 1.5

## Usage

All tasks should be executed through the command terminal. Example usage is shown below.

### Solve for the electrostatic potential and electric field due to a monopole

```bash
python poisson.py -t monopole
```

### Solve for the electrostatic potential and electric field due to a monopole, save data for 2D slice to file, compare to Gauss's law (task 7)

```bash
python poisson.py -t 7
```

### Task 7 but with custom method, tolerance, and system size

```bash
python poisson.py -t 7 -m Gauss-Seidel -tol 0.01 -L 79 
```

### Solve for the magnetic potential and field due to a wire

```bash
python poisson.py -t wire
```

### Solve for the z-component of the magnetic vector potential and magnetic field due to an infinite wire running along the z-axis, save data for 2D slice to file, and compare to Ampere's law (task 9)

```bash
python poisson.py -t 9
```

### Task 9 but with the SOR method

```bash
python poisson.py -t 9 -m SOR -w 1.6
```

### Find the optimal value of the relaxation parameter such as to minimise the number of iterations required for convergence with the SOR method on the electrostatic problem, save raw data to file (task 10)

```bash
python poisson.py -t 10
```

# Dependencies

All code has been developed and tested using Python 3.14.3. Additional package dependencies are listed below.

* numpy 2.4.4
* numba 0.65.0
* matplotlib 3.10.8
* pandas 3.0.2

Optional package dependencies:

* scienceplots 2.2.1 
    - only required to reproduce plots
