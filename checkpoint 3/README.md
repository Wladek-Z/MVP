Name: Wladek Zawadzki<br>	
Date: 06/03/2026

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
        -default: 49

    -t TOLERANCE, --tolerance TOLERANCE
        -Accuracy of final solution 
        -default: 1e-6

    --monopole            
        -Calculate potential due to a single charge at the centre

    --task10              
        -Find optimal value of w in SOR method. (task 10)

    --wire                
        -Calculate potential due to a straight wire through the centre

    -m {Jacobi,Gauss-Seidel,SOR}, --method {Jacobi,Gauss-Seidel,SOR}
        -Method for solving Poisson's equation 
        -default: Jacobi

    -w RELAXATION, --relaxation RELAXATION
        -Relaxation parameter for SOR method 
        -default: 1.5

## Usage

Note: all tasks must be executed through the command line

### Solve for the electrostatic potential and electric field due to a monopole, save data for 2D slice to file, compare to Gauss's law (task 7)

```bash
python poisson.py --monopole
```

### Task 7 but with custom method, tolerance, and system size

```bash
python poisson.py --monopole -m Gauss-Seidel -t 0.01 -L 79 
```

### Solve for the z-component of the magnetic vector potential and magnetic field due to an infinite wire running along the z-axis, save data for 2D slice to file, and compare to Ampere's law (task 9)

```bash
python poisson.py --wire
```

### Task 9 but with the SOR method

```bash
python poisson.py --wire -m SOR -w 1.6
```

### Find the optimal value of the relaxation parameter such as to minimise the number of iterations required for convergence with the SOR method (task 10)

```bash
python poisson.py --task10
```

# Dependencies

* Python (3.12.13)
* numba (0.64.0)
* numpy (1.26.4)
* matplotlib (3.10.8)
* scipy (1.17.1)
