# gol.py

Contains the GameOfLife class, for running simulations of Conway's Game of Life.

## Arguments

    -L SIZE, --size SIZE  
        -System size
        -Type: integer
        -Default: 50

    -i {random,blinker,glider}, --initialstate {random,blinker,glider}
        -Initial state of the game board
        -Type: string
        -Default: random

    -t {animation,2,3}, --task {animation,2,3}
        -Select a task for the simulation
        -Type: string 
        -Default: animation

    -f FILENAME, --filename FILENAME
        -Filepath to save data, if applicable
        -Type: string
        -Default: None

## Usage

The program should be executed via the cmd terminal. Example usage is shown below.

### Run default simulation with animation

```bash
$ python gol.py
```

### Run simulation with animation and custom system size, preset initial state

```bash
$ python gol.py -L 50 -i blinker
```

### Collect data for task 3

```bash
$ python gol.py -t 3 -f gol_task3.txt
```

# gol_analysis.py

Runs analysis of various outputs from the GameOfLife class.

## Arguments

    -t {2,3}, --task {2,3}                 
        -Select which task to perform the analysis for 
        -Type: integer
        -Default: None

    -f FILENAME, --filename FILENAME
        -Filepath to read data for the relevant task 
        -Type: string
        -Default: None

## Usage

The program should be executed via the cmd terminal. Example usage is shown below.

### Show histogram of equilibration times for task 2

```bash
$ python gol_analysis.py -t 2 -f gol_task2.txt
```

### Calculate glider speed for task 3

```bash
$ python gol_analysis.py -t 3 -f gol_task3.txt
```

# sirs.py

Contains the SIRS class, for running simulations of the SIRS model.

## Arguments

    -L SIZE, --size SIZE 
        -System size 
        -Type: integer
        -Default: 50

    -p1 PROBABILITYSI, --probabilitySI PROBABILITYSI
        -Probability of susceptible becoming infected 
        -Type: float
        -Default: 0.5

    -p2 PROBABILITYIR, --probabilityIR PROBABILITYIR
        -Probability of infected becoming recovered
        -Type: float
        -Default: 0.5

    -p3 PROBABILITYRS, --probabilityRS PROBABILITYRS
        -Probability of recoverd becoming susceptible
        -Type: float
        -Default: 0.5

    -s {absorbing,dynamic,cyclic}, --state {absorbing,dynamic,cyclic}
        -Select one of three preset states
        -Type: string
        -Default: None

    -t {animation,3,4,5}, --task {animation,3,4,5}
        -Select a task for the simulation
        -Type: string
        -Default: animation

    -i IMMUNE, --immune IMMUNE
        -Choose fraction of the population with permanent immunity to the infection
        -Type: float
        -Default: 0

    -f FILENAME, --filename FILENAME
        -Filepath to save data, if applicable
        -Type: string
        -Default: None  

## Usage

The program should be executed via the cmd terminal. Example usage is shown below.

### Run default simulation with animation

```bash
$ python sirs.py 
```

### Run simulation of preset cyclic state with custom system size, animation

```bash
$ python sirs.py -s cyclic -L 50
```

### Run simulation with custom probabilities, fraction of permanently immune sites, animation

```bash
$ python sirs.py -p1 0.5 -p2 0.5 -p3 0.5 -i 0.25
```

### Collect data for task 4

```bash
$ python sirs.py -t 4 -f sirs_task4.txt
```

# sirs_analysis.py

Runs analysis of various outputs from the SIRS class.

## Arguments

    -t {3,4,5}, --task {3,4,5}               
        -Select which task to perform the analysis for 
        -Type: integer
        -Default: None

    -f FILENAME, --filename FILENAME
        -Filepath to read data for the relevant task 
        -Type: string
        -Default: None

## Usage

The program should be executed via the cmd terminal. Example usage is shown below.

### Show heat map of average fraction of infected sites for task 3

```bash
$ python sirs_analysis.py -t 3 -f sirs_task3.txt
```

### Show variance of the average fraction of infected sites for task 4

```bash
$ python sirs_analysis.py -t 4 -f sirs_task4.txt
```

### Show average number of infected sites against fraction of immunity for task 5

```bash
$ python sirs_analysis.py -t 5 -f sirs_task5.txt
```