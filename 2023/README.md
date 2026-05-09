# rps_deterministic.py

Contains the RockPaperScissors class, for running simulations of rock paper scissors using a parallel deterministic algorithm.

## Arguments

    -L, --size SIZE       
        -System size 
        -type: integer
        -default: 100

    -t, --task {animation,collect,plot}
        -Select a task for the simulation 
        -type: string
        -default: animation

## Usage

The program should be executed via the cmd terminal. Example usage is shown below.

### Run the animated simulation for a system size of 100

```bash
python rps_deterministic.py -L 100
```

### Run the data collection procedure for task b

```bash
python rps_deterministic.py -t collect
```

### Plot the results for task b

```bash
python rps_deterministic.py -t plot
```

# rps_random.py

Contains the RockPaperScissors class, for running simulations of rock paper scissors using a random sequential algorithm.

## Arguments

    -L, --size SIZE       
        -System size
        -type: integer 
        -default: 50

    -p1 P1                
        -Probability of scissors turning to rock 
        -type: float
        -default: 0.5

    -p2 P2                
        -Probability of rock turning to paper 
        -type: float
        -default: 0.5

    -p3 P3                
        -Probability of paper turning to scissors
        -type: float 
        -default: 0.5

    -t, --task {animation,d,e}
        -Select a task for the simulation 
        -type: string
        -default: animation

    --collect             
        -Collect data for a given task

    --plot                
        -Plot results for a given task

## Usage

The program should be executed via the cmd terminal. Example usage is shown below.

### Run the animated simulation for a system size of 100

```bash
python rps_random.py -L 100
```

### Run the data collection procedure for task d

```bash
python rps_random.py -t d --collect
```

### Plot the results for task d

```bash
python rps_random.py -t d --plot
```

### Run the data collection procedure for task e

```bash
python rps_random.py -t e --collect
```

### Plot the results for task e

```bash
python rps_random.py -t e --plot
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

