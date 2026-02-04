# ising.py

Ising.py contains the Ising class, which runs simulations of the 2D Ising model (with J = 1). Running Ising.py displays an animation of the time evolution of a specified Ising model system, indefinitely. 

## Arguments

-L: --size
    {integer}
    System size
    Default = 50

-T: --temperature
    {float}
    Thermal energy (kBT) of the system
    Default = 2.0

-d: --dynamics
    {string}, choice = {'G', 'K'}
    Dynamics method, either Glauber ('G') or Kawasaki ('K')
    Default = "G"

## Usage

The program can be executed through the cmd terminal.

```bash
$ python ising.py -L 50 -T 1.6 -d 'G'
```

```bash
$ python ising.py
```

```bash
$ python ising.py -T 2.4
```

# task456_collect.py

task456_collect.py runs iterations of the Ising model simulation using varying thermal energy parameters to collect either (i) thermal energy, average total absolute magnetisation, and magnetic susceptibility, or (ii) thermal energy, average total energy, and heat capacity per spin, and writes the data to a file. Case (ii) includes an additional Jackknife standard error computation on the heat capacity data. 

## Usage

To execute the data collection procedure, enter the filepath you wish to save the data to and call the function corresponding to the relevant task number, collect_task4(filepath) or collect_task56(filepath, dynamics) for taskk4, or tasks 5 and 6, respectively. For the latter two, the user must specify the type of dynamics to probe as an argument, 'G' or 'K' for Glauber or Kawasaki dynamics, respectively.

```python
if __name__ == '__main__':
    filepath = 'path/to/file.txt'
    collect_task4(filepath)
```

```python
if __name__ == '__main__':
    filepath = 'path/to/file.txt'
    collect_task56(filepath, 'K')
```

# task456_plot.py

task456_plot.py reads data from a file to produce two graphs. The data in each file must correspond to either (i) average total absolute magnetisation and magnetic susceptibility, or (ii) thermal energy, average total energy, heat capacity per spin, and Jackknife standard error on the heat capacity per spin. Thermal energy is always plotted as the independent variable, whilst the dependent variables are the second and third quantities from the data file for the first and second graphs, respectively. Additionally, the graph of heat capacity per spin includes the fourth quantity as error bars.

## Usage

To plot each pair of graphs, enter the filepath you wish to read the data from and call the function corresponding to the relevant task number, plot_task4(filepath) or plot_task56(filepath, dynamics). The function for plotting the task 5 and 6 data automatically determines which type of dynamics the given data file corresponds to by checking the filepath string and adjusts the graph titles accordingly. The title will specify Glauber or Kawasaki dynamics depending on whether the filepath string contains the substrings 'task4' or 'task5' for Glauber, or 'task6' for Kawasaki. Alternatively, if the filepath contains neither substring, the title will specify that the dynamics are unknown. The user may also specify the dynamics type manually by adding an additional positional argument 'G' or 'K' to the function call for Glauber or Kawasaki dynamics, respectively.

```python
if __name__ == '__main__':
    # set filepath to magnetisation/susceptibility data file
    filepath = 'path/to/file_task4.txt'
    plot_task4(filepath)
```

```python
if __name__ == '__main__':
    # set filepath to energy/heat capacity data file
    filepath = 'path/to/file.txt'
    plot_task56(filepath, 'G')
```

# collect.sh

Bash script for running the chosen data collection procedure via SLURM.

## Usage

'''bash
$ sbatch collect.sh
'''