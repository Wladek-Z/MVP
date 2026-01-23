# Ising.py

Ising.py contains the Ising class, which can run simulations of the Ising model. Running Ising.py displays an animation of the time evolution of a specified Ising model system, indefinitely. The program can be executed through the cmd terminal, and may take 3 arguments: L (system size), kBT (thermal energy), dynamics (dynamics method, 'G' or 'K' for Glauber or Kawasaki dynamics, respectively). Alternatively, one can run Ising.py without specifying any arguments, in which case the program will default to L=50, kBT=2, and dynamics='G'.

## Usage

```bash
$ python ising.py 50 1.6 'G'
```

```bash
$ python ising.py
```





# task456_collect.py

```python
filepath = 'path/to/file.txt'
collect_task4.py(filepath)
```

```python
filepath = 'path/to/file.txt'
collect_task56.py(filepath, 'K')
```