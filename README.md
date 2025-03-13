# feynman-kac-python-prototype
A little prototype for a Feynman-Kac based Poisson solver for my bachelor thesis.

# Installation
Assuming you have python and pip run first clone then in the root dir. 
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Useage
Run the `python main.py` in the src folder and any number of the optional arguments:
- `-e, --epsilon` defaults to .01 and sets the standard deviation that you want
to reach
- `-d, --dt0` defaults to .01 and sets the timestep at level 0 (or just the 
timestep for non-mlmc simulations)
- `-x, --x` defaults to .5 the starting x position
- `-y, --y` defaults to .5 the starting y position
- `--non_homogeneous` defaults to false uses the non-homogeneous test function
- `-s, --standard_mc` defaults to false if set runs non-mlmc simulation
- `-N, --N_samples` defaults to 256000 the number of samples for non-mlmc runs
- `-w, --plot_walk` defaults to false if set plots two correlated random walks
- `-d, --debug` defaults to false if set activates some more print statements

## Jupyter notebooks
enter the virtual environment and start your jupyter lab or notebook from there
