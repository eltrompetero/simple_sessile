# Numerical solutions to spatial competition in sessile organisms
#### Author: Eddie Lee, edlee@santafe.edu

This is the Github code repository for the manuscript "Growth, death, and competition in
sessile organisms" by Edward D. Lee, Chris P. Kempes, and Geoffrey B. West.  The preprint
is located [here](https://arxiv.org/abs/2009.14699).


## Installation
You can use Anaconda to set up your Python environment to reproduce the automaton
simulation results. First, git clone the repo and create the appropriate environment. Note
that the code below installs some custom modules to run.
```bash
$ git clone https://github.com/eltrompetero/forests.git
$ git clone https://github.com/eltrompetero/workspace.git
$ git clone https://github.com/eltrompetero/misc.git
$ conda env create -f forests/forests.yml
$ conda activate forests
$ mkdir cache
```
This will create and activate the appropriate Anaconda environment named forests. Please
note that this environment is optimized for an AMD processor and an Intel-based machine
may require a different set of compiled packages.


## Reproduction
The code and parameter settings for simulations shown in the figures are in
[pyutils/pipeline.py](pyutils/pipeline.py).  The figures are in
[plotting.ipynb](plotting.ipynb).

Simulation results are shown in the Jupyter notebook [plotting.ipynb](plotting.ipynb). To
run the notebook, the reader might run (after following the installation instructions
above)
```bash
$ jupyter notebook
```
The code in the notebook relies on pickles generated from the pipeline functions
that cache simulation output.

Mathematica code for running the mean-field solutions is in the mathematica directory.


## Further simulation and extensions
We suggest that those interested in running our simulations for particular systems modify
the parameter values detailed in [pipeline.py](pyutils/pipeline.py) appropriately. The
source code there also provides examples of how to run our automaton simulations.


## Technical specs
The code must be run on a multi-threaded machine with ample RAM (we suggest at least
32GB available) and sufficient hard drive space (~50GB). Some of the simulations may take
many hours to run. 

We used an Ubuntu system running on a system with an AMD Ryzen 7 1700
Eight-Core Processor (with 16 threads) at 3.0GHz, 1.5TB of SSD space, 32GB of RAM, and 256GB of
PCIe drive swap space, which was much more than ample to finish each individual simulation
call inside [pipeline.py](pyutils/pipeline.py) within hours.
