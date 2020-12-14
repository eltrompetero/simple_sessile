# Forest automaton simulation
#### Author: Eddie Lee, edlee@santafe.edu

This is the Github code repository for the manuscript "Dynamics of growth, death, and
competition in sessile organisms" by Edward D. Lee, Chris P. Kempes, and Geoffrey B. West.
The preprint is located [here](https://arxiv.org/abs/2009.14699).

The original code base to be published with the manuscript will be available on the Zenodo
archive.


## Installation
You can use Anaconda to set up your Python environment to reproduce the results.  First,
git clone the repo and create the appropriate environment.
```bash
$ git clone https://github.com/eltrompetero/forests.git
$ conda env create -f forests/forests.yml
$ conda activate forests
```
This will create and activate the appropriate Anaconda environment named forests. Please
note that this environment is optimized for an AMD processor and an Intel-based machine
way require a different set of compiled packages.

We must also create directories where the results will be saved.
```bash
$ mkdir cache
```


## Reproduction
The code and parameter settings for simulations shown in the figures are in
"pyutils/pipeline.py".  The figures are in "plotting.ipynb".

Simulation results are shown in the Jupyter notebook "plotting.ipynb". To run the
notebook, the reader might run (after following the installation instructions above)
```bash
$ jupyter notebook
```
The code in the notebook relies on pickles generated from the pipeline functions
that cache simulation output.


## Technical specs
The code must be run on a multi-threaded machine with ample RAM (we suggest at least
32GB available) and sufficient hard drive space (~50GB). Some of the simulations may take
many hours to run. 

We used an Ubuntu system running on a system with an AMD Ryzen 7 1700
Eight-Core Processor (with 16 threads), 1.5TB of SSD space, 64GB of RAM, and 256GB of
PCIe drive swap space, which was much more than ample to finish each individual simulation
within hours.
