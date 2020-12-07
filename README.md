# Forest automaton simulation
#### Author: Eddie Lee, edlee@santafe.edu

This is the code repository for the manuscript "Dynamics of growth, death, and competition
in sessile organisms" by Edward D. Lee, Chris P. Kempes, Geoffrey B. West. The preprint is
located [here](https://arxiv.org/abs/2009.14699).

The code and parameter settings for simulations shown in the figures are in "pipeline.py".

You can use Anaconda to set up your Python environment to generate reproduce the results.
First, git clone the repo and rename it.
```bash
$ git clone https://github.com/eltrompetero/forests.git
$ mv forests pyutils
$ mv pyutils/plotting.ipynb ./
$ conda env create -f forests/forests.yml
$ conda activate forests
```
This will create and activate the appropriate Python environment named forests.

We must also create directories where the results will be saved.
```bash
$ mkdir cache
$ mkdir plotting
```

Simulation results are shown in the Jupyter notebook "plotting.ipynb". To run the
notebook, the reader might run (after the above)
```bash
$ jupyter notebook
```
Then, the notebook should run after running the pipeline functions to generate simulation
output.

The code must be run on a multi-threaded machine with ample RAM (we suggest at least
32GB available) and sufficient hard drive space (~50GB).
