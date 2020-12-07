# Forest automaton simulation
#### Author: Eddie Lee, edlee@santafe.edu

This is the code repository for the manuscript "Dynamics of growth, death, and competition
in sessile organisms" by Edward D. Lee, Chris P. Kempes, Geoffrey B. West. The preprint is
located [here](https://arxiv.org/abs/2009.14699).

The code and parameter settings for simulations shown in the figures are in "pipeline.py".

Simulation results are shown in the Jupyter notebook "plotting.ipynb". To run the
notebook, clone the repo and use it as a Python module named "pyutils" for the notebook.
For example, to do this the reader might run
```bash
$ git clone https://github.com/eltrompetero/forests.git
$ mv forests pyutils
$ mv pyutils/plotting.ipynb ./
$ jupyter notebook
```
Then, the notebook should run after running the pipeline functions to generate simulation
output.
