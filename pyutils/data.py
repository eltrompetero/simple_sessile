# ====================================================================================== #
# Useful routines for analysis of forest data.
# Author : Eddie Lee, edlee@santafe.edu
# 
#
# MIT License
# 
# Copyright (c) 2021 Edward D. Lee
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.
# ====================================================================================== #
from .utils import *
from . import nearest_neighbor as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection



def plot(xy, rbh, L,
         fig=None,
         fig_kw={'figsize':(6,6)},
         ax=None,
         plot_kw={},
         show_root=True,
         show_center=False,
         scale_factor=1):
    """Plotting of tree data given as coordinates and radii.

    Parameters
    ----------
    xy : ndarray
    rbh : radius
    fig : matplotlib.Figure, None
    fig_kw : dict, {'figsize':(6,6)}
    ax: mpl.Axes, None
    plot_kw : dict, {}
    class_ix : list, None
        Tree compartment indices to show.
    show_root : bool, True
    show_center : bool, False
    scale_factor : float, 1
        Scale factor by which to multiply the given radius for plotting.

    Returns
    -------
    matplotlib.Figure (optional)
        Only returned if ax was not given.
    """
    
    if ax is None:
        if fig is None:
            fig = plt.figure(**fig_kw)
        ax = fig.add_subplot(1,1,1)
        ax_given = False
    else:
        ax_given = True
    
    # root area
    if show_root:
        patches = []
        for i, xy_ in enumerate(xy):
            patches.append(Circle(xy_, rbh[i] * scale_factor))
        pcollection = PatchCollection(patches, facecolors='brown', alpha=.15)
        ax.add_collection(pcollection)

    # centers
    if show_center:
        ax.plot(xy[:,0], xy[:,1], 'k.', ms=2)
    
    # plot settings
    ax.set(xlim=(0, L), ylim=(0, L), **plot_kw)
    
    if not ax_given:
        return fig

def namibia_corr_fcn():
    df = pd.read_csv('../data/Tarnita/termite_mound_location_field_data/Namib_G1.txt',
                     sep='\t',
                     header=None)
    xy = df.values

    # exclude out 100 meters from boundary of plot
    p, r = nn.pair_correlation(xy, np.linspace(0, 10, 50), (100, 100, 400, 400))

    return p, r
