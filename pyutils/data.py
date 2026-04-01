# ====================================================================================== #
# Useful routines for analysis of observational data and simulation results.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from .utils import *
from . import nearest_neighbor as nn



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
    """Calculate correlation function for Namibian termate data set from Tarnita et al.
    """
    
    # load data
    df = pd.read_csv('../data/Tarnita/termite_mound_location_field_data/Namib_G1.txt',
                     sep='\t',
                     header=None)
    xy = df.values

    # exclude area 100 meters in from boundaries of rectangular plot
    p, r = nn.pair_correlation(xy, np.linspace(0, 10, 50), (100, 100, 400, 400))

    return p, r



class BCI():
    def __init__(self, csv_file='../data/BCI/bci.tree.parquet'):
        """
        Parameters
        ----------
        csv_file : str, '../data/BCI/bci.tree.parquet'
        """
        self.csv_file = csv_file
        self.conn = self.dbconn()

    def dbconn(self):
        """Duckdb connection to parquet file with bci demographic data."""
        conn = db.connect(':memory:', read_only=False)
        conn.execute(f'''CREATE TABLE bci AS SELECT * FROM parquet_scan('{self.csv_file}');''')
        return conn

    def execute(self, q, fetchdf=True):
        """Execute query on parquet file.
        
        Parameters
        ----------
        q : str
        fetchdf : bool, True

        Returns
        -------
        pd.DataFrame or None
        """
        if fetchdf:
            return self.conn.execute(q).fetchdf()
        return self.conn.execute(q)
#end BCI



class SCBI():
    def __init__(self,
                 csv_file='../data/ForestGeoDatasets/5_NAmerica_1_SCBI/SCBI_initial_woody_stem_census_2012.csv'):
        """
        Parameters
        ----------
        csv_file : str, '../data/ForestGeoDatasets/5_NAmerica_1_SCBI/SCBI_initial_woody_stem_census_2012.csv'):
        """
        self.csv_file = csv_file
        self.conn = self.dbconn()

    def dbconn(self):
        """Duckdb connection to csv file with demographic data."""
        conn = db.connect(':memory:', read_only=False)
        conn.execute(f'''CREATE TABLE scbi AS SELECT * FROM read_csv('{self.csv_file}');''')
        conn.execute('ALTER TABLE scbi RENAME COLUMN DBH TO dbh;')
        return conn

    def execute(self, q, fetchdf=True):
        """Execute query on parquet file.
        
        Parameters
        ----------
        q : str
        fetchdf : bool, True

        Returns
        -------
        pd.DataFrame or None
        """
        if fetchdf:
            return self.conn.execute(q).fetchdf()
        return self.conn.execute(q)

    def dbh(self):
        q = f'''
            SELECT dbh
            FROM scbi
            WHERE Stem='main'
                AND Status='alive'
                AND dbh IS NOT NULL
            '''
        return self.execute(q)
#end SCBI

