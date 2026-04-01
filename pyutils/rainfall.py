# Rainfall data from BCI.
# Author: Eddie Lee, edlee@csh.ac.at
import pandas as pd
import duckdb as db
from scipy.signal import fftconvolve
import time

from .posterior import random, init_to_sample, NUTS, MCMC
from .utils import *

BCI_DATADR = '../data/ForestGeoDatasets/bci_elect_cl_ra'
NEON_DATADR = '../data/NEON_precip-tipping'



# ======== #
# Wrappers #
# ======== #
def collect_rainfall_sample(model,
                            X,
                            n_loop,
                            key=None,
                            iprint=True,
                            **mcmc_kwargs):
    """MCMC with NUTS kernel to obtain samples from posterior.

    Parameters
    ----------
    model : callable
        Model to sample from.
    X : ndarray
        Data to sample from.
    n_loop : int
        Number of iterations to run.
    key : jax.random.PRNGKey, optional
        Random key for sampling, by default None.
    iprint : bool, optional
        Whether to print progress, by default True.
    mcmc_kwargs : dict, optional
        Additional arguments for MCMC, by default {}.
        - num_warmup : int
            Number of warmup steps.
        - num_samples : int
            Number of samples to draw.
        - num_chains : int
            Number of chains to run in parallel.
        - thinning : int
            Thinning interval for samples.
    """
    sample = []
    key = key if not key is None else random.PRNGKey(np.random.randint(2**32-1))

    for i in range(n_loop):
        if iprint: t0 = time.time()
        nuts_kernel = NUTS(model, dense_mass=True, max_tree_depth=10, init_strategy=init_to_sample)
        sampler = MCMC(nuts_kernel, **mcmc_kwargs)
        
        key, subkey = random.split(key)
        sampler.run(subkey, X)
        sample.append(sampler.get_samples(True).copy())
        if iprint: print(f"Done with iteration {i+1}/{n_loop} in {time.time()-t0:.2f} seconds.")
    return sample



# ==== #
# Data #
# ==== #
def count_consecutive_zeros(arr):
    # Identify where the array is zero
    is_zero = (arr == 0).astype(int)
    
    # Find the start and end of consecutive zero sequences using `np.diff`
    zero_diff = np.diff(is_zero, prepend=0, append=0)
    starts = np.where(zero_diff == 1)[0]  # Start of zero sequences
    ends = np.where(zero_diff == -1)[0]   # End of zero sequences
    
    # Calculate lengths of consecutive zero sequences
    lengths = ends - starts
    return lengths

def prep_bci_rainfall():
    """Convert CSV file from Chris Kempes to parquet (for speed and space).
    """
    q = f'''
        COPY(
            SELECT *
            FROM read_csv('{BCI_DATADR}/bci_cl_ra_elect.csv')
        ) TO '{BCI_DATADR}/bci_cl_ra_elect.parquet' (FORMAT PARQUET);
        '''
    conn = db.connect(':memory:', read_only=False)
    conn.execute(q)

class BCI_Rainfall:
    """Rainfall data from BCI.
    """
    DT = 5  # minutes

    def __init__(self):
        self.name = 'bci_rainfall'
        self.conn = db.connect(':memory:', read_only=False)
        self.load()
        self.conn.execute('ALTER TABLE rainfall5 ADD COLUMN smoothed_ra DOUBLE;')

    def load(self):
        """Load rainfall data from parquet file.
        """
        if not os.path.exists(f'{BCI_DATADR}/bci_cl_ra_elect.parquet'):
            prep_bci_rainfall()

        q = f'''SET enable_progress_bar = false;

            -- import original data and cleanup
            CREATE TABLE rainfall_ AS
            SELECT ROW_NUMBER() OVER () as row, datetime, date, GREATEST(ra, 0) AS ra, raw, chk_note, chk_fail
            FROM parquet_scan('{BCI_DATADR}/bci_cl_ra_elect.parquet')
            ORDER BY datetime;

            -- remove duplicate row
            DELETE FROM rainfall_
            WHERE row NOT IN (
                SELECT MIN(row)
                FROM rainfall_
                WHERE datetime = '2017-08-04 05:55:00'
                GROUP BY datetime
            )
            AND datetime = '2017-08-04 05:55:00';

            -- create five-minute binned rainfall data, filling in missing intervals with 0 rainfall
            CREATE TABLE rainfall5 AS
            WITH rainfall_summarized AS (
                SELECT interval_end, SUM(ra_sum) AS ra_sum
                FROM (SELECT TIME_BUCKET('5 minutes', datetime, INTERVAL '5 minutes') AS interval_end,
                            SUM(ra) AS ra_sum
                    FROM rainfall_
                    GROUP BY interval_end)
                GROUP BY interval_end
                ORDER BY interval_end
            ), intervals AS (
                -- Generate five-minute intervals (this is chosen as one of the smaller measurement
                -- intervals common in BCI surveys)
                SELECT 
                    MIN(datetime) AS start_time,
                    MAX(datetime) AS end_time
                FROM rainfall_
            ), all_intervals AS (
                    -- Generate a complete series of five-minute intervals
                    SELECT 
                        UNNEST(GENERATE_SERIES(start_time, end_time, '5 minutes')) AS interval_end
                    FROM intervals
            )
            -- Merge the complete intervals with the summarized data
            SELECT 
                ai.interval_end AS datetime,
                EXTRACT(YEAR FROM ai.interval_end) AS year,
                EXTRACT(MONTH FROM ai.interval_end) AS month,
                EXTRACT(DAY FROM ai.interval_end) AS day,
                COALESCE(rs.ra_sum, 0) AS ra
            FROM all_intervals ai
            LEFT JOIN rainfall_summarized rs
                ON ai.interval_end = rs.interval_end
            ORDER BY ai.interval_end;
        '''
        self.conn.execute(q)

    def smooth_rainfall_exp(self, decay_timescale, measurement_dt=None):
        """Smooth rainfall trajectory with exponential kernel.

        Parameters
        ----------
        decay_timescale : float
            Decay timescale in days that goes into exponential kernel.
        measurement_dt : float, 5
            Spacing between measurements in minutes.

        Returns
        -------
        np.ndarray
        """
        measurement_dt = measurement_dt or self.DT
        ra = self.conn.execute('select ra from rainfall5').fetchdf()['ra'].values.ravel()
        if decay_timescale==0:
            raise NotImplementedError('Decay timescale cannot be 0.')
            
        decay_timescale *= 24 * 60  # convert into minutes
        
        kernel = np.zeros(int(decay_timescale//measurement_dt * 4 * 2 + 1))
        kernel[kernel.size//2:] += np.exp(-np.arange(kernel.size//2+1)/(decay_timescale/measurement_dt))
        smoothed_ra = pd.DataFrame({'datetime':self.conn.execute('select datetime from rainfall5').fetchdf().values.ravel(),
                                    'ra':fftconvolve(ra, kernel, mode='same')})
        q = f'''
            UPDATE rainfall5
            SET smoothed_ra = smoothed_ra.ra
            FROM smoothed_ra
            WHERE rainfall5.datetime = smoothed_ra.datetime
            '''
        self.conn.execute(q)

    def q(self, q):
        """Run a query on the rainfall data and return dataframe.

        Parameters
        ----------
        q : str
            SQL query.
        """
        return self.conn.execute(q).fetchdf()
    
    def by_year(self, *args, threshold_fcn=np.mean):
        """Time spent in below-threshold rainfall.
        
        Parameters
        ----------
        year : int, two ints, or twople
            Year range [year[0], year[1]). Inclusive of first year, exclusive of second year.
        threshold_fcn : function, np.mean

        Returns
        -------
        list
            Times between "wet" spells in seconds.
        """
        if len(args)==2:
            years = args
        else:
            years = args[0]
        if not hasattr(years, '__len__'):
            years = (years, years+1)
        assert len(years)==2 and years[0]<years[1]
        
        q = f'''
            SELECT *
            FROM (SELECT datetime, year, ra, smoothed_ra
                FROM rainfall5)
            WHERE year>={years[0]} AND year<{years[1]}
            ORDER BY datetime
            '''
        return self.q(q)
#end BCI_Rainfall



class SCBI_Rainfall:
    """Rainfall data from SCBI (NEON tipping bucket precipitation).
    """
    DT = 5  # minutes

    def __init__(self, filter_qf=False):
        """
        Parameters
        ----------
        filter_qf : bool, False
            If True, only keep rows where finalQF==0 (pass).
        """
        self.name = 'scbi_rainfall'
        self.filter_qf = filter_qf
        self.conn = db.connect(':memory:', read_only=False)
        self.load()
        self.conn.execute('ALTER TABLE rainfall5 ADD COLUMN smoothed_ra DOUBLE;')

    def load(self):
        """Load rainfall data from NEON 1-minute CSVs.
        """
        glob_pattern = f'{NEON_DATADR}/NEON.D02.SCBI.*/NEON.D02.SCBI.*TIPPRE_1min*.csv'
        qf_filter = 'WHERE finalQF = 0' if self.filter_qf else ''

        q = f'''SET enable_progress_bar = false;

            -- import all 1-minute CSVs and cleanup
            CREATE TABLE rainfall_ AS
            SELECT ROW_NUMBER() OVER () as row,
                   startDateTime AS datetime,
                   GREATEST(precipBulk, 0) AS ra
            FROM read_csv('{glob_pattern}')
            {qf_filter}
            ORDER BY datetime;

            -- create five-minute binned rainfall data, filling in missing intervals with 0 rainfall
            CREATE TABLE rainfall5 AS
            WITH rainfall_summarized AS (
                SELECT interval_end, SUM(ra_sum) AS ra_sum
                FROM (SELECT TIME_BUCKET('5 minutes', datetime, INTERVAL '5 minutes') AS interval_end,
                            SUM(ra) AS ra_sum
                    FROM rainfall_
                    GROUP BY interval_end)
                GROUP BY interval_end
                ORDER BY interval_end
            ), intervals AS (
                SELECT
                    MIN(datetime) AS start_time,
                    MAX(datetime) AS end_time
                FROM rainfall_
            ), all_intervals AS (
                    SELECT
                        UNNEST(GENERATE_SERIES(start_time, end_time, '5 minutes')) AS interval_end
                    FROM intervals
            )
            SELECT
                ai.interval_end AS datetime,
                EXTRACT(YEAR FROM ai.interval_end) AS year,
                EXTRACT(MONTH FROM ai.interval_end) AS month,
                EXTRACT(DAY FROM ai.interval_end) AS day,
                COALESCE(rs.ra_sum, 0) AS ra
            FROM all_intervals ai
            LEFT JOIN rainfall_summarized rs
                ON ai.interval_end = rs.interval_end
            ORDER BY ai.interval_end;
        '''
        self.conn.execute(q)

    def smooth_rainfall_exp(self, decay_timescale, measurement_dt=None):
        """Smooth rainfall trajectory with exponential kernel.

        Parameters
        ----------
        decay_timescale : float
            Decay timescale in days that goes into exponential kernel.
        measurement_dt : float, 5
            Spacing between measurements in minutes.

        Returns
        -------
        np.ndarray
        """
        measurement_dt = measurement_dt or self.DT
        ra = self.conn.execute('select ra from rainfall5').fetchdf()['ra'].values.ravel()
        if decay_timescale==0:
            raise NotImplementedError('Decay timescale cannot be 0.')

        decay_timescale *= 24 * 60  # convert into minutes

        kernel = np.zeros(int(decay_timescale//measurement_dt * 4 * 2 + 1))
        kernel[kernel.size//2:] += np.exp(-np.arange(kernel.size//2+1)/(decay_timescale/measurement_dt))
        smoothed_ra = pd.DataFrame({'datetime':self.conn.execute('select datetime from rainfall5').fetchdf().values.ravel(),
                                    'ra':fftconvolve(ra, kernel, mode='same')})
        q = f'''
            UPDATE rainfall5
            SET smoothed_ra = smoothed_ra.ra
            FROM smoothed_ra
            WHERE rainfall5.datetime = smoothed_ra.datetime
            '''
        self.conn.execute(q)

    def q(self, q):
        """Run a query on the rainfall data and return dataframe.

        Parameters
        ----------
        q : str
            SQL query.
        """
        return self.conn.execute(q).fetchdf()

    def by_year(self, *args, threshold_fcn=np.mean):
        """Time spent in below-threshold rainfall.

        Parameters
        ----------
        year : int, two ints, or twople
            Year range [year[0], year[1]). Inclusive of first year, exclusive of second year.
        threshold_fcn : function, np.mean

        Returns
        -------
        list
            Times between "wet" spells in seconds.
        """
        if len(args)==2:
            years = args
        else:
            years = args[0]
        if not hasattr(years, '__len__'):
            years = (years, years+1)
        assert len(years)==2 and years[0]<years[1]

        q = f'''
            SELECT *
            FROM (SELECT datetime, year, ra, smoothed_ra
                FROM rainfall5)
            WHERE year>={years[0]} AND year<{years[1]}
            ORDER BY datetime
            '''
        return self.q(q)
#end SCBI_Rainfall
