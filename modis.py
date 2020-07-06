# ====================================================================================== #
# Forest analysis from NASA's MODIS.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *



class EVImap():
    def __init__(self, fname, rng=None):
        """
        Parameters
        ----------
        fname : str
            Name of NETCDF4 file to load.
        rng : np.random.RandomState, None
        """
        
        self.fname = fname
        self.ds = Dataset(fname, 'r', format='NETCDF4')
        self.rng = rng or np.random.RandomState()

        self.set()

    def set(self):
        """Extract info from EVI file.

        Get EVI filtered by pixel reliability.
        """
        
        # extract prefix
        key = [k for k in self.ds.variables.keys() if '_EVI' in k][0][:-4]

        evi = self.ds[key+'_EVI'][:]
        # reliability does not need a mask for our purposes, so get rid of it
        reliability = ma.filled(self.ds[key+'_pixel_reliability'][:], 2)
        evi.mask[reliability!=0] = True

        self.evi = evi
        self.reliability = reliability
        self.xdim = self.ds['xdim'][:]
        self.ydim = self.ds['ydim'][:]

        self.nonEmptyIx = ~self.evi.mask.all(0).ravel()

    def sample_d(self, n_sample,
                 return_pairs=False,
                 rel_length_scale=5):
        """Sample distances between pairs of pixels with preferential sampling
        for nearby pixels using Gaussian.

        Parameters
        ----------
        n_sample : int
        return_pairs : bool, False
        rel_length_scale : float, 5.

        Returns
        -------
        ndarray
            Distances.
        list of tuples (optional)
            Pairs of pixels for which distances were calculated.
        """

        xdim = self.xdim
        ydim = self.ydim
        randij = []
        dmat = np.zeros(n_sample)
      
        counter = 0
        while counter<n_sample:
            # select two random pixels to compare
            ij1 = self.rng.randint(ydim.size), self.rng.randint(xdim.size)
            # select pixel within a typical distance from the first pixel
            ij2 = (int(self.rng.normal(loc=ij1[0], scale=ydim.size/rel_length_scale))%ydim.size,
                   int(self.rng.normal(loc=ij1[1], scale=xdim.size/rel_length_scale))%xdim.size)

            if ij1!=ij2:
                dmat[counter] = np.sqrt((ydim[ij1[0]] - ydim[ij2[0]])**2 +
                                        (xdim[ij1[1]] - xdim[ij2[1]])**2)

                randij.append((ij1,ij2))
                counter += 1 
        
        if return_pairs:
            return dmat, randij
        return dmat

    def sample_corr(self, pairij=None, return_pairs=False):
        """Calculate correlation through time between pairs of pixels.

        Parameters
        ----------
        pairij : list of tuples
        return_pairs : bool, False

        Returns
        -------
        ndarray
            Sampled correlations including nan.
        """
        
        if pairij is None:
            raise NotImplementedError

        C = np.zeros(len(pairij))
        for i, (ij1, ij2) in enumerate(pairij):
            # normalized by the number of nonzero entries
            notmaskix = ~(self.evi.mask[:,ij1[0],ij1[1]] | self.evi.mask[:,ij2[0],ij2[1]])
            T = notmaskix.sum()
            if T:
                C[i] = self.evi[:,ij1[0],ij1[1]].dot(self.evi[:,ij2[0],ij2[1]]) / T 
                C[i] -= self.evi[notmaskix,ij1[0],ij1[1]].mean() * self.evi[notmaskix,ij2[0],ij2[1]].mean()
            # if no nonzero entries, then set to nan
            else:
                C[i] = np.nan
        return C

    def d(self, ix=None):
        """Condensed matrix of distances between pixels. By default, only
        non-empty pixels are returned.

        Parameters
        ----------
        ix : ndarray, None
            If provided, only these pixels will be used to compute distance
            matrix, i.e. ix.size * (ix.size-1) // 2 distance calculations will
            be returned. This can be useful for subsampling.

        Returns
        -------
        ndarray
            Upper triangular part of distance matrix (only non-empty pixels).
            This can be expanded by using squareform().
        """

        xdim = self.xdim
        ydim = self.ydim
        x, y = np.meshgrid(xdim, ydim)

        # calculate indices of items to keep
        if ix is None:
            nonEmptyIx = self.nonEmptyIx
        else:
            # exclude all items that are either empty or not in the allowed list of ix
            nonEmptyIx = self.nonEmptyIx.copy()
            boolix = np.ones_like(nonEmptyIx)
            boolix[ix] = False
            nonEmptyIx[boolix] = False

        return pdist(np.vstack((x.ravel()[nonEmptyIx], y.ravel()[nonEmptyIx])).T)

    def corr(self, ix=None):
        """Correlation between pixels over time.
        
        The 2D grid is iterated over by rows first and excludes empty pixels
        (ones that have no records).
        
        Parameters
        ----------
        ix : ndarray, None
            If provided, only these elements will be used to compute distance
            matrix. This can be useful for subsampling.

        Returns
        -------
        ndarray
            Upper triangular array of dot product of vectors of time series.
        """
        
        evi = self.evi
        if ix is None:
            nonEmptyIx = self.nonEmptyIx
        else:
            # exclude all items that are either empty or not in the allowed list of ix
            nonEmptyIx = self.nonEmptyIx.copy()
            boolix = np.ones_like(nonEmptyIx)
            boolix[ix] = False
            nonEmptyIx[boolix] = False

        # reshape EVI such that each row follows a pixel over time
        X = evi.copy()
        X = np.moveaxis(X, 0, 2)
        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        X = X[nonEmptyIx]
        
        # take dot product between vectors in time
        # normalize by no. of nonzero entries in times
        counts = (~X.mask).dot((~X.mask.T).astype(int))
        return (X.dot(X.T) / counts)[np.triu_indices(X.shape[0], k=1)]

    def corrcoef(self):
        """Correlation coefficient between pixels over time.
        
        The 2D grid is iterated over by rows first and excludes empty pixels
        (ones that have no records).

        Returns
        -------
        ndarray
        """
        
        # reshape EVI such that each column follows a pixel over time
        X = self.evi.copy()
        X = np.moveaxis(X, 0, 2)
        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        X = X[self.nonEmptyIx]

        C = 1 - pdist(X, 'correlation')

        return C

    def coarse_grain_xy(self, factor=2, fun='mean'):
        """Coarse grain spatial resolution by given factor. There are multiple
        ways to coarse grain in real space. We will simply take the average of
        the visible pixels within a box of dimension factor x factor.

        Parameters
        ----------
        factor: int, 2
        fun : str, 'mean'
            Function with which to coarse-grain, can be 'mean', 'max', 'min'.

        Returns
        -------
        None
        """
        
        factor = int(factor)
        T = self.evi.shape[0]
        Lx = self.xdim.size//factor * factor
        Ly = self.ydim.size//factor * factor
        
        # average positions are simple to calculate
        xdim = ma.array([self.xdim[i*factor:(i+1)*factor].mean() for i in range(Lx//factor)])
        ydim = ma.array([self.ydim[i*factor:(i+1)*factor].mean() for i in range(Ly//factor)])
        
        # averaged spatial patterns over the 2D grid
        shape = (T, ydim.size, xdim.size)
        evi = ma.zeros(shape)
        reliability = np.zeros(shape, dtype=np.int8)

        evi.mask = np.zeros_like(evi, dtype=bool)
        
        # iterate through each box to coarse-grain
        # only consider all boxes that fill the entire factor x factor area
        for i in range(Ly//factor):
            for j in range(Lx//factor):
                # to use below
                sl = np.s_[:,i*factor:(i+1)*factor,j*factor:(j+1)*factor]

                r = self.reliability[sl]
                ix = (r==0).any(1).any(1)  # indicates where at least one pixel is reliable
                # reliability is by default 0 unless all pixels were unreliable
                reliability[~ix,i,j] = 1

                # take averaged evi only over non-empty pixels and only over the
                # two dimensions of space (not time)
                if fun=='mean':
                    evi[:,i,j] = self.evi[sl].mean(1).mean(1)
                elif fun=='max':
                    evi[:,i,j] = self.evi[sl].max(1).max(1)
                elif fun=='min':
                    evi[:,i,j] = self.evi[sl].min(1).min(1)
                else:
                    raise NotImplementedError("Unrecognized coarse-graining function.")
        
        # some checks
        assert xdim.size==evi.shape[-1]==reliability.shape[-1]
        assert ydim.size==evi.shape[1]==reliability.shape[1]
        assert T==evi.shape[0]==reliability.shape[0]

        self.xdim = xdim
        self.ydim = ydim
        self.evi = evi
        self.reliability = reliability
        self.nonEmptyIx = ~self.evi.mask.all(0).ravel()

    def spatial_corr(self, dmat, C, frac=1, bins=100):
        """Spatial correlation function. This organizes already calculated
        correlation function and distances.

        Parameters
        ----------
        dmat : ndarray
            Distances between pixels whose correlations are given in C.
        C : ndarray
        frac : float, 1
            Random fraction of elements to sample.
        bins : int or ndarray, 100
            Bins correspond to distances between pixels into which correlation
            function will be binned.

        Returns
        -------
        ndarray
            Distance.
        ndarray
            Average correlation for corresponding distance.
        ndarray
            Std of correlation.
        ndarray
            Number of observations.
        """
        
        if frac<1:
            # subsample
            randix = self.rng.randint(dmat.size, size=int(dmat.size*frac))
            dmat = dmat[randix]
            C = C[randix]
        
        # construct bins for digitizing
        if type(bins) is int:
            ud = np.linspace(0, dmat.max()+1, bins)
        else:
            ud = bins
        # index of 0 corresponds to anything <=bins[0]
        uix = np.digitize(dmat, ud, right=True) - 1

        # take averages of items in bins
        m = np.zeros(uix.max()+1)
        s = np.zeros(uix.max()+1)
        n = np.zeros(uix.max()+1)
        for i in range(uix.max()+1):
            ix = uix==i
            m[i] = C[ix].mean()
            s[i] = C[ix].std()
            n[i] = ix.sum()
        
        return ud, m, s, n
#end EVImap 
