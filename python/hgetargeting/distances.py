import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from . import utils

# RGB coefficients
datadir = utils.datadir()
MCOEF = Table.read(datadir+'tmass_rgb_coef.fits')

def distance(jk,hmag,mh=None):
    """ Estimates distances from the photometry with dereddened J-Ks colors and Hmag
        and assuming solar metallicity isochrone """

    dist = np.zeros(len(jk),float)
    
    # No metallicity information input
    #  using solar metallicity
    if feh is None:
        cc = MCOEF['coef'][6]
        abshmag = np.polyval(cc,jk)
        dmod = hmag-abshmag
        dist = 10**((dmod+5)/5)/1e3

    # Use input metallicity information
    else:
        # Find closest metallicity, mh
        # [-1.5 , -1.25, -1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25]
        mh0 = MCOEF['mh'][0]
        dmh = MCOEF['mh'][1]-MCOEF['mh'][0]
        mhclip = np.clip(mh,MCOEF['mh'][0],MCOEF['mh'][-1])
        # Get the metallicity index for each star
        mind = np.round((mhclip-mh0)/dmh).astype(int)
        order = MCOEF['coef'].shape[1]
        abshmag = np.zeros(len(jk))
        # Loop over polynomial order and add that component for each star
        for i in range(order):
            cc = MCOEFS['coef'][mind,i]
            # p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
            abshmag += cc*jk**(order-i)
        dmod = hmag-abshmag
        dist = 10**((dmod+5)/5)/1e3

    return dist
