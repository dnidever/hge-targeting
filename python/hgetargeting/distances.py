import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from . import utils

datadir = utils.datadir()
MCOEF = Table.read(datadir+'tmass_rgb_coef.fits')

def distance(jk,hmag,feh=None):
    """ Estimates distances from the photometry with dereddened J-Ks colors and Hmag
        and assuming solar metallicity isochrone """

    dist = np.zeros(len(jk),float)
    
    # No metallicity information input
    #  using solar metallicity
    if feh is None:
        cc = MCOEF['coef'][6]
        abshmag = np.polyval(cc,targs['jk0'])
        dmod = targs['h0']-abshmag
        dist = 10**((dmod+5)/5)/1e3

    # Use input metallicity information
    else:
        pass


    return dist
