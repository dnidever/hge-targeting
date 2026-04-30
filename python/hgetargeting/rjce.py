import os
import numpy as np

def rjce(hmag,_45mag,wise=False):
    """ Estimate extinction using the RJCE method """
    n = len(hmag)
    dt = [('ak',float),('ejk',float),('ah',float)]
    out = np.zeros(len(hmag),dtype=np.dtype(dt))
    out['ak'] = np.nan
    out['ejk'] = np.nan
    out['ah'] = np.nan

    gd, = np.where(np.isfinite(hmag) & np.isfinite(_45mag))
    
    # Spitzer/GLIMPSE
    if wise==False:
        out['ak'][gd] = 0.918*(hmag[gd] - _45mag[gd] - 0.08)
    
    # WISE
    else:
        out['ak'][gd] = 0.918*(hmag[gd] - _45mag[gd] - 0.05)
    # ah = ak*1.55
    out['ah'][gd] = out['ak'][gd]*1.55
    # ejk = 1.50*ak
    out['ejk'][gd] = out['ak'][gd]*1.5

    return out
