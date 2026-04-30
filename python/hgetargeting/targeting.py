import os
import numpy as np
from astropy.table import Table
from astropy.io import fits


def targeting():
    pass


def rjce_rgb(tab):
    # RGB cuts
    # fardisk_targets.fardisk_tragets()
    jkcut = [0.90,1.4,1.4,0.90,0.90]
    #hcut = [11.8+0.5,9.8,5.4,7.8,11.8+0.5]
    hcut = [11.8+1.0,9.8+0.5,5.4,7.8,11.8+1.0]
    ind,cutind = dln.roi_cut(jkcut,hcut,jk0,h0)
    rgb = {k: v[cutind] for k, v in d.items()}

    return rgb
