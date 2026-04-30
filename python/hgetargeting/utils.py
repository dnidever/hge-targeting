#!/usr/bin/env python

import os
import time
import numpy as np
import time
import shutil
from datetime import datetime
import subprocess
import tempfile
import logging
import re
from glob import glob
from astropy.io import fits,ascii
from astropy.time import Time
from astropy.table import Table
from astropy.wcs import WCS
from dlnpyutils import utils as dln
import struct
from itertools import zip_longest
from itertools import accumulate
from io import StringIO

def datadir():
    """ Return the doppler data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

def download_data(force=False):
    """ Download the data from my Google Drive."""

    # Check if the "done" file is there
    if os.path.exists(datadir()+'done') and force==False:
        return
    
    #https://drive.google.com/drive/folders/1SXId9S9sduor3xUz9Ukfp71E-BhGeGmn?usp=share_link
    # The entire folder: 1SXId9S9sduor3xUz9Ukfp71E-BhGeGmn
    
    data = [{'id':'17EDOUbzNr4cDzn7KZdPsw7R2lAj_o_np','output':'cannongrid_3000_18000_hotdwarfs_norm_cubic_model.pkl'},   
            {'id':'1hy1Mk6Kl8OAH6UouyLeGwtBXsOKoDq1r','output':'cannongrid_3000_18000_coolstars_norm_cubic_model.pkl'}]


    # Do the downloading
    t0 = time.time()
    print('Downloading '+str(len(data))+' Doppler data files')
    for i in range(len(data)):
        print(str(i+1)+' '+data[i]['output'])
        fileid = data[i]['id']
        url = f'https://drive.google.com/uc?id={fileid}'
        output = datadir()+data[i]['output']  # save to the data directory
        if os.path.exists(output)==False or force:
            gdown.download(url, output, quiet=False)

    print('All done in {:.1f} seconds'.format(time.time()-t0))


def isfloat(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def file_isfits(filename):
    """
    Check if this file is a FITS file or not. 
 
    Parameters
    ----------
    filename   The name of the file to check. 
 
    Returns
    -------
    return     1 if the file is a FITS file and 0 otherwise. 
 
    Example
    ------
  
    test = file_isfits(filename) 
 
    By D. Nidever, Jan 2019 
    Based partially from is_fits.pro by Dave Bazell 
    Translated to python by D. Nidever, April 2022
    """

    # Does the file exist 
    if os.path.exists(filename)==False:
        return False
     
    # Four possible possibilities: 
    # 1) Regular FITS file (this includes fpacked FITS files) 
    # 2) Gzipped FITS file 
    # 3) ASCII file 
    # 4) Some other binary file 
     
    # Try to open the file normally
    # astropy will catch lots of problems
    #  empty file, corrupted file, not a FITS file
    try:
        hdu = fits.open(filename)
    except:
        return False

    hdu.close()
    return True


def date2jd(dateobs,mjd=False):
    """ Converte DATE-OBS to JD or MJD."""
    t = Time(dateobs, format='fits')
    if mjd:
        return t.mjd
    else:
        return t.jd

