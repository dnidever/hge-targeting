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


def uniformcmdsampling(jk,hmag):
    """ Uniformly sample the stars in the CMD space """

    nstars = len(jk)
    
    # Pick lots of random colors and mags
    jkr = [np.min(jk),np.max(jk)]
    hr = [np.min(hmag),np.max(hmag)]
    # scale jk and hmag from 0-1
    jkscaled = (jk-jkr[0])/(jkr[1]-jkr[0])
    hmagscaled = (hmag-hr[0])/(hr[1]-hr[0])
    #xrnd = np.random.rand(10*nstars)
    #yrnd = np.random.rand(10*nstars)
    
    ## find the closest matches with KDTree
    #X1 = np.vstack((jkscaled,hmagscaled)).T
    #X2 = np.vstack((xrnd,yrnd)).T

    #kdt = cKDTree(X2)
    #k = 100
    #dist, ind = kdt.query(X1, k=k, distance_upper_bound=0.05)
    ##index = ind[:,0]
    ##si = np.argsort(index)

    ## need to deal with duplicate matches
    #taken = np.zeros(len(xrnd),bool)
    #index = np.zeros(nstars,int)-1
    #for i in range(nstars):
    #    count = 0
    #    while taken[ind[i,count]] and count<k-1:
    #        count += 1
    #    if taken[ind[i,count]]==False and count<k-1:
    #        index[i] = ind[i,count]
    #        taken[index[i]] = True
    #    else:
    #        print('problem')
    #        #import pdb; pdb.set_trace()
    #
    #priority = np.zeros(nstars,int)
    #si = np.argsort(index)
    #priority[si] = np.arange(nstars)+1

    #bad, = np.where(index==-1)
    #if len(bad)>0:
    #    print('some unmatched stars')

    # find the closest matches with KDTree
    jkscaled = (jk-jkr[0])/(jkr[1]-jkr[0])
    hmagscaled = (hmag-hr[0])/(hr[1]-hr[0])
    X1 = np.vstack((jkscaled,hmagscaled)).T
    kdt = cKDTree(X1)

    flag = True
    count = 0
    left = np.arange(nstars)
    index = []
    while (flag):
        rnd = np.random.rand(2)
        X2 = np.atleast_2d(rnd)
        dist = np.sqrt((X1[:,0]-rnd[0])**2+(X1[:,1]-rnd[1])**2)
        ind = np.argmin(dist)
        mindist = np.min(dist)
        #dist, ind = kdt.query(X2, k=1, distance_upper_bound=0.1)
        #if np.isfinite(dist)==False:
        #    print(count,dist,ind,len(left),'no matches')
        #    count += 1
        #    continue
        # Found a good one
        index.append(left[ind])
        X1 = np.delete(X1,ind,axis=0)
        #kdt = cKDTree(X1)
        left = np.delete(left,ind)
        #print(count,mindist,ind,len(left))
        count += 1
        if len(left)==0:
            flag = False

    priority = np.zeros(nstars,int)
    si = np.argsort(index)
    priority[si] = np.arange(nstars)+1
        
    return priority
