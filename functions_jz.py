# functions_jz.py - a module to accompany jz.py
# by alexanderjames
#https://github.com/alexjsolar/Jz

#Imports
import os
from os import listdir                          #for finding input files
from os.path import isfile, join                #for reading input files
import numpy as np                              #for 'np.linspace' plotting grid
import astropy.units as u                       #for units
from scipy.ndimage.interpolation import shift   #shift arrays for grdients
import scipy.constants as constants             #for constants (mu_0)

#-----------------------------------------------------------------------------#

def check_make_dir(path, verbose=False):
    """
    Check if the directory "path" exists, and if it doesn't, make it.
    :param path: directory.
    :param verbose: print message when making directory.
    """
    if not os.path.exists(path):        #if directory doesn't exist
        if verbose:                     #if printing error messages
            print('making directory')   #print error message
        os.makedirs(path)               #make directory
    return

#-----------------------------------------------------------------------------#

def find_fits(path,criteria='',ext='.fits'):
    """
    Create a list of all desired FITS files from a directory,
    optionally only those that match a critera in their names.
    Example call:
        files = find_fits('/Users/path/', 'variable', '.fits')
    """
    #Find all desired FITS files
    files = [path+f for f in listdir(path) if isfile(join(path, f))
             and criteria+ext in f]
    #Sort alphanumerically for correct date/time order
    files = sorted(files)
    #Reformat list as array
    files = np.asarray(files)
    #Return the list of FITS files
    return files

#-----------------------------------------------------------------------------#

def compute_jz(tmap, pmap, mask=False, tr_cutoff=100):
    """
    Computes a map of vertical current density, jz, from maps of Bp and Bt.
    If mask=True, jz is set to 0 in pixels where the transverse magnetic field
    strength is weaker than a cutoff value, tr_cutoff.
    Note: Reversing the sign of tmap is handled in this function.
    """
    #Conversion: hmi pixel = 0.5 arcsec = 0.03 deg. 1 arcsec at 1 AU = 725.27 km
    degree_to_km = ((0.5*u.arcsec)/(0.03*u.deg)) * ((725.27*u.km)/(1*u.arcsec))
    #Quantities needed for computing Jz
    dx = (tmap.scale[0])*(1*u.pix) * degree_to_km #pizel width in km
    dy = (tmap.scale[1])*(1*u.pix) * degree_to_km #pixel height in km
    dby = (shift(-tmap.data, [0,-1], cval=np.NaN)-shift(-tmap.data, [0,1], cval=np.NaN)) * u.gauss
    dbx = (shift(pmap.data, [-1,0], cval=np.NaN)-shift(pmap.data, [1,0], cval=np.NaN)) * u.gauss
    npix = 2 #relative number of pixels shifted when getting dby and dbx (1--1=2)
    dbydx = dby/(npix*dx) #gradient of By in x
    dbxdy = dbx/(npix*dy) #gradient of Bx in y
    u_0 = constants.mu_0 * (u.tesla*u.m/u.amp) #constant
    #Compute Jz
    jz = ((dbydx-dbxdy)/u_0).decompose()    #compute Jz and simplify units
    unit = jz.unit                          #take unit of Jz
    jz[np.isnan(jz)] = 0                    #change nans in Jz to zeroes
    #mask current where transverse field is weak
    if mask:
        mag_tr = np.sqrt(tmap.data**2 + pmap.data**2) #transverse field magnitude
        mask = mag_tr<=tr_cutoff        #find pixels below threshold
        jz[mask] = 0                    #set masked pixels to 0
    #format as an array with correct units
    jz = np.asarray(jz) * unit
    #Return the Jz array
    return jz

#-----------------------------------------------------------------------------#
