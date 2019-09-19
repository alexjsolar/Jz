# Jz.py
# alexanderjames

#---- Imports ----#

from os import listdir                          #for finding input files
from os.path import isfile, join                #for reading input files
import numpy as np                              #for 'np.linspace' plotting grid
import sunpy.map                                #for making maps from files
import astropy.units as u                       #for units
from scipy.ndimage.interpolation import shift   #shift arrays for grdients
import matplotlib.pyplot as plt                 #for plotting
from datetime import date                       #for dates

import matplotlib as mpl                        #for setting plot parameters
mpl.rcParams['figure.dpi'] = 300                #image quality of figures

#---- Functions ----#

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
    #give list as array
    files = np.asarray(files)
    #Return the list of FITS files
    return files

def compute_jz(tmap, pmap, mask=False, tr_cutoff=100):
    """
    Computes a map of vertical current density, jz, from maps of Bp and Bt.
    If mask=True, jz is set to 0 in pixels where the transverse magnetic field
    strength is weaker than a cutoff value, tr_cutoff.
    Note: Reversing the sign of tmap is handled in this function.
    """
    #hmi pixel = 0.5 arcsec = 0.03 deg. 1 arcsec at 1 AU = 725.27 km
    degree_to_km = ((0.5*u.arcsec)/(0.03*u.deg)) * ((725.27*u.km)/(1*u.arcsec))

    dx = (tmap.scale[0])*(1*u.pix) * degree_to_km #pizel width in km
    dy = (tmap.scale[1])*(1*u.pix) * degree_to_km #pixel height in km
    dby = (shift(-tmap.data, [0,-1], cval=np.NaN)-shift(-tmap.data, [0,1], cval=np.NaN)) * u.gauss
    dbx = (shift(pmap.data, [-1,0], cval=np.NaN)-shift(pmap.data, [1,0], cval=np.NaN)) * u.gauss
    npix = 2 #relative number of pixels shifted when getting dby and dbx (1--1=2)
    dbydx = dby/(npix*dx) #gradient of By in x
    dbxdy = dbx/(npix*dy) #gradient of Bx in y
    u_0 = (4*np.pi*1e-7)*(u.tesla*u.m/u.amp) #constant

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

    return jz

#--- Inputs ----#

#path to fits files
data_path = ('/Users/alexanderjames/anaconda3/alexpy/jz/20110215_0100_0200'
             '/data/')

#path to save outputs
out_path = ('/Users/alexanderjames/anaconda3/alexpy/jz/20110215_0100_0200'
            '/plots/')

plot_all = True                         #plot all fits files?
if plot_all == False:                   #if only plotting some fits files:
    start, end, every = 0, 1, 1             #which fits files to use

crop = False                            #crop maps?
x0,x1, y0,y1 = 150, 740, 0, 320         #pixels to crop

jz_sat = 0.03                           #saturation for Jz map. unit of Jz is A /m^2
br_sat = 2000                           #saturation for Br map. unit of Br = Gauss
jz_contour = 0.075                      #draw contours of Jz on Br. unit of Jz is A /m^2

save = True
show = False

#---- Code ----#

rfiles = find_fits(data_path, 'Br')   #find Br fits files in path
pfiles = find_fits(data_path, 'Bp')   #find Bp fits files in path
tfiles = find_fits(data_path, 'Bt')   #find Bt fits files in path

#if not plotting all fits files
if plot_all == False:
    #take a selection of fits files
    rfiles = pfiles[start:end:every]
    pfiles = pfiles[start:end:every]
    tfiles = tfiles[start:end:every]

for a in range(len(pfiles)):            #for all selected fits files:
    rfile, pfile, tfile = rfiles[a], pfiles[a], tfiles[a]     #take a file

    #map the radial, poloidal, and toroidal fields
    rmap = sunpy.map.Map(rfile)
    pmap = sunpy.map.Map(pfile)
    tmap = sunpy.map.Map(tfile)

    #crop maps
    if crop:
        rmap = rmap.submap((x0, y0)*u.pix, (x1, y1)*u.pix)
        pmap = pmap.submap((x0, y0)*u.pix, (x1, y1)*u.pix)
        tmap = tmap.submap((x0, y0)*u.pix, (x1, y1)*u.pix)

    #lazy attempt at arcsec conversion i.e. using corner coordinates only
##    bl_hp = rmap.bottom_left_coord.transform_to(frames.Helioprojective)
##    tr_hp = rmap.top_right_coord.transform_to(frames.Helioprojective)
##    hp_l, hp_r = bl_hp.Tx.value, tr_hp.Tx.value
##    hp_b, hp_t = bl_hp.Ty.value, tr_hp.Ty.value

    #compute Jz
    jz = compute_jz(tmap, pmap, mask=True, tr_cutoff=100)

    #begin figure
    fig = plt.figure(figsize=(6, 6))
    title = pmap.date.strftime('%d-%b-%Y %H:%M:%S')
    label = f'{title} UT'
    plt.suptitle(label)
    #plot Br
    ax1 = fig.add_subplot(211, projection=tmap)
    brplot = ax1.imshow(rmap.data, origin='lower', cmap='Greys_r', vmin=-br_sat, vmax=br_sat)#,
                        #extent=[hp_l,hp_r,hp_b,hp_t])
    ax1.contour(jz, levels=[-jz_contour,jz_contour],colors=['b','r'])
    cbar1 = plt.colorbar(brplot, extend='both')
    cbar1.ax.set_ylabel(f'Br ({u.gauss})')
    ax1.set_xlabel('X (degrees)')
    ax1.set_ylabel('Y (degrees)')
    #plot current
    ax2 = fig.add_subplot(212, projection=tmap)
    jplot = ax2.imshow(jz, origin='lower', cmap='bwr', vmin=-jz_sat, vmax=jz_sat)#,
                       #extent=[hp_l,hp_r,hp_b,hp_t])
    cbar2 = plt.colorbar(jplot, extend='both')
    cbar2.ax.set_ylabel(f'Jz ({jz.unit})')
    ax2.set_xlabel('X (degrees)')
    ax2.set_ylabel('Y (degrees)')
    #save and show
    file_date = pmap.date.strftime('%Y%m%d_%H%M%S')
    if save:
        plt.savefig(f'{out_path}jz_{file_date}.png')
    if show:
        plt.show()
