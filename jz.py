# Jz.py
# by alexanderjames
#https://github.com/alexjsolar/Jz

#---- Imports ----#

import sunpy.map                                #for making maps from files
import matplotlib.pyplot as plt                 #for plotting
from datetime import date                       #for dates

from functions_jz import *                      #import functions from module
# import os
# from os import listdir                        #for finding input files
# from os.path import isfile, join              #for reading input files
# import numpy as np                            #for 'np.linspace' plotting grid
# import astropy.units as u                     #for units
# from scipy.ndimage.interpolation import shift #shift arrays for gradients

import matplotlib as mpl                        #for setting plot parameters
mpl.rcParams['figure.dpi'] = 300                #image quality of figures

#--- Inputs ----#

#path to fits files
data_path = ('/Users/alexanderjames/anaconda3/alexpy/jz/events'
             '/20120614_1300/data/')

#path to save outputs
out_path = ('/Users/alexanderjames/anaconda3/alexpy/jz/events'
            '/20120614_1300/plots/')

plot_all = True                         #plot all fits files?
if plot_all == False:                   #if only plotting some fits files:
    start, end, every = 0, 1, 1             #which fits files to use

crop = False                            #crop maps?
x0,x1, y0,y1 = 150, 740, 0, 320         #pixels to crop

jz_sat = 0.03                           #saturation for Jz map. unit of Jz is A /m^2
br_sat = 2000                           #saturation for Br map. unit of Br = Gauss
jz_contour = 0.075                      #draw contours of Jz on Br. unit of Jz is A /m^2

save = True                             #save plots?
show = True                             #show plots?

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
        check_make_dir(out_path)
        plt.savefig(f'{out_path}jz_{file_date}.png')
    if show:
        plt.show()
