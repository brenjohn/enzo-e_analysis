#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:00:37 2024

@author: brennan
"""

import glob
import h5py

import numpy as np
import matplotlib.pyplot as plt

# Subgrid model thresholds
rho_threshold = 0.265


#%% Read in and set cosmology variables.
cosmology_params = {'comoving_box_size'   : 0,
                    'final_redshift'      : 0,
                    'hubble_constant_now' : 0,
                    'initial_redshift'    : 0,
                    'max_expansion_rate'  : 0,
                    'omega_baryon_now'    : 0,
                    'omega_cdm_now'       : 0,
                    'omega_lambda_now'    : 0,
                    'omega_matter_now'    : 0}

param_file = glob.glob('*.libconfig')[0]
with open(param_file, 'r') as params:
    for line in params.readlines():
        words = line.replace(' ', '').split('=')
        if words[0] in cosmology_params:
            cosmology_params[words[0]] = float(words[-1].split(';')[0])
            
h  = cosmology_params['hubble_constant_now']
Om = cosmology_params['omega_matter_now']
zi = cosmology_params['initial_redshift']


#%% Constants
G         = 6.67430e-8            # Gravitational constant in cgs
Mpc_cm    = 3.0856775809623245e24 # 1Mpc in cm
H0_over_h = 1.0e7 / Mpc_cm        # H0/h = 100(km/s)(1/Mpc) = (1e7cm)/(3e24cm)/s
mass_hydrogen = 1.67262171e-24    # mass of hydrogen in cgs


#%% Conversion factors
# Density: 1 code_denisty = rho_to_cgs * (1 + z)**3 [g/cm**3]
# (Note - this comes from comments made in EnzoPhysicsCosmology::density_units)
rho_to_cgs = ((3 * Om * H0_over_h**2) / (8 * np.pi * G)) * h**2

# Length: 1 code_length = Li [CM Mpc/h] = Li / (1 + z) [Mpc/h] = length_to_cm / (1+z) [cm]
Li = cosmology_params['comoving_box_size']
length_to_cm = Mpc_cm * Li / h

# Time: 1 code_time = time_to_s [s] 
# (Note - from comments made in EnzoPhysicsCosmology::time_units)
time_to_s = np.sqrt(2/3) / (H0_over_h * h * np.sqrt(Om * (1+zi)**3))

# Velocity: 1 code_velocity = v_to_cgs [cm/s] 
# (I think this is wrong and there's a factor of (1+current_redshift) missing
# in the enzoe code base - see velociy_units in EnzoPhysicsCosmology.hpp)
v_to_cgs = 1.0e7 * Li * np.sqrt(3 * Om * (1 + zi) / 2)


#%% Find and read HDF5 files

rho = np.zeros(0)
nds = np.zeros(0)
div = np.zeros(0)
rH2 = np.zeros(0)
tff = np.zeros(0)
tcl = np.zeros(0)
z = 0

files = glob.glob('*.h5')
for file in files:
    print('Opening file', file)
    data = h5py.File(file, 'r')
    
    block_names = data.keys()
    
    for block in block_names:
        
        # Skip if block is not a leaf block.
        if not (len(block) == 9):
            continue
        
        block_data   = data[block]
        density      = block_data['field_density']
        vx           = block_data['field_velocity_x']
        vy           = block_data['field_velocity_y']
        vz           = block_data['field_velocity_z']
        cooling_time = block_data['field_cooling_time']
        d_el         = block_data['field_e_density']
        dHI          = block_data['field_HI_density']
        dHII         = block_data['field_HII_density']
        dHeI         = block_data['field_HeI_density']
        dHeII        = block_data['field_HeII_density']
        dHeIII       = block_data['field_HeIII_density']
        dHM          = block_data['field_HM_density']
        dH2I         = block_data['field_H2I_density']
        dH2II        = block_data['field_H2II_density']
        z            = block_data.attrs['enzo_redshift'][0]
        cell_length  = block_data.attrs['enzo_CellWidth'][0]
        
        xi, yi, zi = block_data.attrs['enzo_GridStartIndex']
        xf, yf, zf = block_data.attrs['enzo_GridEndIndex']
        
        N_cells = (xf - xi) * (yf - yi) * (zf -zi)
        rho_block = np.zeros(N_cells)
        nds_block = np.zeros(N_cells)
        div_block = np.zeros(N_cells)
        rH2_block = np.zeros(N_cells)
        tff_block = np.zeros(N_cells)
        tcl_block = np.zeros(N_cells)
        
        ind = -1
        for i in range(xi, xf):
            for j in range(yi, yf):
                for k in range(zi, zf):
                    ind += 1
                    
                    overdense = False
                    converging = False
                    jeans_unstable = False
                    
                    # Check if cell is overdense.
                    rho_i = density[i, j, k] * rho_to_cgs * (1 + z)**3
                    rho_block[ind] = rho_i
                    ndens_times_mH = d_el[i, j, k]       + \
                                     dHI[i, j, k]        + \
                                     dHII[i, j, k]       + \
                                     dHeI[i, j, k]   / 4 + \
                                     dHeII[i, j, k]  / 4 + \
                                     dHeIII[i, j, k] / 4 + \
                                     dHM[i, j, k]        + \
                                     dH2I[i, j, k]   / 2 + \
                                     dH2II[i, j, k]  / 2
                    mu = density[i, j, k] / ndens_times_mH
                    mean_particle_mass = mu * mass_hydrogen
                    number_density = rho_i / mean_particle_mass
                    nds_block[ind] = number_density
                       
                    # Check if cell has negative divergence.
                    div_v = 0
                    div_v += vx[i+1, j, k] - vx[i-1, j, k]
                    div_v += vy[i, j+1, k] - vy[i, j-1, k]
                    div_v += vz[i, j, k+1] - vz[i, j, k-1]
                    div_block[ind] = div_v * v_to_cgs / (cell_length * length_to_cm / (1 + z))
                        
                    # Check if cell is not Jeans stable.
                    dyn_time = np.sqrt(3 * np.pi / (32 * G * rho_i))
                    tff_block[ind] = dyn_time
                    tcl_block[ind] = cooling_time[i, j, k] * time_to_s
                        
                    # Check if cell has a high enough H2 fraction.
                    rho_H2 = dH2I[i, j, k] * rho_to_cgs * (1 + z)**3
                    rH2_block[ind] = rho_H2
             
        # TODO: It would be better to preallocate all the memory I need for
        # this array. Should add a step to the start of the script to work out
        # how many cells there are or read it directly from param file like
        # the cosmology params at the beginning.
        rho = np.concatenate((rho, rho_block))
        nds = np.concatenate((nds, nds_block))
        div = np.concatenate((div, div_block))
        rH2 = np.concatenate((rH2, rH2_block))
        tff = np.concatenate((tff, tff_block))
        tcl = np.concatenate((tcl, tcl_block))
        
        # TODO: data.close()
        
#%%
figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))

axes[0].scatter(rho, rH2, alpha=0.2)
axes[0].set_xlabel('Density (cgs)')
axes[0].set_ylabel('H2 Density (cgs)')

axes[1].scatter(tff, tcl, alpha=0.2)
axes[1].set_xlabel('Free-fall time (cgs)')
axes[1].set_ylabel('Cooling time (cgs)')

axes[2].scatter(nds, div, alpha=0.2)
axes[2].set_xlabel('Number Density (cgs)')
axes[2].set_ylabel('Velocity Divergence (cgs)')

figure.suptitle(f'Redshift {z}')
figure.savefig('star_formation_scatter_plots.png', dpi=300)
