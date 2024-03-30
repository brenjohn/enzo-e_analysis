#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:39:12 2024

@author: brennan
"""

import glob
import h5py

import numpy as np

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
output = open('stars.txt', 'w')
output.write('================== stars found ==================\n')

files = glob.glob('*.h5')
for file in files:
    print('Opening file', file)
    data = h5py.File(file, 'r')
    
    block_names = data.keys()
    
    for block in block_names:
        
        # Skip if block is not a leaf block.
        if not (len(block) == 9):
            continue
        
        output.write(block + '\n')
        
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
        
        xi, yi, zi = block_data.attrs['enzo_GridStartIndex']
        xf, yf, zf = block_data.attrs['enzo_GridEndIndex']
        
        for i in range(xi, xf):
            for j in range(yi, yf):
                for k in range(zi, zf):
                    
                    overdense = False
                    converging = False
                    jeans_unstable = False
                    
                    # Check if cell is overdense.
                    rho = density[i, j, k] * rho_to_cgs * (1 + z)**3
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
                    number_density = rho / mean_particle_mass
                    if number_density > 100:
                        overdense = True
                       
                    # Check if cell has negative divergence.
                    div_v = 0
                    div_v += vx[i+1, j, k] - vx[i-1, j, k]
                    div_v += vy[i, j+1, k] - vy[i, j-1, k]
                    div_v += vz[i, j, k+1] - vz[i, j, k-1]
                    if div_v < 0:
                        converging = True
                        
                    # Check if cell is not Jeans stable.
                    dyn_time = np.sqrt(3 * np.pi / (32 * G * rho))
                    if dyn_time < cooling_time[i, j, k] * time_to_s:
                        jeans_unstable = True
                        
                    # Check if cell has a high enough H2 fraction.
                    rho_H2 = dH2I[i, j, k] * rho_to_cgs * (1 + z)**3
                    if rho_H2 / rho > 0.001:
                        H2_rich = True
                        
                    # If all conditions are satisfied, the cell forms a star.
                    if overdense and converging and jeans_unstable and H2_rich:
                        output.write(f'Star at {i}, {j}, {k} with '
                                     + f'rho = {rho}, div(v) = {div_v}\n')
                        

output.close()

#%%


# For H_0 in km/s/Mpc and G in cgs units
# 1 code_density =            3 * H_0^2 * Omega_m / (8 * pi * G) [CM g/cm^3]
#                = (1+z)**3 * 3 * H_0^2 * Omega_m / (8 * pi * G) [g/cm^3]
      
