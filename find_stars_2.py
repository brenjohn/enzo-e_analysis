#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:36:51 2024

@author: brennan
"""

from enzoe_tools import constants, simulation_metadata, block_tree

import time
import glob
import h5py

import numpy as np

#%%
def process_block(block, ind, output):
    
    # Skip if block is not a leaf block.
    if not md.tree.is_leaf(block.name):
        return ind
    
    block_data   = data[block.enzo_name()]
    density      = np.array(block_data['field_density'])
    vx           = np.array(block_data['field_velocity_x'])
    vy           = np.array(block_data['field_velocity_y'])
    vz           = np.array(block_data['field_velocity_z'])
    cooling_time = np.array(block_data['field_cooling_time'])
    d_el         = np.array(block_data['field_e_density'])
    dHI          = np.array(block_data['field_HI_density'])
    dHII         = np.array(block_data['field_HII_density'])
    dHeI         = np.array(block_data['field_HeI_density'])
    dHeII        = np.array(block_data['field_HeII_density'])
    dHeIII       = np.array(block_data['field_HeIII_density'])
    dHM          = np.array(block_data['field_HM_density'])
    dH2I         = np.array(block_data['field_H2I_density'])
    dH2II        = np.array(block_data['field_H2II_density'])
    z            = block_data.attrs['enzo_redshift'][0]
    cell_length  = block_data.attrs['enzo_CellWidth'][0]
    
    xi, yi, zi = block_data.attrs['enzo_GridStartIndex']
    xf, yf, zf = block_data.attrs['enzo_GridEndIndex']
    
    mass_unit = md.rho_to_cgs * (1 + z)**3
    dyn_coeff = np.sqrt(3 * np.pi / (32 * constants.G))
    
    for i in range(xi, xf + 1):
        for j in range(yi, yf + 1):
            for k in range(zi, zf + 1):
                ind += 1
                
                overdense = False
                converging = False
                jeans_unstable = False
                H2_rich = False
                
                # Check if cell is overdense.
                rho_i = density[i, j, k] * mass_unit
                ndens_times_mH = d_el[i, j, k]       + \
                                 dHI[i, j, k]        + \
                                 dHII[i, j, k]       + \
                                 dHeI[i, j, k]   * 0.25 + \
                                 dHeII[i, j, k]  * 0.25 + \
                                 dHeIII[i, j, k] * 0.25 + \
                                 dHM[i, j, k]        + \
                                 dH2I[i, j, k]   * 0.5 + \
                                 dH2II[i, j, k]  * 0.5
                # mu = density[i, j, k] / ndens_times_mH
                # mean_particle_mass = mu * constants.mass_hydrogen
                # number_density = rho_i / mean_particle_mass
                # nds[ind] = number_density
                nds = mass_unit * ndens_times_mH / constants.mass_hydrogen
                if nds > 100:
                    overdense = True
                   
                # Check if cell has negative divergence.
                div_v = 0
                div_v += vx[i+1, j, k] - vx[i-1, j, k]
                div_v += vy[i, j+1, k] - vy[i, j-1, k]
                div_v += vz[i, j, k+1] - vz[i, j, k-1]
                div = div_v * md.v_to_cgs / (cell_length * md.length_to_cm / (1 + z))
                if div < 0:
                    converging = True
                    
                # Check if cell is not Jeans stable.
                tff = dyn_coeff * np.sqrt(1 / rho_i)
                tcl = cooling_time[i, j, k] * md.time_to_s
                if tff < tcl:
                    jeans_unstable = True
                    
                # Check if cell has a high enough H2 fraction.
                rho_H2 = dH2I[i, j, k] * mass_unit
                if rho_H2 / rho_i > 0.001:
                    H2_rich = True
                
                # If all conditions are satisfied, the cell forms a star.
                if overdense and converging and jeans_unstable and H2_rich:
                    message = f'Star in {block.enzo_name()} at {i}, {j}, {k}'
                    message += ' with: '
                    message += f'tff = {tff}, tcl = {tcl}, '
                    message += f'nds = {nds}, rho_H2 = {rho_H2}, '
                    message += f'rho = {rho_i}, div(v) = {div_v}\n'
                    output.write(message)
                    
    return ind


#%% Read in and set cosmology variables.
snapshot_dir = 'data/EnzoE-snapshot-example/'
param_file = glob.glob(snapshot_dir + '*.libconfig')[0]
block_list = glob.glob(snapshot_dir + '*.block_list')[0]
md = simulation_metadata.SimulationMetadata(param_file, block_list)


#%% Find and read HDF5 files

with open('stars.txt', 'w') as output:
    
    ti_file = time.time()
    ind = -1
    for file in md.tree.files.keys():
        print('Opening file', file)
        data = h5py.File(snapshot_dir + file, 'r')
        
        blocks = md.tree.files[file]
        
        ti_blocks = time.time()
        for block in blocks:
            ind = process_block(block, ind, output)
        
        data.close()
        print('processing this file took', time.time() - ti_blocks)