#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:00:37 2024

@author: brennan
"""

from enzoe_tools import constants, simulation_metadata, block_tree

import time
import glob
import h5py

import numpy as np
import matplotlib.pyplot as plt

# Subgrid model thresholds
rho_threshold = 0.265


#%% Read in and set cosmology variables.
snapshot_dir = 'data/EnzoE-snapshot-example/'

param_file = glob.glob(snapshot_dir + '*.libconfig')[0]
md = simulation_metadata.SimulationMetadata(param_file)

param_file = glob.glob(snapshot_dir + '*.block_list')[0]
tree = block_tree.BlockTree(param_file)


#%%
def process_block(block, start, rho, nds, div, rH2, tff, tcl, data):
    pass


#%% Find and read HDF5 files

N_sites_per_block = md.mesh.get_bock_mesh_length()
N_leaf_blocks = tree.get_num_leaves()
N = N_sites_per_block**3 * N_leaf_blocks
ghost_depth = md.mesh['ghost_depth']

rho = np.zeros(N)
nds = np.zeros(N)
div = np.zeros(N)
rH2 = np.zeros(N)
tff = np.zeros(N)
tcl = np.zeros(N)
z = 0

ti_file = time.time()
ind = -1
for file in tree.files.keys():
    print('Opening file', file)
    data = h5py.File(snapshot_dir + file, 'r')
    
    blocks = tree.files[file]
    
    ti_blocks = time.time()
    for block in blocks:
        
        # Skip if block is not a leaf block.
        if not tree.is_leaf(block.name):
            continue
        
        process_block(block, start, rho, nds, div, rH2, tff, tcl, data)
        
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
                    
                    # Check if cell is overdense.
                    rho_i = density[i, j, k] * mass_unit
                    rho[ind] = rho_i
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
                    nds[ind] = mass_unit * ndens_times_mH / constants.mass_hydrogen
                       
                    # Check if cell has negative divergence.
                    div_v = 0
                    div_v += vx[i+1, j, k] - vx[i-1, j, k]
                    div_v += vy[i, j+1, k] - vy[i, j-1, k]
                    div_v += vz[i, j, k+1] - vz[i, j, k-1]
                    div[ind] = div_v * md.v_to_cgs / (cell_length * md.length_to_cm / (1 + z))
                        
                    # Check if cell is not Jeans stable.
                    tff[ind] = dyn_coeff * np.sqrt(1 / rho_i)
                    tcl[ind] = cooling_time[i, j, k] * md.time_to_s
                        
                    # Check if cell has a high enough H2 fraction.
                    rho_H2 = dH2I[i, j, k] * mass_unit
                    rH2[ind] = rho_H2
        
        # TODO: data.close()
    print('processing this file took', time.time() - ti_blocks)
        
#%%
# figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))

# axes[0].scatter(rho, rH2, alpha=0.2)
# axes[0].set_xlabel('Density (cgs)')
# axes[0].set_ylabel('H2 Density (cgs)')

# axes[1].scatter(tff, tcl, alpha=0.2)
# axes[1].set_xlabel('Free-fall time (cgs)')
# axes[1].set_ylabel('Cooling time (cgs)')

# axes[2].scatter(nds, div, alpha=0.2)
# axes[2].set_xlabel('Number Density (cgs)')
# axes[2].set_ylabel('Velocity Divergence (cgs)')

# figure.suptitle(f'Redshift {z}')
# figure.savefig('star_formation_scatter_plots.png', dpi=300)
