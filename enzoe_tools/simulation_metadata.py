#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:37:27 2024

@author: brennan
"""

import numpy as np

from .cosmology_params import CosmologyParams
from .constants import H0_over_h, G, Mpc_cm
from .mesh_metadata import MeshMetadata

class SimulationMetadata:
    
    def __init__(self, filename):
        self.cosmology = CosmologyParams(filename)
        self.mesh = MeshMetadata(filename)
        self.init_units()
        
    def init_units(self):
        
        h  = self.cosmology['hubble_constant_now']
        Om = self.cosmology['omega_matter_now']
        zi = self.cosmology['initial_redshift']
        
        # Density: 1 code_denisty = rho_to_cgs * (1 + z)**3 [g/cm**3]
        # (Note - this comes from comments made in 
        # EnzoPhysicsCosmology::density_units)
        self.rho_to_cgs = ((3 * Om * H0_over_h**2) / (8 * np.pi * G)) * h**2

        # Length: 1 code_length = Li [CM Mpc/h] = Li / (1 + z) [Mpc/h] 
        #                       = length_to_cm / (1+z) [cm]
        Li = self.cosmology['comoving_box_size']
        self.length_to_cm = Mpc_cm * Li / h

        # Time: 1 code_time = time_to_s [s] 
        # (Note - from comments made in EnzoPhysicsCosmology::time_units)
        self.time_to_s = np.sqrt(2/3)
        self.time_to_s /= (H0_over_h * h * np.sqrt(Om * (1+zi)**3))

        # Velocity: 1 code_velocity = v_to_cgs [cm/s] 
        # (I think this is wrong and there's a factor of (1+current_redshift) 
        # missing in the enzoe code base - see velociy_units in 
        # EnzoPhysicsCosmology.hpp)
        self.v_to_cgs = 1.0e7 * Li * np.sqrt(3 * Om * (1 + zi) / 2)