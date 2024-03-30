#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:29:15 2024

@author: brennan
"""

class CosmologyParams:
    
    def __init__(self, filename):
        
        cosmology_params = {'comoving_box_size'   : 0,
                            'final_redshift'      : 0,
                            'hubble_constant_now' : 0,
                            'initial_redshift'    : 0,
                            'max_expansion_rate'  : 0,
                            'omega_baryon_now'    : 0,
                            'omega_cdm_now'       : 0,
                            'omega_lambda_now'    : 0,
                            'omega_matter_now'    : 0}
        
        with open(filename, 'r') as params:
            for line in params.readlines():
                words = line.replace(' ', '').split('=')
                if words[0] in cosmology_params:
                    cosmology_params[words[0]] = float(words[-1].split(';')[0])
                    
        self.cosmology_params = cosmology_params
                    
    def __getitem__(self, key):
        return self.cosmology_params[key]