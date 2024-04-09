#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:08:44 2024

@author: john
"""

class MeshMetadata:
    
    def __init__(self, filename):
        
        mesh_params = {}
        
        with open(filename, 'r') as params:
            for line in params.readlines():
                words = line.replace(' ', '').split('=')
                
                if words[0] == 'min_level':
                    mesh_params['min_level'] = int(words[-1].split(';')[0])
                    
                if words[0] == 'root_blocks':
                    mesh_params['root_blocks'] = int(line.split()[-2])
                    
                if words[0] == 'root_size':
                    mesh_params['root_size'] = int(line.split()[-2])
                    
                if words[0] == 'ghost_depth':
                    mesh_params['ghost_depth'] = int(words[-1].split(';')[0])
                    
        self.mesh_params = mesh_params
        
        
    def __getitem__(self, key):
        return self.mesh_params[key]
    
    def get_bock_mesh_length(self):
        root_length = self.mesh_params['root_size'] 
        block_length = root_length // self.mesh_params['root_blocks']
        return block_length