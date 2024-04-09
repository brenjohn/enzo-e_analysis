#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:54:12 2024

@author: brennan
"""

class BlockTree:
    
    def __init__(self, filename):
        
        self.blocks = {}
        self.files = {}
        
        with open(filename) as blocklist:
            
            for line in blocklist.readlines():
                block_enzo_name, block_file = line.split()
                block = Block(block_enzo_name, block_file)
                self.blocks[block.name] = block
                
                if block_file in self.files:
                    self.files[block_file].append(block)
                else:
                    self.files[block_file] = [block]
                
        
    def is_leaf(self, block_name):
        block = self.blocks[block_name]
        return not block.children[0] in self.blocks
    
    def get_leaves(self):
        return [block 
                for name, block in self.blocks.items() if self.is_leaf(name)]
    
    def get_num_leaves(self):
        return len(self.get_leaves())
                
                
class Block:
    
    def __init__(self, enzo_name, file_name):
        self.name, self.parent, self.children = self.block_details(enzo_name)
        self.file = file_name
        
    def block_details(self, block):
        name = block[1:].replace(':', '')
        parent = self.get_parent(name)
        children = self.get_children(name)
        
        return name, parent, children
        
    def get_parent(self, name):
        coords = name.split('_')
        parent_coords = [c[:-1] for c in coords]
        parent_name = '_'.join(parent_coords)
        return parent_name if parent_name != name else None
    
    def get_children(self, name):
        coords = name.split('_')
        children = 8 * [None]
        
        for n, child in enumerate((('0', '0', '0'),
                                   ('0', '0', '1'),
                                   ('0', '1', '0'),
                                   ('0', '1', '1'),
                                   ('1', '0', '0'),
                                   ('1', '0', '1'),
                                   ('1', '1', '0'),
                                   ('1', '1', '1'))):
            child_coords = [ci + cci for ci, cci in zip(coords, child)]
            children[n] = '_'.join(child_coords)
            
        return children
    
    def enzo_name(self):
        return 'B' + self.name
    
#%%
# import glob

# param_file = glob.glob('../data/EnzoE-snapshot-example/*.block_list')[0]
# tree = BlockTree(param_file)