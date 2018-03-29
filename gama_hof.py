# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:17:03 2018

@author: s105307
"""

class HallOfFame(object):
    
    def __init__(self, filename):
        self._filename = filename
        self._pop = []
        
    def update(self, pop):
        self._pop += pop
        
        with open(self._filename,'a') as fh:
            #print('-gen-')
            #print('\n'.join([str((str(ind), ind.fitness.values[0])) for ind in pop]))
            fh.write('\n'.join([str((str(ind), ind.fitness.values[0])) for ind in pop]))