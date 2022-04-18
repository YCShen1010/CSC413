#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 07:42:03 2018

@author: manoj
"""

import h5py
import numpy as np
import os

def main(file1,file2,target):
       
    
    file1_txt = file1.replace('.hdf5','.txt')
    file2_txt = file2.replace('.hdf5','.txt')

    x = np.loadtxt(file1_txt,delimiter='\n',dtype=str)
    file1_files =x.tolist()
    #x = np.loadtxt('val2014.txt',delimiter='\n',dtype=str)
    #train=x.tolist()
    x = np.loadtxt(file2_txt,delimiter='\n',dtype=str)
    file2_files=x.tolist() 
    
    
    mode = 'both'
    path_hdf5 = target
    hdf5_file = h5py.File(path_hdf5, 'w')


    N1 = len(file1_files)
    N2 = len(file2_files)
    
    nb_images = N1 + N2
    
    if mode == 'both' or mode == 'att':
        shape_att = (nb_images, 2048, 14, 14)
        hdf5_att = hdf5_file.create_dataset('att', shape_att,
                                            dtype='f')#, compression='gzip')
    if mode == 'both' or mode == 'noatt':
        shape_noatt = (nb_images, 2048)
        hdf5_noatt = hdf5_file.create_dataset('noatt', shape_noatt,
                                              dtype='f')#, compression='gzip')


    print ("Reading file1 .....")
    file1_hdf5 = h5py.File(file1,'r')   
             
    for idx in range(N1):
        read_att  =  file1_hdf5['att']
        read_noatt = file1_hdf5['noatt']

        if mode == 'both' or mode == 'att':
            hdf5_att[idx]   = read_att[idx]
        if mode == 'both' or mode == 'noatt':
            hdf5_noatt[idx] = read_noatt[idx]
            
        if idx % 10 == 0:
            print('Extract: [{0}/{1}]\t'.format(idx,nb_images))

    print ("Reading file2 .....")
    file2_hdf5 = h5py.File(file2,'r')           
    for idx in range(N1,N1 + N2):
        data_att  =  file2_hdf5['att']
        data_noatt = file2_hdf5['noatt']

        if mode == 'both' or mode == 'att':
            hdf5_att[idx]   = data_att[idx - N1]
        if mode == 'both' or mode == 'noatt':
            hdf5_noatt[idx] = data_noatt[idx -N1]       
        if idx % 10 == 0:
            print('Extract: [{0}/{1}]\t'.format(idx, nb_images))

    print ("Appending done .....")            
    hdf5_file.close()


if __name__ == '__main__':
    

    file1a = './hr.h5' 
    file1b = './lr/h5'     
    main(file1a,file1b,target='DIV2K_valid_LR_difficult.h5')
    # main(file1b,file2,target='val2014_vg.hdf5')
    
    