# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:53:58 2020

@author: MrHossein
"""
import training_modes
import torch

if __name__ == '__main__':
    no_cuda = False
    print('==================================================================================')
    print('> Menu: Please Choose one of the below options to run the program ')
    print('==================================================================================')
    print('1: Training Normal AutoEncoder Network with 128*128 input image.')
    print('2: Training AutoEncoder Network with Batch Normalization for 128*128 input image.')
    print('3: Training AutoEncoder Network with Batch Normalization for 256*256 input image.')
    print('4: Training AutoEncoder Network with Batch Normalization for Augmented Data.')
    print('9: Exit.')
    print('==================================================================================')
    print('Please Choose The Number : ')
    option = int(input())
    if option == 9:
        no_cuda = True
    else:
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        dataset_path = '../hazelnut'
        epoches = 200
        training_modes.main(dataset_path, option, epoches, device, kwargs)
