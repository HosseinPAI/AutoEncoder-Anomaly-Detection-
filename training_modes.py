# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:47:40 2020

@author: MrHossein
"""
import functions
from model import AD_auto_encoder
from torch.optim.lr_scheduler import StepLR
import time
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main(dataset_path, option, epoches, device, kwargs):
    if option == 1:
        print('Training Normal AutoEncoder Netwrok for 128*128 input image is Started.')
        train_data = functions.prepare_data(dataset_path, 'train', kwargs)

        start_time = time.time()
        torch.manual_seed(101)
        model = AD_auto_encoder.AutoEncoder(batch_normalization=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.97)
        loss = nn.MSELoss().to(device)
        train_loss = np.zeros((1, epoches))
        # train epochs
        for i in range(epoches):
            train_output, train_loss[0][i] = functions.train(model, train_data, optimizer, loss, i + 1, device)
            scheduler.step()
        # save the trained model
        torch.save(model.state_dict(), "Anomaly_Detection_without_BN.pt")
        end_time = time.time()
        print('Total Time for Training : {:.2f}'.format((end_time - start_time) / 60), 'minute')
        functions.loss_plot(train_loss[0], 'Loss_Basic_128.jpg', 'Training Loss For Basic AutoEncoder')

        # test
        main(dataset_path, 5, epoches, device, kwargs)

    elif option == 2:
        print('Training AUtoEncoder Network with Batch Normalization for 128*128 input image is Started.')
        train_data = functions.prepare_data(dataset_path, 'train', kwargs)

        start_time = time.time()
        torch.manual_seed(101)
        model = AD_auto_encoder.AutoEncoder(batch_normalization=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.97)
        loss = nn.MSELoss()
        train_loss = np.zeros((1, epoches))
        # train epochs
        for i in range(epoches):
            train_output, train_loss[0][i] = functions.train(model, train_data, optimizer, loss, i + 1, device)
            scheduler.step()
        # save the trained model
        torch.save(model.state_dict(), "Anomaly_Detection_BatchN.pt")

        end_time = time.time()
        print('Total Time for Training : {:.2f}'.format((end_time - start_time) / 60), 'minute')
        functions.loss_plot(train_loss[0], 'Loss_BatchN_128.jpg',
                            'Training Loss For AutoEncoder with Batch Normalization')
        # test
        main(dataset_path, 6, epoches, device, kwargs)

    elif option == 3:
        print('Training AUtoEncoder Network with Batch Normalization for 256*256 input image is Started.')
        train_data = functions.prepare_data(dataset_path, 'train', kwargs, 256)

        start_time = time.time()
        torch.manual_seed(101)
        model = AD_auto_encoder.AutoEncoder(batch_normalization=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.97)
        loss = nn.MSELoss()
        train_loss = np.zeros((1, epoches))
        # train epochs
        for i in range(epoches):
            train_output, train_loss[0][i] = functions.train(model, train_data, optimizer, loss, i + 1, device)
            scheduler.step()
        # save the trained model
        torch.save(model.state_dict(), "Anomaly_Detection_BatchN_256.pt")

        end_time = time.time()
        print('Total Time for Training : {:.2f}'.format((end_time - start_time) / 60), 'minute')
        functions.loss_plot(train_loss[0], 'Loss_Batch_256.jpg',
                            'Training Loss For AutoEncoder with Batch Normalization')

        # test
        main(dataset_path, 7, epoches, device, kwargs)

    elif option == 4:
        print('Training AutoEncoder Network with Batch Normalization for Augmented data is Started.')
        train_data = functions.prepare_data(dataset_path, 'train', kwargs, 256, augmentation=True)

        start_time = time.time()
        torch.manual_seed(101)
        model = AD_auto_encoder.AutoEncoder(batch_normalization=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.97)
        loss = nn.MSELoss()
        train_loss = np.zeros((1, epoches))
        # train epochs
        for i in range(epoches):
            train_output, train_loss[0][i] = functions.train(model, train_data, optimizer, loss, i + 1, device)
            scheduler.step()
        # save the trained model
        torch.save(model.state_dict(), "Anomaly_Detection_BatchN_256_Aug.pt")

        end_time = time.time()
        print('Total Time for Training : {:.2f}'.format((end_time - start_time) / 60), 'minute')
        functions.loss_plot(train_loss[0], 'Loss_Batch_Aug.jpg', 'Training Loss For AutoEncoder')

        # test
        main(dataset_path, 8, epoches, device, kwargs)

    elif option == 5:
        print('Testing Normal AutoEncoder Netwrok for 128*128 input image is started.')
        try:
            model = AD_auto_encoder.AutoEncoder(batch_normalization=False)
            model.load_state_dict(torch.load("Anomaly_Detection_without_BN.pt"))
            functions.test_process(dataset_path, model, 128, kwargs, option)
        except:
            print('There is no such a file to Create Network Model. Please First Train your Network.')
    elif option == 6:
        print('Testing AutoEncoder Netwrok with Batch Normaliztion for 128*128 input image is started.')
        try:
            model = AD_auto_encoder.AutoEncoder(batch_normalization=True)
            model.load_state_dict(torch.load("Anomaly_Detection_BatchN.pt"))
            functions.test_process(dataset_path, model, 128, kwargs, option)
        except:
            print('There is no such a file to Create Network Model. Please First Train your Network.')
    elif option == 7:
        print('Testing AutoEncoder Netwrok with Batch Normaliztion for 256*256 input image is started.')
        try:
            model = AD_auto_encoder.AutoEncoder(batch_normalization=True)
            model.load_state_dict(torch.load("Anomaly_Detection_BatchN_256.pt"))
            functions.test_process(dataset_path, model, 256, kwargs, option)
        except:
            print('There is no such a file to Create Network Model. Please First Train your Network.')
    elif option == 8:
        print('Testing AutoEncoder Netwrok with Batch Normaliztion for Aygmented data is started.')
        try:
            model = AD_auto_encoder.AutoEncoder(batch_normalization=True)
            model.load_state_dict(torch.load("Anomaly_Detection_BatchN_256_Aug.pt"))
            functions.test_process(dataset_path, model, 256, kwargs, option)
        except:
            print('There is no such a file to Create Network Model. Please First Train your Network.')
    else:
        print('You input number is wrong, Please Run the Program Agian and input the correct number.')
