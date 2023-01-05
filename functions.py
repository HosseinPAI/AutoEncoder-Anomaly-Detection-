# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:44:51 2020

@author: MrHossein
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


def prepare_data(dataset_path, data_type, args, image_size=128, augmentation=False):
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    data = datasets.ImageFolder(os.path.join(dataset_path, data_type), transform=data_transform)
    if data_type == 'train':
        if augmentation:
            aug_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
            ])
            aug_data = datasets.ImageFolder(os.path.join(dataset_path, data_type), transform=aug_transform)
            torch.manual_seed(42)
            data_loader = DataLoader(data + aug_data, batch_size=32, shuffle=True, **args)
        else:
            torch.manual_seed(42)
            data_loader = DataLoader(data, batch_size=32, shuffle=True, **args)
    else:
        data_loader = DataLoader(data, batch_size=1, shuffle=False)

    return data_loader


def train(model, train_data, optimizer, loss, epoch, device):
    total_loss = 0
    output_log = []
    print('Training Epoch: {}'.format(epoch))
    model.train()
    for batch_idx, (images, label) in enumerate(train_data):
        print('.', end='')
        data = images.to(device)
        optimizer.zero_grad()
        output = model(data)
        output_log.append(output)
        loss_out = loss(output, data)
        loss_out.backward()
        optimizer.step()
        total_loss += loss_out.item()

    print('\tTrain Loss: {:.6f}'.format(total_loss))
    return output_log, total_loss


def valid(model, valid_data, loss, device):
    total_loss = 0
    output_log = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, label) in enumerate(valid_data):
            data = images.to(device)
            output = model(data)
            output_log.append(output)
            loss_out = loss(output, data)
            total_loss += loss_out.item()

    print('\t\t    Valid Loss: {:.6f}\n'.format(total_loss))
    return output_log, total_loss


def test(model, test_data):
    net_output = []

    model.eval()
    for batch_idx, (images, label) in enumerate(test_data):
        data = images
        output = model(data)
        net_output.append(output)

    return net_output


def loss_plot(data1, save_file, title):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.title(title, color='darkblue')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.plot(data1)
    plt.savefig(save_file, dpi=200)
    plt.show()


def Loss_Accuracy_Test(dataset_path, model, image_size, args, option, threshold=[0.09, 0.12, 0.1, 0.07]):
    ground_truth_data = prepare_data(dataset_path, 'ground_truth', args, image_size, augmentation=False)
    ground_image = []
    for batch_idx, (images, label) in enumerate(ground_truth_data):
        ground_image.append(images)

    test_data = prepare_data(dataset_path, 'test', args, image_size, augmentation=False)
    test_image = []
    for batch_idx, (images, label) in enumerate(test_data):
        if label != 2:
            test_image.append(images)

    test_output = []
    test_output = test(model, test_data)

    if option == 3 or option == 7 or option == 4 or option == 8:
        bound = [120, 45, 45, 55]
    else:
        bound = [50, 25, 30, 40]

    error = []
    b_image = []
    for i in range(len(test_image)):
        if i < 17:
            thr = threshold[0]
        elif i < 35:
            thr = threshold[1]
        elif i < 53:
            thr = threshold[2]
        else:
            thr = threshold[3]

        original_image = torch.tensor(test_image[i][0])
        reconst_image = torch.tensor(test_output[i][0])
        diff = np.abs(original_image - reconst_image)

        RGB_image = np.transpose(diff.numpy(), (1, 2, 0))
        Gray_image = color.rgb2gray(RGB_image)
        Binary_image = 1.0 * (Gray_image > thr)
        b_image.append(Binary_image)

        gound_truth_image = torch.tensor(ground_image[i][0])
        diff = np.abs(gound_truth_image - Binary_image)
        error.append(np.linalg.norm(diff))

    correct = 0
    for i in range(len(error)):
        if i < 17:
            if error[i] < bound[0]:
                correct += 1
        elif i < 35:
            if error[i] < bound[1]:
                correct += 1
        elif i < 53:
            if error[i] < bound[2]:
                correct += 1
        else:
            if error[i] < bound[3]:
                correct += 1

    return error, (correct / len(error)), b_image, ground_image, test_image, test_output


def show_plot(indices, test_image, output_net_image, ground_truth_image, binary_image):
    for i in range(4):
        plt.figure(figsize=(20, 5))
        plt.subplot(151)
        plt.title('Original Hazelnut Image')
        plt.imshow(np.transpose(torch.tensor(test_image[indices[i]][0]), (1, 2, 0)))
        plt.subplot(152)
        plt.title('Reconstruct Image')
        plt.imshow(np.transpose(torch.tensor(output_net_image[indices[i]][0]), (1, 2, 0)))
        plt.subplot(153)
        plt.title('Differntial Image')
        plt.imshow(
            np.transpose(np.abs(torch.tensor(test_image[indices[i]][0] - output_net_image[indices[i]][0])), (1, 2, 0)))
        plt.subplot(154)
        plt.title('Binerized Image')
        plt.imshow(binary_image[indices[i]], cmap=plt.cm.gray)
        plt.subplot(155)
        plt.title('Ground Truth Image')
        plt.imshow(np.transpose(torch.tensor(ground_truth_image[indices[i]][0]), (1, 2, 0)))
        plt.savefig('image_output_{}.jpg'.format(i), dpi=200)
        plt.show()


def test_process(dataset_path, model, image_size, args, option):
    if option == 1 or option == 5 or option == 2 or option == 6:
        threshold = [0.07, 0.08, 0.15, 0.18]
    elif option == 3 or option == 7:
        threshold = [0.08, 0.14, 0.20, 0.22]
    else:
        threshold = [0.11, 0.14, 0.24, 0.27]

    error, accuracy, binary_image, ground_truth_image, test_image, output_net_image = Loss_Accuracy_Test(dataset_path,
                                                                                                         model,
                                                                                                         image_size,
                                                                                                         args, option,
                                                                                                         threshold)
    print('The Accuracy of Network on Test Data = {:.2f}'.format(accuracy))
    show_plot([8, 31, 48, 62], test_image, output_net_image, ground_truth_image, binary_image)
    # for_test(binary_image, error)


def for_test(bin_img, error):
    index = 0
    plt.figure(figsize=(20, 20))
    idx = 1
    for i in range(4):
        for j in range(5):
            plt.subplot(4, 5, idx)
            plt.title('error = {:.2f}'.format(error[index]))
            plt.imshow(bin_img[index], cmap=plt.cm.gray)
            idx += 1
            index += 1
            if index == 18:
                break
    plt.show()

    plt.figure(figsize=(20, 20))
    idx = 1
    for i in range(4):
        for j in range(5):
            plt.subplot(4, 5, idx)
            idx += 1
            plt.title('error = {:.2f}'.format(error[index]))
            plt.imshow(bin_img[index], cmap=plt.cm.gray)
            index += 1
            if index == 35:
                break

    plt.show()

    plt.figure(figsize=(20, 20))
    idx = 1
    for i in range(4):
        for j in range(5):
            plt.subplot(4, 5, idx)
            idx += 1
            plt.title('error = {:.2f}'.format(error[index]))
            plt.imshow(bin_img[index], cmap=plt.cm.gray)
            index += 1
            if index == 53:
                break
    plt.show()

    plt.figure(figsize=(20, 20))
    idx = 1
    for i in range(4):
        for j in range(5):
            plt.subplot(4, 5, idx)
            idx += 1
            plt.title('error = {:.2f}'.format(error[index]))
            plt.imshow(bin_img[index], cmap=plt.cm.gray)
            index += 1
            if index == 70:
                break
    plt.show()