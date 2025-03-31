# -*- coding: utf-8 -*-
import os
import pickle
import random
import torch
import torch_directml
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Train_Set_Gen():
    def __init__(self,params_range,num_of_points_k,num_of_points_w, img_range,bands_number = 2):
        self.bands_number = bands_number
        self.params_range = params_range
        self.img_range = img_range
        self.num_of_points_k = num_of_points_k
        self.num_of_points_w = num_of_points_w

    ##########################################################################
    def multiband_spectra(self):
        x_start, x_end = self.img_range['x']
        y_start_val, y_end_val = self.img_range['y']
        W = np.linspace(y_start_val, y_end_val, num = self.num_of_points_w)
        K = np.linspace(x_start , x_end, num = self.num_of_points_k)
        k,w = np.meshgrid(K,W)
        spectra = np.zeros((self.num_of_points_w,self.num_of_points_k))
        alpha = [random.uniform(self.params_range['alpha'][0],self.params_range['alpha'][1]) for _ in range(self.bands_number)]
        Lambda =[random.uniform(self.params_range['lam'][0],self.params_range['lam'][1]) for _ in range(self.bands_number)]
        m  = [random.uniform(self.params_range['m'][0],self.params_range['m'][1]) for _ in range(self.bands_number)]
        l = [random.uniform(self.params_range['l'][0],self.params_range['l'][1])  for _ in range(self.bands_number)]
        Imp = random.uniform(self.params_range['Imp'][0],self.params_range['Imp'][1]) # розсіяння на домішках
        x_step = abs((x_start - x_end)/self.num_of_points_k)
        y_step = abs((y_start_val - y_end_val)/self.num_of_points_w)

        shirley = random.uniform(0.1,0.2)
        #shirley = random.uniform(0.1,3)
        shirley_noise =  shirley* w**2
        for i in range(self.bands_number):
            m_i = m[i]
            l_i = l[i]
            Lam_i = Lambda[i]
            alpha_i = alpha[i]  #
            sigm2= alpha_i *( w ** 2) + Imp
            arr =   (0.5 * sigm2/( ((1-Lam_i)*w - (m_i *k**2 + l_i) ) ** 2  + (sigm2 ) ** 2)) #-((sigm2 )
            spectra += arr

        disp = np.full((self.num_of_points_w,self.num_of_points_k), 1.)
        for i in range(self.bands_number):
            m_i = m[i]
            l_i = l[i]
            Lam_i = Lambda[i]
            alpha_i = alpha[i]
            for mm in range(self.num_of_points_w):
                for nn in range(self.num_of_points_k):
                    w = mm * y_step + y_start_val
                    k = nn * x_step + x_start
                    if ((1-Lam_i)*w - (m_i *k**2 + l_i))**2 < 0.005:
                        disp[mm,nn] = 0
#                        if disp[mm,nn] == 0:
#                            disp[mm,nn] = 1
#                        else:
#                            disp[mm,nn] = 0

        #arr -=  shirley_noise
        #spectra_noise = spectra + shirley_noise
        spectra_noise = spectra + np.random.random(spectra.shape) * spectra + shirley_noise
        #spectra_noise = (spectra_noise-np.min(spectra_noise))/(np.max(spectra_noise)-np.min(spectra_noise))
        #disp[spectra < 0.0005] = 1
        return spectra_noise, disp

    ##########################################################################
    def create_train_data(self,set_len):
        x_train = []
        y_train = []
        for _ in range(set_len):
                x,y = self.multiband_spectra()
                x_train.append(x)
                y_train.append(y)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train,(-1,self.num_of_points_w,self.num_of_points_k,1))
        y_train = np.reshape(y_train,(-1,self.num_of_points_w,self.num_of_points_k,1))
        y_train = to_categorical(y_train)
        return (x_train,y_train)

class Fermi_level(Train_Set_Gen):
    def multiband_spectra(self):
        x_start, x_end = self.img_range['x']
        y_start_val, y_end_val = self.img_range['y']
        spectra_noise, disp = super().multiband_spectra()
        W = np.linspace(y_start_val, y_end_val, num = self.num_of_points_w)
        K = np.linspace(x_start , x_end, num = self.num_of_points_k)
        k,w = np.meshgrid(K,W)
        kb = 8.617 *1.E-5
        T = random.uniform(0.5,300)
        Ef = random.uniform(0,0.25)
        #print(T,Ef)
        exp = 1./(np.exp((w-Ef)/(kb*T)) + 1)

        res = spectra_noise*exp
        eps = 1e-7
        std_val = np.std(res)
        spectra_noise = (res - np.mean(res)) / max(std_val, eps)
        outputs = torch.sigmoid(torch.from_numpy(spectra_noise))

        return outputs.detach().cpu().numpy(), np.round(disp*exp)

##########################################################################
## MAIN
##########################################################################    
"""# Data Generation:"""
params_range = {'lam':(2.1,7.5),'m':(-4,14),'l':(-0.2,1.6),
'alpha':(0.1,4.),'Imp' :(0.,2), 'noise_level':(0.,0.1)}

img_range= {'x':(-0.4, 0.4),'y':(-0.7,0.) }

fermi = Fermi_level(params_range,128,128,img_range,bands_number = 2)

x_train, y_train = fermi.create_train_data(2000)

x_val, y_val = fermi.create_train_data(200)



with open('INPUT_data.pkl', 'wb') as f:  # open a text file    
        pickle.dump((x_train, y_train, x_val, y_val), f) # serialize the list



# Plot the first 10 images from x_train y_train
fig, axes = plt.subplots(2, 10, figsize=(15, 5))

for i in range(10):
        axes[0][i].imshow(x_train[i, :, :, 0], cmap='gray_r')  # Assuming single-channel images
        axes[0][i].axis('off')
        axes[0][i].set_title(f"Image {i+1}")
        axes[1][i].imshow(y_train[i, :, :, 0], cmap='gray_r')  # Assuming single-channel images
        axes[1][i].axis('off')
        axes[1][i].set_title(f"Image {i+1}")

plt.tight_layout()
plt.show()

# Plot the second 10 images from x_train y_train
fig, axes = plt.subplots(2, 10, figsize=(15, 5))

for i in range(10):
        axes[0][i].imshow(x_train[10+i, :, :, 0], cmap='gray_r')  # Assuming single-channel images
        axes[0][i].axis('off')
        axes[0][i].set_title(f"Image {i+1}")
        axes[1][i].imshow(y_train[10+i, :, :, 0], cmap='gray_r')  # Assuming single-channel images
        axes[1][i].axis('off')
        axes[1][i].set_title(f"Image {i+1}")

plt.tight_layout()
plt.show()

