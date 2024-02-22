import copy
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
from math import *
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchmetrics import Accuracy, Recall, F1Score

import lightning as pl # Pytorch lightning is a wrapper for pytorch that makes it easier to train models
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import os

NUM_CLASSES = 6
device = torch.device(0 if torch.cuda.is_available() else 'cpu')

#ALL
def read_dfs(specifier=["_data"]*3):
    train_dfs= pd.read_pickle(f"./data/train{specifier[0]}.pickle")
    val_dfs= pd.read_pickle(f"./data/val{specifier[1]}.pickle")
    test_dfs= pd.read_pickle(f"./data/test{specifier[2]}.pickle")
    return train_dfs, val_dfs, test_dfs

#MLP
# CLASS AND FUNCTIONS FOR DATALOADER CREATION
class OCDataMLP(Dataset):
    # Dataset Class
    def __init__(self, input, output, index, means, stds, transform=None):
        self.input = input
        self.output = output
        self.transform = transform
        self.index = index
        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        z = self.index[idx]

        x = (x-self.means)/self.stds
        # If there is a transform, apply it here to your sample
        if self.transform is not None:
            x, y = self.transform(x,y)
        x = torch.tensor(x,dtype=torch.float32).to(device=device)           
        return x,y,z
    
def object_distance(map_layout, player, target):
    #Closest Object
    dist = len(map_layout)+len(map_layout[0])+1
    object_pos = None
    for row_idx in range(len(map_layout)):
        for col_idx in range(len(map_layout[0])):
            if map_layout[row_idx][col_idx]==target:
                temp_dist = abs(player[0]-col_idx)+abs(player[1]-row_idx)
                if temp_dist<dist:
                    object_pos = (col_idx,row_idx)
                    dist = temp_dist
                elif temp_dist==dist:
                    if np.random.rand()>0.5: #50-50s
                        object_pos = (col_idx,row_idx)
    return object_pos

def encode_mlp(dfs,step=1):
    #Fixed sized inputs for NN
    L = len(dfs)
    input_list = [] 
    output_list = [] 
    index_list = []
    
    # Processing into array for NN
    for i in range(0,L,step):
        row = dfs.iloc[i]
        ja = row['joint_action']
        ja = ja.replace("\'","\"")
        y_act = json.loads(ja)
        player_idx = row['player_idx']

        # Action / Output / GT
        y_row = y_act[player_idx]
        y=-1
        if y_row == 'INTERACT':
            y=0
        elif y_row[0]==1:
            y=1
        elif y_row[0]==-1:
            y=2
        elif y_row[1]==1:
            y=3
        elif y_row[1]==-1:
            y=4
        else:
            print("Unknown Action", y, i, player_idx, y_act)
            continue
        
        # Input Data 
        # 0 - P2 PosX (r)
        # 1 - P2 PosY (r)
        # 2 - P1 O1
        # 3 - P1 O2
        # 4 - P1 onion
        # 5 - P1 dish
        # 6 - P1 soup
        # 7 - P2 O1
        # 8 - P2 O2
        # 9 - P2 onion
        # 10 - P2 dish
        # 11 - P2 soup
        # 12 - TP PosX (r)
        # 13 - TP PosY (r)
        # 14 - TO PosX (r)
        # 15 - TO PosY (r)
        # 16 - TD PosX (r)
        # 17 - TD PosY (r)
        # 18 - TS PosX (r)
        # 19 - TS PosY (r)


        x = np.zeros(20)
        state = json.loads(row['state'])
        players = [state['players'][player_idx%2], state['players'][(player_idx+1)%2]]
        posX = players[0]['position'][0]
        posY = players[0]['position'][1]

        #Relative Position of P2
        x[0] = players[1]['position'][0] - posX
        x[1] = players[1]['position'][1] - posY

        for j in range(2):
            player = players[j]

            #Orientation
            x[2+2*j]=(player['orientation'][0])
            x[3+2*j]=(player['orientation'][1])

            #Held object converter
            if player["held_object"]!=None:
                if player["held_object"]['name'] == "onion":
                    x[6+3*j]=1
                elif player["held_object"]['name'] == "dish":
                    x[7+3*j]=1
                elif player["held_object"]['name'] == "soup":
                    x[8+3*j]=1
                else:
                    print("Unknown Item", player["held_object"]['name'])
        
        #World Data - 8 (2 per target)
        map_layout = row['layout']
        map_layout = map_layout.replace("\'","\"")
        map_layout = json.loads(map_layout)
        
        target_list = ['P','O','D','S']
        for target_idx in range(len(target_list)):
            target = target_list[target_idx]
            target_pos = object_distance(map_layout, (posX,posY), target)
            if target_pos is None:
                print("Not found error", target)
            x[12+2*target_idx]=(target_pos[0]-posX)
            x[13+2*target_idx]=(target_pos[1]-posY)
        
        
        input_list.append(x)
        output_list.append(y)
        index_list.append(row['Unnamed: 0'])

    return input_list, output_list, index_list

# TRAIN AND TEST TRANSFORMS MLP
def flip_mlp(x,y,axis=0):
    # Input Data 
    # 0 - P2 PosX (r)
    # 1 - P2 PosY (r)
    # 2 - P1 O1
    # 3 - P1 O2
    # 4 - P1 onion
    # 5 - P1 dish
    # 6 - P1 soup
    # 7 - P2 O1
    # 8 - P2 O2
    # 9 - P2 onion
    # 10 - P2 dish
    # 11 - P2 soup
    # 12 - TP PosX (r)
    # 13 - TP PosY (r)
    # 14 - TO PosX (r)
    # 15 - TO PosY (r)
    # 16 - TD PosX (r)
    # 17 - TD PosY (r)
    # 18 - TS PosX (r)
    # 19 - TS PosY (r)
    
    idx0_list = [0,12,14,16,18,2,7]

    for idx in idx0_list:
        x[idx+axis] = -x[idx+axis]
    
    if axis==0:
        if y==1:
            y=2
        elif y==2:
            y=1
    elif axis==1:
        if y==3:
            y=4
        elif y==4:
            y=3

    return x,y

def rotate_mlp(x,y):
    # Input Data 
    # 0 - P2 PosX (r)
    # 1 - P2 PosY (r)
    # 2 - P1 O1
    # 3 - P1 O2
    # 4 - P1 onion
    # 5 - P1 dish
    # 6 - P1 soup
    # 7 - P2 O1
    # 8 - P2 O2
    # 9 - P2 onion
    # 10 - P2 dish
    # 11 - P2 soup
    # 12 - TP PosX (r)
    # 13 - TP PosY (r)
    # 14 - TO PosX (r)
    # 15 - TO PosY (r)
    # 16 - TD PosX (r)
    # 17 - TD PosY (r)
    # 18 - TS PosX (r)
    # 19 - TS PosY (r)
    
    idx0_list = [0,12,14,16,18,2,7]

    for idx in idx0_list:
        x[idx],x[idx+1] = (x[idx+1],x[idx])
    
    if y==2:
        y=4
    elif y==4:
        y=2
    elif y==3:
        y=1
    elif y==1:
        y=3

    return x,y
    
def train_transform_mlp(x,y):
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    rand3 = np.random.rand()
    if rand1>0.5:
        x,y = flip_mlp(x,y,axis=0)
    if rand2>0.5:
        x,y = flip_mlp(x,y,axis=1)
    if rand3>0.5:
        x,y = rotate_mlp(x,y)
    return x,y

def make_data_mlp(train_dfs, val_dfs, test_dfs, BATCH_SIZE, num_workers=0, print_stats=False):
    #Training Dataloader
    input_list, output_list, index_list = encode_mlp(train_dfs)
    input_arr = np.array(input_list)
    train_means = np.mean(input_arr,axis=0)
    train_stds = np.std(input_arr,axis=0)

    idx0_list = [0,12,14,16,18,2,7]
    for idx in idx0_list:
        shared_mean = np.mean(input_arr[:,idx:idx+2])
        train_means[idx] = shared_mean
        train_means[idx+1] = shared_mean

        shared_std = np.std(input_arr[:,idx:idx+2])
        train_stds[idx] = shared_std
        train_stds[idx+1] = shared_std

    if print_stats:
        print(f"STD: {train_stds}\n Mean: {train_means}\n Input Shape: {input_arr.shape}")

    
    train_dataset = OCDataMLP(input_list,output_list,index_list, means = train_means, stds= train_stds,transform=train_transform_mlp)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    #Validation loader
    input_list, output_list, index_list = encode_mlp(val_dfs)
    val_dataset = OCDataMLP(input_list,output_list,index_list, means = train_means, stds= train_stds)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    #Test loader
    input_list, output_list, index_list = encode_mlp(test_dfs)
    test_dataset = OCDataMLP(input_list,output_list,index_list, means = train_means, stds= train_stds)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


#Encoded CNN
class OCDataEncodedCNN(Dataset):
    def __init__(self, input_list, output_list, z,  transform=False):
        self.input = input_list
        self.output = output_list
        self.transform = transform
        self.z = z
    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        z = self.z[idx]

        rand1 = np.random.rand()
        rand2 = np.random.rand()
        
        if self.transform:
            if rand1>0.5:
                # x = np.flip(x,1) #axis 1 is rows (swaps rows)
                x[13,:,:],x[14,:,:]= (x[14,:,:],x[13,:,:])
                x[17,:,:],x[18,:,:]= (x[18,:,:],x[17,:,:])

                if y==3:
                    y=4
                elif y==4:
                    y=3
                    
            if rand2>0.5:
                # x = np.flip(x,2) #axis 2 is cols (swaps cols)
                x[12,:,:],x[11,:,:]= (x[11,:,:],x[12,:,:])
                x[15,:,:],x[16,:,:]= (x[16,:,:],x[15,:,:])
                if y==1:
                    y=2
                elif y==2:
                    y=1

        x = torch.tensor(x.copy(),dtype=torch.float32).to(device=device)

        if self.transform:
            if rand1>0.5:
                    x = transforms.functional.vflip(x)
            if rand2>0.5:
                x = transforms.functional.hflip(x)
        return x,y,z

def encode_cnn(dfs,step=1):
    
    L = len(dfs)
    input_list = [] 
    output_list = [] 
    index_list = []
    
    target_list = ['X','P','O','D','S']
    # Processing into array for NN
    for i in range(0,L,step):
        row = dfs.iloc[i]
        ja = row['joint_action']
        ja = ja.replace("\'","\"")
        y_act = json.loads(ja)
        player_idx = row['player_idx']

        # Action / Output / GT
        y_row = y_act[player_idx]
        y=-1
        if y_row == 'INTERACT':
            y=0
        elif y_row[0]==1:
            y=1
        elif y_row[0]==-1:
            y=2
        elif y_row[1]==1:
            y=3
        elif y_row[1]==-1:
            y=4
        else:
            print("Unknown Action", y, i, player_idx, y_act)
            continue
        
        map_layout = row['layout']
        map_layout = map_layout.replace("\'","\"")
        map_layout = json.loads(map_layout)

        x = np.zeros([19,5,9]) #biggest map size
        pot_locations = []
        state = json.loads(row['state'])
        

        ## partial 8-10, 11-14

        for player in [state['players'][player_idx], state['players'][(player_idx+1)%2]]:
            #Position
            player_pos = (player['position'][0],player['position'][1])
            #Orientation
            player_or = (player['orientation'][0],player['orientation'][1])

            or_idx = 11
            if player_idx==1:
                or_idx = 15
            
            if player_or[0]==-1:
                x[or_idx, player_pos[1],player_pos[0]] = 1
            elif player_or[0]==1:
                x[or_idx+1, player_pos[1],player_pos[0]] = 1
            elif player_or[1]==-1:
                x[or_idx+2, player_pos[1],player_pos[0]] = 1
            elif player_or[1]==1:
                x[or_idx+3, player_pos[1],player_pos[0]] = 1

            #Held object converter
            if player["held_object"] is None:
                pass
            else:
                if player["held_object"]['name'] == "onion":
                    x[8, player_pos[1],player_pos[0]] = 1
                elif player["held_object"]['name'] == "dish":
                    x[9, player_pos[1],player_pos[0]] = 1
                elif player["held_object"]['name'] == "soup":
                    x[10, player_pos[1],player_pos[0]] = 1
                else:
                    print("unknown held item", player["held_object"]['name'])

        ## 0-5
        for map_row_idx in range(len(map_layout)):
            for map_col_idx in range(len(map_layout[0])):
                # 0
                x[0,map_row_idx,map_col_idx]=1

                # 1-5
                tile = map_layout[map_row_idx][map_col_idx]
                for target_idx in range(len(target_list)):
                    if target_list[target_idx] == tile:
                        x[target_idx+1, map_row_idx,map_col_idx] = 1

        ## partial 8-10, 6-7
        object_list = state['objects']
        for obj in object_list:
            obj_name = obj['name']
            obj_pos = obj['position']

            if obj_name == "onion":
                x[8, obj_pos[1],obj_pos[0]] = 1
            elif obj_name == "dish":
                x[9, obj_pos[1],obj_pos[0]] = 1
            elif obj_name == "soup":
                x[6, obj_pos[1],obj_pos[0]] = len(obj['_ingredients'])/3
                x[10, obj_pos[1],obj_pos[0]] = 1
                if obj['is_ready']:
                    x[7, obj_pos[1],obj_pos[0]] = obj['cooking_tick']/20
        
        input_list.append(x)
        output_list.append(y)
        index_list.append(row['Unnamed: 0'])

    return input_list, output_list, index_list
    
def make_data_encoded_cnn(train_dfs, val_dfs, test_dfs, BATCH_SIZE, num_workers=0):
    #Training Dataloader
    input_list, output_list, index_list = encode_cnn(train_dfs)
    # means, stds = norm_stats(input_list)

    train_dataset = OCDataEncodedCNN(input_list,output_list,index_list, transform=False) #Better performance with transform off
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    #Validation loader
    input_list, output_list, index_list = encode_cnn(val_dfs)
    val_dataset = OCDataEncodedCNN(input_list,output_list,index_list)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    #Test loader
    input_list, output_list, index_list = encode_cnn(test_dfs)
    test_dataset = OCDataEncodedCNN(input_list,output_list,index_list)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader



















