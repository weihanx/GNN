from torch_geometric.transforms.pad import EdgeTypePadding, NodeTypePadding
import torch
from torch_geometric.data import Data
import numpy as np
import time
import pathlib
from pathlib import Path
import torch.nn.functional as F
from config import *
from torch_geometric.typing import EdgeType, NodeType
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys
import Sckgroupmat
import data_processing
from math import ceil
from torch_geometric.nn.norm import LayerNorm
import matplotlib.pyplot as plt
from pyScoreParser.musicxml_parser.mxp import MusicXMLDocument
from torch_geometric.nn import dense_diff_pool
from torch.nn.parallel import DistributedDataParallel as DDP # parallel
import pyScoreParser.score_as_graph as score_graph
import pickle
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.transforms import Pad
import csv
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Dataset, HeteroData
import logging
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch_geometric.transforms as T
from data_processing import HeterGraph
from utils import start_logger, prepare_data_loaders, \
    prepare_model, plot_metrics, train_loop, validation_loop

"""
Dataset Loading
"""


start_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRIAN_NAMES = "train-names.txt"
SAVE_FOLDER = "hetergraph0509_newfeat/"
CAT_CARDINALITIES = [35, 58, 17]
NUM_CONT_FEAT = 111
EMB_DIM = 64
HIDDEN_DIM = 512
NUM_CLASS = 15
dataset_class = data_processing.HeterGraph
train_loader, valid_loader = prepare_data_loaders(TRIAN_NAMES, SAVE_FOLDER, dataset_class)
print(f" next(iter(train_loader)) = { next(iter(train_loader))}")
NUM_FEAT = next(iter(train_loader))['data']['note'].x.shape[1] 
model, optimizer, scheduler = prepare_model(NUM_FEAT,EMB_DIM, HIDDEN_DIM, NUM_CLASS, model_class=Sckgroupmat.GroupMat, device = device)
optimizer = torch.optim.Adam([
    {'params': model.gnn_cluster1.parameters(), 'lr': 0.001, 'name': 'cluster1'}
])
model.to(device)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x_dict, data.edge_index_dict,data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum()) 
    return correct / len(loader.dataset) 



def train_sck(model, loss_fn, train_loader, num_layer):
    logging.debug(f"Training...")
    train_loss = []
    count = 0
    correct = 0
    model.train()
    patience = 1
    num_layers = 1
    prev_loss = float('inf') 
    for databatch in train_loader:
        count = count + 1
        # filename = Path(databatch['name'][0])
        data = databatch['data']
        sck_mat_tuple = databatch['cluster']
        data = data.to(device)
        final_emb, s1_pred, s2_pred, tri_M_G = model(data)
        # print(f"s1_pred = {s1_pred}, s2_pred = {s2_pred}")
        # s1_pred_np = s1_pred.detach().cpu().numpy()
        # s2_pred_np = s2_pred.detach().cpu().numpy()
        final_emb = final_emb.detach().cpu().numpy()
        # true_label = data.y
        # y_np = true_label.cpu().detach().numpy()
        # filename.parent.mkdir(parents=True, exist_ok=True)
        # print(f"s1_pred_np = {s1_pred_np}")
        loss_not_decreasing_count = 0
        # Save the numpy arrays. Adjust the filenames as needed.
        # np.save(filename.with_name(f"{filename.stem}_s1_pred_weighted_{class_weight[0]}_{class_weight[1]}.npy"), s1_pred_np)
        # np.save(filename.with_name(f"{filename.stem}_s2_pred_weighted_{class_weight[0]}_{class_weight[1]}.npy"), s2_pred_np)
        # np.save(filename.with_name(f"{filename.stem}_training_sck_out_weighted_{class_weight[0]}_{class_weight[1]}.npy"), final_emb)
        # np.save(filename.with_name(f"{filename.stem}_true_train_label_weighted_{class_weight[0]}_{class_weight[1]}.npy"), y_np)
        cluster1 = sck_mat_tuple[0][0].to(device)
        s1 = torch.matmul(cluster1, cluster1.t())
        s1_pred = torch.matmul(s1_pred, s1_pred.t())
        # print(f"shape of tri_M_G = {tri_M_G}, s1 = {s1}")
        # print(f"tri_M_G = {tri_M_G}")
        loss2 = loss_fn(tri_M_G.squeeze(), s1.squeeze()) 
        
        s1_pred = s1_pred.to(device)
        


        # loss3 = loss_fn(s2_pred.float().unsqueeze(0), s2.float().unsqueeze(0))




        if num_layers == 1:
            loss = loss2
            sum_loss = loss2
            # print(f"one layer loss = {loss}")
        else:
            cluster2 = sck_mat_tuple[1][0]
            s2 = torch.matmul(cluster1, cluster2).to(device)
            s2 = torch.matmul(s2, s2.t())
            s2 = s2.to(device)  
            s2_pred = torch.matmul(s2_pred, s2_pred.t())
            # print(f"shape of tri_M_G = {tri_M_G.shape}, s1 = {s2.shape}")
            loss3 = loss_fn(tri_M_G.squeeze(), s2.squeeze()) # directly compare with grouping matrix
            loss = loss3 
            sum_loss = loss3 + loss2 
        train_loss.append(sum_loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        # _, predicted_labels = torch.max(out, 1)

        # correct += (predicted_labels == data.y).sum().item()
        # train_acc = correct / count
    return np.mean(train_loss), num_layer # average loss for this epoch


def validate_sck(model, loss_fn, valid_loader, class_weight):
    logging.debug(f"Validating...")
    model.eval()
    with torch.no_grad():
        correct = 0
        val_loss = []
        count = 0
        for databatch in valid_loader:
            filename = Path(databatch['name'][0])
            data = databatch['data']
            sck_mat_tuple = databatch['cluster']
            data = data.to(device)
            final_emb, s1_pred, s2_pred = model(data)

            s1_pred_np = s1_pred.detach().cpu().numpy()
            s2_pred_np = s2_pred.detach().cpu().numpy()
            final_emb = final_emb.detach().cpu().numpy()
            true_label = data.y
            y_np = true_label.cpu().detach().numpy()
            filename.parent.mkdir(parents=True, exist_ok=True)

            np.save(filename.with_name(f"{filename.stem}_s1_pred_weighted_{class_weight[0]}_{class_weight[1]}.npy"), s1_pred_np)
            np.save(filename.with_name(f"{filename.stem}_s2_pred_weighted_{class_weight[0]}_{class_weight[1]}.npy"), s2_pred_np)
            np.save(filename.with_name(f"{filename.stem}_valid_sck_out_weighted_{class_weight[0]}_{class_weight[1]}.npy"), final_emb)
            np.save(filename.with_name(f"{filename.stem}_true_valid_label_weighted_{class_weight[0]}_{class_weight[1]}.npy"), y_np)
            cluster1 = sck_mat_tuple[0][0]# (1,8,5), (1,5,4): it has batch dimension
            cluster2 = sck_mat_tuple[1][0]
            s1 = torch.matmul(cluster1, cluster1.t())
            s2 = torch.matmul(cluster1, cluster2)
            s2 = torch.matmul(s2, s2.t())
            s1 = s1.to(device)
            s2 = s2.to(device)

            s1_pred = torch.matmul(s1_pred, s1_pred.t()) # cluster to similarity
            s2_pred =  torch.matmul(s2_pred, s2_pred.t())# cluster to similarity
            # loss2 = loss_fn(s1_pred.float().unsqueeze(0), s1.float().unsqueeze(0))
            # loss3 = loss_fn(s2_pred.float().unsqueeze(0), s2.float().unsqueeze(0))
            
            loss2 = F.cosine_similarity(s1_pred.reshape(1, -1), s1.reshape(1, -1))
            loss3 = F.cosine_similarity(s2_pred.reshape(1, -1), s2.reshape(1, -1))
            loss = class_weight[0] * loss2 + class_weight[1] * loss3
            count += 1
            val_loss.append(loss.item())
            # _, predicted_labels = torch.max(out, 1)

            # correct += (predicted_labels == data.y).sum().item()

        # val_acc = correct / count
    return np.mean(val_loss)


def train_nosck(model, loss_fn, train_loader):
    # should be key prediction loss
    logging.debug(f"Training...")
    train_loss = []
    count = 0
    for databatch in train_loader:
        count = count + 1
        filename = Path(databatch['name'][0])
        data = databatch['data']

        data = data.to(device)
        final_emb, out, s1_pred, s2_pred = model(data.x_dict, data.edge_index_dict, data['note'].batch)
        
        s1_pred_np = s1_pred.detach().cpu().numpy()
        s2_pred_np = s2_pred.detach().cpu().numpy()
        final_emb = final_emb.detach().cpu().numpy()
        true_label = data.y
        y_np = true_label.cpu().detach().numpy()
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Save the numpy arrays. Adjust the filenames as needed.
        np.save(filename.with_name(f"{filename.stem}_training_out_nosck.npy"), final_emb)
        np.save(filename.with_name(f"{filename.stem}_true_train_label.npy"), y_np)
        np.save(filename.with_name(f"{filename.stem}_s1_pred_train_nosck.npy"), s1_pred_np)
        np.save(filename.with_name(f"{filename.stem}_s2_pred_train_nosck.npy"), s2_pred_np)

        loss = loss_fn(out, data.y)

        train_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        _, predicted_labels = torch.max(out, 1)

        correct += (predicted_labels == data.y).sum().item()
        train_acc = correct / count
    return np.mean(train_loss),train_acc # average loss for this epoch

def validate_nosck(model, loss_fn, valid_loader):
    logging.debug(f"Validating...")
    model.eval()
    with torch.no_grad():
        correct = 0
        val_loss = []
        count = 0
        for databatch in valid_loader:
            count += 1
            filename = Path(databatch['name'][0])
            data = databatch['data']

            data = data.to(device)
            final_emb, out, s1_pred, s2_pred = model(data.x_dict, data.edge_index_dict, data['note'].batch)

            s1_pred_np = s1_pred.detach().cpu().numpy()
            s2_pred_np = s2_pred.detach().cpu().numpy()
            final_emb = final_emb.detach().cpu().numpy()
            true_label = data.y
            y_np = true_label.cpu().detach().numpy()
            filename.parent.mkdir(parents=True, exist_ok=True)

            np.save(filename.with_name(f"{filename.stem}_valid_out_nosck.npy"), final_emb)
            np.save(filename.with_name(f"{filename.stem}_true_valid_label.npy"), y_np)
            np.save(filename.with_name(f"{filename.stem}_s1_pred_valid_nosck.npy"), s1_pred_np)
            np.save(filename.with_name(f"{filename.stem}_s2_pred_valid_nosck.npy"), s2_pred_np)
            
            loss = loss_fn(out, data.y)
            
            val_loss.append(loss.item())
            _, predicted_labels = torch.max(out, 1)

            correct += (predicted_labels == data.y).sum().item()

        val_acc = correct / count

    return np.mean(val_loss), val_acc


# Assuming your model, optimizer, and loss function are defined
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training and validation loop
num_epochs = 500
train_loss_curve = []
valid_loss_curve = []
train_acc_curve = []
valid_acc_curve = []
loss_fn = torch.nn.BCELoss()    
# CLASS_WEIGHT = [0.5, 0.5]
num_layer = 1
prev_loss = float('inf')
train_loss = 0 
for epoch in range(num_epochs):
    train_loss, num_layer = train_sck(model, loss_fn, train_loader, num_layer)
    if train_loss < prev_loss and num_layer == 1: # training loss is decreasing
        print(f"one layer loss = {train_loss}")
        prev_loss = train_loss
    elif train_loss > prev_loss and num_layer == 1: # training loss is not decreasing, unfreeze second layer
        num_layer += 1
        for param in model.gnn_cluster2.parameters():
            param.requires_grad_ = True  
        model.isfreeze == False
        # Update the learning rate for the first layer to a very small value
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'cluster1':
                param_group['lr'] = 1e-10  # Set a new learning rate
        # for param in model.gnn_cluster1.parameters():
        #     param.requires_grad_ = False   
        # Add the second layer's parameters to the optimizer if not already included
        optimizer.add_param_group({'params': model.gnn_cluster2.parameters(), 'lr': 0.001})
        print(f"two layer loss = {train_loss}")
    else: # num layer ==2 
        print(f"two layer loss = {train_loss}")
    # valid_loss = validate_sck(model, loss_fn, valid_loader, CLASS_WEIGHT)
    # print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}')
    # scheduler.step()
    train_loss_curve.append(train_loss)

    # valid_loss_curve.append(valid_loss)
    # train_acc_curve.append(train_accuracy)
    # valid_acc_curve.append(valid_accuracy)
logging.info(f"Plotting loss and acc curve")
epochs = np.arange(0, 50, dtype=int)

fig, axs = plt.subplots(2, 1, figsize=(12, 12))
fig.suptitle('Loss Curve and Acc Curve', fontsize=16, y=1.02)

# Loss curves
axs[0].plot(epochs, train_loss_curve, label='Training Loss')
axs[0].plot(epochs, valid_loss_curve, label='Validation Loss')
axs[0].set_xlabel('Epoch Number')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss Curves')
axs[0].legend()

# Accuracy curves
axs[1].plot(epochs, train_acc_curve, label='Training Accuracy')
axs[1].plot(epochs, valid_acc_curve, label='Validation Accuracy')
axs[1].set_xlabel('Epoch Number')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Accuracy Curves')
axs[1].legend()

plt.tight_layout()
plt.savefig("SAGE_bsz1.png")
