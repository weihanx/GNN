import logging
from pathlib import Path
import sys

from tqdm import tqdm

from config import *
from torch_geometric.transforms import Pad
from torch_geometric.loader import DataLoader
import dataset_heter
import torch
import matplotlib.pyplot as plt
import numpy as np
# from config import MatrixNormLoss
TRAIN_PERCENT = 0.8
BATCH_SIZE = 1

LEARNING_RATE = 0.001
SCHEDULER_GAMMA = 0.1
MAX_GRAD_NORM = 0.1


def start_logger():
    logging.getLogger('matplotlib.font_manager').disabled = True
    out_dict = Path(f"{ROOT_PATH}/final_code/Logging")
    out_dict.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            logging.FileHandler(out_dict / "key_pred_sage.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def prepare_data_loaders(train_names, save_folder, transform=None, dataset_class=dataset_heter.HeterGraph):
    with open(train_names, "r") as file:
        train_names = file.readlines()
    train_names = [line.strip() for line in train_names]

    dataset = dataset_class(
        root= save_folder,
        train_names=train_names,
        transform=None, # no need for padding
        pre_transform=None
    )

    train_size = int(len(dataset) * TRAIN_PERCENT )  # 5200
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, valid_loader


def prepare_model(num_features,embedding_dim, hidden_dim, num_class, model_class, device):
    model = model_class(num_features, embedding_dim, hidden_dim, num_class, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # adamW
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)
    return model, optimizer, scheduler


def calculate_accuracy(output, labels):
    predictions = output.argmax(dim=1)  # Get the index of the highest value
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def train_loop(model, optimizer, loader, device):
    print(f"model device = {model.DEVICE}")
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batch in tqdm(loader, desc="training"):
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict, batch['note'].batch)
        loss = CLASSIFICATION_CRITERION(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += calculate_accuracy(out, batch.y)
    average_loss = total_loss / len(loader)
    average_accuracy = total_accuracy / len(loader)
    return average_loss, average_accuracy


def validation_loop(model, loader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="validation"):
            batch.to(device)  # Move the batch to the specified device
            out = model(batch.x_dict, batch.edge_index_dict, batch['note'].batch)
            loss = CLASSIFICATION_CRITERION(out, batch.y)
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(out, batch.y)
    average_loss = total_loss / len(loader)
    average_accuracy = total_accuracy / len(loader)
    return average_loss, average_accuracy

def train_loop_sck(model, optimizer, loader, device):
    print(f"model device = {model.DEVICE}")
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batch in tqdm(loader, desc="training"):
        batch.to(device)
        optimizer.zero_grad()
        out, s1_pred, s2_pred = model(batch.x_dict, batch.edge_index_dict, batch['note'].batch)
        s_true = torch.matmul(batch.s1, batch.s2) # clustering matrix
        s_out = torch.matmul(s1_pred, s2_pred) # clustering matrix
        # print(f"shap of s1, s2 = {s1_pred.shape}, {s2_pred.shape}, output shape = {s_out.shape}")
        loss = CLASSIFICATION_CRITERION(out, batch.y) + SIM_CRITERION(s1_pred, batch.s1) + SIM_CRITERION(s2_pred, batch.s2) + SIM_CRITERION(s_out, s_true)
        # loss = CLASSIFICATION_CRITERION(out, batch.y) + SIM_CRITERION(s1_pred, s1_pred) + SIM_CRITERION(s2_pred, s2_pred) + SIM_CRITERION(s_out, s_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += calculate_accuracy(out, batch.y)
    average_loss = total_loss / len(loader)
    average_accuracy = total_accuracy / len(loader)
    return average_loss, average_accuracy


def validation_loop_sck(model, loader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="validation"):
            batch.to(device)  # Move the batch to the specified device: 100->50->1: 100*100, 50*50
            ### batch.s1, batch.s2: all clustering matrix, square clustering matrix gives me similairty martix
            out, s1_pred, s2_pred = model(batch.x_dict, batch.edge_index_dict, batch['note'].batch)
            s_true = torch.matmul(batch.s1, batch.s2)
            s_out = torch.matmul(s1_pred, s2_pred)
            # test whether the loss is doable:
            loss = CLASSIFICATION_CRITERION(out, batch.y) + SIM_CRITERION(s1_pred, s1_pred) + SIM_CRITERION(s2_pred, s2_pred) + SIM_CRITERION(s_out, s_out)
            # loss = CLASSIFICATION_CRITERION(out, batch.y) + SIM_CRITERION(s1_pred, batch.s1) + SIM_CRITERION(s2_pred, batch.s2) + SIM_CRITERION(s_out, s_true)
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(out, batch.y)
    average_loss = total_loss / len(loader)
    average_accuracy = total_accuracy / len(loader)
    return average_loss, average_accuracy



def plot_metrics(
        train_loss_curve,
        valid_loss_curve,
        train_acc_curve,
        valid_acc_curve,
        save_name=None
):
    epochs = np.arange(NUM_EPOCHS)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Loss Curve and Acc Curve for batch size = 1 with SAGE', fontsize=16, y=1.02)

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
    if save_name:
        plt.savefig(save_name)


